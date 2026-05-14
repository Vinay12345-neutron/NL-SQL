#!/usr/bin/env python3
"""
Phase 3: Execution-Guided Cascade (Three-Tiered Confidence Cascade)
====================================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

Architecture:
    Tier 1  → Qwen Dense Retrieval (Top-15 candidates)
    Tier 2  → RoBERTa Cross-Encoder scores + MLP routing decision
    Tier 3  → SQL execution cascade for ambiguous queries

Routing Logic:
    MLP label = 0 (Unambiguous) → output Cross-Encoder Top-1 directly
    MLP label = 1 (Ambiguous)   → execution cascade:
        a) Generate SQL for Top-3 CE-ranked candidate DBs (DeepSeek V3)
        b) Execute each SQL against local SQLite files
        c) Single POPULATED result → pick that DB
        d) Multiple POPULATED or all non-POPULATED → Gemini judge

Input:
    data/spider_mlp_training_data.csv     — MLP labels + scores per query
    results/spider_baseline2_scores.json  — CE scores + ranked DBs per query
    models/spider_ambiguity_mlp/          — Trained MLP + scaler

Output:
    results/spider_cascade_results.json   — Per-query routing decisions + final DB
    
Metrics Reported:
    R@1 of the full cascade vs Baseline 1 (Qwen) and Baseline 2 (RoBERTa CE)
"""

import os
import re
import json
import time
import pickle
import asyncio
import sqlite3
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.jsonl")
CSV_PATH    = os.path.join(BASE_DIR, "data",    "spider_mlp_training_data.csv")
MLP_DIR     = os.path.join(BASE_DIR, "models",  "spider_ambiguity_mlp")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "spider_cascade_results.jsonl")
TOP15_PATH  = os.path.join(BASE_DIR, "data",    "spider_baseline1_top15.json")

# Tables paths for schema loading
TABLES_PATHS = [
    os.path.join(BASE_DIR, "processed_data", "spider_tables.json"),
    os.path.join(BASE_DIR, "processed_data", "spider_test_tables.json"),
]

# API models (matching spider_misclassified_pipeline.py)
SQL_MODEL    = "deepseek/deepseek-chat-v3.1"   # SQL generation
JUDGE_MODEL  = "google/gemini-2.0-flash-001"   # Final judge

MAX_COMPLETION_TOKENS = 1024
SQLITE_TIMEOUT        = 5.0
ASYNC_SEMAPHORE       = 3                       # Concurrent async SQL gen calls
TOP_K_EXEC            = 8                       # Expanded Top-K candidates for execution
BACKOFF_BASE          = 10

# Execution status constants
EXEC_ERROR          = "ERROR"
EXEC_VALID_STRUCTURE = "VALID_STRUCTURE"
EXEC_POPULATED      = "POPULATED"


# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

or_sync  = OpenAI(api_key=OPENROUTER_API_KEY,  base_url="https://openrouter.ai/api/v1")
or_async = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


# ---------------------------------------------------------------------------
# MLP Model (must match 2b_train_mlp.py)
# ---------------------------------------------------------------------------
class AmbiguityMLP(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def build_schema_context(db_entry: dict) -> str:
    """Same as Phase 1A/1C — compact schema context."""
    db_id       = db_entry["db_id"]
    table_names = db_entry["table_names_original"]
    col_names   = db_entry["column_names_original"]
    table_cols  = {i: [] for i in range(len(table_names))}
    for tidx, cname in col_names:
        if tidx != -1:
            table_cols[tidx].append(cname)
    lines = [f"  - {t}({', '.join(table_cols.get(i, [])[:8])})"
             for i, t in enumerate(table_names)]
    return f"Database: {db_id}\nTables:\n" + "\n".join(lines)


def get_full_ddl(db_id: str, schemas: dict) -> str:
    """Return DDL-style schema string for SQL generation prompt."""
    ctx = schemas.get(db_id, "")
    lines = []
    for line in ctx.splitlines():
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
            paren = line.find("(")
            if paren > 0:
                tbl  = line[:paren]
                cols = line[paren+1:].rstrip(")")
                lines.append(f"CREATE TABLE {tbl} ({cols});")
    return "\n".join(lines) if lines else ctx


def get_db_path(db_id: str) -> Optional[str]:
    """Find the .sqlite file for a given db_id."""
    candidates = [
        os.path.join(BASE_DIR, "spider_data", "database",     db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "spider_data", "test_database", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "dev_20240627", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "train", "train_databases",     db_id, f"{db_id}.sqlite"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*',          '', text, flags=re.DOTALL)
    return text.strip()


def call_llm_sync(model: str, prompt: str,
                  system: str = "You are a database expert.",
                  temperature: float = 0.0) -> str:
    attempt = 0
    while True:
        try:
            resp = or_sync.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_COMPLETION_TOKENS,
                timeout=60.0,
            )
            return strip_think(resp.choices[0].message.content or "")
        except Exception as e:
            wait = min(300, BACKOFF_BASE * (2 ** attempt))
            print(f"  [Sync Error] sleeping {wait}s... ({e})")
            time.sleep(wait)
            attempt += 1


async def call_llm_async(semaphore: asyncio.Semaphore, model: str,
                          prompt: str, system: str = "You are a SQL expert.",
                          temperature: float = 0.1) -> str:
    async with semaphore:
        attempt = 0
        while True:
            try:
                resp = await or_async.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": prompt}],
                    temperature=temperature,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    timeout=60.0,
                )
                return strip_think(resp.choices[0].message.content or "")
            except Exception as e:
                wait = min(300, BACKOFF_BASE * (2 ** attempt))
                await asyncio.sleep(wait)
                attempt += 1


# ---------------------------------------------------------------------------
# SQL Generation (DeepSeek V3)
# ---------------------------------------------------------------------------
def build_sql_prompt(query: str, db_id: str, ddl: str) -> str:
    return f"""You are a SQL expert. Write ONE SQLite-compatible SQL query.

Database: {db_id}
Schema:
{ddl}

Question: {query}

Rules:
- ONE complete SQL query only
- Use ONLY tables/columns from the schema above
- SQLite syntax (LIMIT not TOP)
- Output ONLY the SQL. No explanation, no markdown, no backticks."""


async def generate_sql_for_candidates(
    query: str, candidates: List[str], schemas: dict
) -> Dict[str, str]:
    semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE)

    async def gen_one(db_id: str) -> Tuple[str, str]:
        ddl    = get_full_ddl(db_id, schemas)
        prompt = build_sql_prompt(query, db_id, ddl)
        sql    = await call_llm_async(semaphore, SQL_MODEL, prompt,
                                      system="You are a SQL expert. Output ONLY SQL.")
        sql = re.sub(r'^```\w*\n?', '', sql.strip())
        sql = re.sub(r'\n?```$',    '', sql.strip())
        sql = sql.strip().rstrip(';') + ';' if sql.strip() else ""
        return db_id, sql

    results = await asyncio.gather(*[gen_one(db) for db in candidates],
                                   return_exceptions=True)
    sql_map = {}
    for r in results:
        if not isinstance(r, Exception):
            db_id, sql = r
            sql_map[db_id] = sql
    return sql_map


# ---------------------------------------------------------------------------
# SQL Execution
# ---------------------------------------------------------------------------
def execute_sql_map(sql_map: Dict[str, str]) -> Dict[str, dict]:
    results = {}
    for db_id, sql in sql_map.items():
        db_path = get_db_path(db_id)
        entry   = {"sql": sql, "status": EXEC_ERROR, "result": None,
                   "error": None, "db_path": db_path}

        if not sql:
            entry["error"] = "No SQL generated"
            results[db_id] = entry
            continue
        if not db_path:
            entry["error"] = f"SQLite not found: {db_id}"
            results[db_id] = entry
            continue

        try:
            conn   = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
            cursor = conn.cursor()
            timer  = threading.Timer(SQLITE_TIMEOUT, conn.interrupt)
            timer.start()
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
            finally:
                timer.cancel()
            conn.close()

            if not rows:
                entry["status"] = EXEC_VALID_STRUCTURE
                entry["result"] = []
            elif len(rows) == 1 and len(rows[0]) == 1:
                val = rows[0][0]
                entry["status"] = EXEC_VALID_STRUCTURE if (val == 0 or val is None) else EXEC_POPULATED
                entry["result"] = rows[:5]
            else:
                entry["status"] = EXEC_POPULATED
                entry["result"] = rows[:5]
        except Exception as e:
            entry["error"]  = str(e)
            entry["status"] = EXEC_ERROR

        results[db_id] = entry
    return results


# ---------------------------------------------------------------------------
# Gemini Judge
# ---------------------------------------------------------------------------
def judge_with_gemini(query: str, exec_results: Dict[str, dict],
                      schemas: dict) -> str:
    candidates = list(exec_results.keys())
    if len(candidates) == 1:
        return candidates[0]

    evidence_lines = []
    for db_id, r in exec_results.items():
        evidence_lines.append(
            f"Database: {db_id}\n"
            f"  SQL: {r['sql']}\n"
            f"  Status: {r['status']}\n"
            f"  Error: {r['error'] or 'None'}"
        )

    prompt = f"""You are a database routing judge for the Spider dataset.

CRITICAL CONTEXT: Spider has many empty databases. VALID_STRUCTURE means the SQL ran fine but the DB is empty. Do NOT prefer POPULATED over VALID_STRUCTURE automatically — prefer the DB whose schema best fits the question.

Decision Rules:
1. ELIMINATE databases with ERROR status (schema mismatch).
2. Pick the DB whose SQL uses the most dedicated, exact table/column names matching the question.
3. Break ties by structural schema fit, not by whether rows were returned.

Question: {query}

Execution Evidence:
{chr(10).join(evidence_lines)}

On the final line: <FINAL_DB>database_name</FINAL_DB>"""

    response = call_llm_sync(JUDGE_MODEL, prompt)
    match    = re.search(r'<FINAL_DB>\s*([\w_]+)\s*</FINAL_DB>', response)
    if match:
        return match.group(1).strip()
    # Fallback: return first non-error DB
    for db_id, r in exec_results.items():
        if r["status"] != EXEC_ERROR:
            return db_id
    return candidates[0]


# ---------------------------------------------------------------------------
# Execution Cascade (for one ambiguous query)
# ---------------------------------------------------------------------------
def execution_cascade(query: str, ce_ranked_dbs: List[str],
                      schemas: dict) -> Tuple[str, dict]:
    """Run SQL execution on Top-K CE candidates. Return (predicted_db, details)."""
    candidates = [db for db in ce_ranked_dbs[:TOP_K_EXEC] if db in schemas]
    if not candidates:
        return ce_ranked_dbs[0] if ce_ranked_dbs else "", {"tier3_path": "fallback_no_schema"}

    # Execute SQL (async)
    sql_map   = asyncio.run(generate_sql_for_candidates(query, candidates, schemas))
    exec_res  = execute_sql_map(sql_map)

    populated = [db for db, r in exec_res.items() if r["status"] == EXEC_POPULATED]

    if len(populated) == 1:
        # Single clear winner
        return populated[0], {"tier3_path": "single_populated", "exec_results": {
            db: r["status"] for db, r in exec_res.items()
        }}
    else:
        # Multiple populated or all failed → Gemini judge
        predicted = judge_with_gemini(query, exec_res, schemas)
        return predicted, {"tier3_path": "gemini_judge", "exec_results": {
            db: r["status"] for db, r in exec_res.items()
        }}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load CE scores
   # 1. Load LLM Tier 2 scores
    # 1. Load LLM Tier 2 scores (Bulletproof JSONL Parser)
    ce_records = []
    print(f"Loading CE scores from {SCORES_PATH}...")
    with open(SCORES_PATH, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines or stray JSON array brackets
            if not line or line in ('[', ']'):
                continue
            # Remove trailing commas if they accidentally exist
            if line.endswith(','):
                line = line[:-1]
                
            try:
                ce_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f" [Warning] Skipping malformed JSON on line {line_num}: {e}")

    ce_map = {r["question"]: r for r in ce_records}
    print(f"Successfully loaded {len(ce_records)} records.")

    # 2. Load Top-15 (Qwen) for Tier 1 top-1 and Tier 3 candidate pool
    with open(TOP15_PATH) as f:
        top15_data = json.load(f)
    qwen_top1_map = {item["question"]: item["top15_dbs"][0] for item in top15_data}
    qwen_top15_map = {item["question"]: item["top15_dbs"] for item in top15_data}

    # 3. Load CSV labels (from Phase 2A)
    df = pd.read_csv(CSV_PATH)
    score_cols = [f"s{i+1}" for i in range(15)]

    # 4. Load MLP + scaler
    print("Loading MLP ambiguity detector...")
    with open(os.path.join(MLP_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    mlp = AmbiguityMLP()
    mlp.load_state_dict(torch.load(os.path.join(MLP_DIR, "mlp.pt"), weights_only=True))
    mlp.eval()

    X_all  = scaler.transform(df[score_cols].values.astype(np.float32))
    Xt_all = torch.tensor(X_all, dtype=torch.float32)
    with torch.no_grad():
        mlp_probs  = mlp(Xt_all).numpy().flatten()
    mlp_labels = (mlp_probs >= 0.5).astype(int)
    print(f"  Unambiguous: {(mlp_labels==0).sum()} | Ambiguous: {(mlp_labels==1).sum()}\n")

    # 5. Load schemas
    print("Loading schemas...")
    all_schemas = {}
    for path in TABLES_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                for entry in json.load(f):
                    all_schemas[entry["db_id"]] = build_schema_context(entry)
    print(f"  Loaded {len(all_schemas)} schemas.\n")

    # 6. Cascade routing
    results       = []
    correct_count = 0

    ambiguous_qs   = df[mlp_labels == 1]["gold_db"].index.tolist()
    unambiguous_qs = df[mlp_labels == 0]["gold_db"].index.tolist()

    import concurrent.futures
    import threading

    existing_records = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_records.add(rec["question"])
    if existing_records:
        print(f"Resuming: skipping {len(existing_records)} already processed queries.")

    def process_query(idx, question, gold_db, ce_ranked, label, mlp_prob):
        if label == 0:
            pred = ce_ranked[0] if ce_ranked else qwen_top1_map.get(question, "")
            tier = 2
            details = {}
        else:
            # The LLM Reranker was confused (low margin). Its Top-5 is compromised.
            # Fall back to the Qwen dense retrieval candidates which have much higher recall!
            qwen_ranked = qwen_top15_map.get(question, ce_ranked)
            
            # Merge CE's Top 3 with Qwen's Top 5 to create a robust candidate pool (up to 8 candidates)
            merged_candidates = list(dict.fromkeys(ce_ranked[:3] + qwen_ranked[:5]))
            
            pred, details = execution_cascade(question, merged_candidates, all_schemas)
            tier = 3
        return {
            "question"    : question,
            "gold_db"     : gold_db,
            "predicted_db": pred,
            "tier"        : tier,
            "correct"     : (pred == gold_db),
            "mlp_prob"    : float(mlp_prob),
            "ce_top1"     : ce_ranked[0] if ce_ranked else "",
            **details,
        }

    results       = []
    correct_count = 0

    print(f"Starting cascade over {len(df)} queries (Parallel: 15 workers)...")
    
    tasks = []
    for idx, row in df.iterrows():
        ce_rec = ce_records[idx]
        if ce_rec["question"] in existing_records:
            continue
        tasks.append((idx, ce_rec["question"], ce_rec["gold_db"], ce_rec["ranked_dbs"], mlp_labels[idx], mlp_probs[idx]))

    print(f"Remaining tasks to execute: {len(tasks)}")
    write_lock = threading.Lock()

    with open(OUTPUT_PATH, "a" if existing_records else "w") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_query, *t) for t in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cascading"):
                res = f.result()
                with write_lock:
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()

    print(f"\nSaved records incrementally to:\n  {OUTPUT_PATH}\n")

    # 8. Metrics (Only computes if we run all from scratch or load them all)
    # Actually, we can just print a message that accuracy check script handles this.
    print("=" * 65)
    print("  PHASE 3 — RUN COMPLETE")
    print("  Use scripts/4_check_accuracy.py to view final metrics!")
    print("=" * 65)


if __name__ == "__main__":
    main()

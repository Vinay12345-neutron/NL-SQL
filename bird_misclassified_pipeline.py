#!/usr/bin/env python3
"""
BIRD Misclassified Hybrid Pipeline
==================================
Runs specifically on BIRD queries where the baseline dense retriever failed (R@1=0).

Stage 0: Qwen3-Embedding-8B (local GPU) -> Top 20
Stage 1: Qwen3-32B via Groq API
Stage 2: DeepSeek V3.1 via OpenRouter (deepseek/deepseek-chat-v3-5) -> ASYNC_SEMAPHORE=5
Stage 3: SQLite execution (Local)
Stage 4: Gemini 2.0 Flash via OpenRouter (google/gemini-2.0-flash-001)

Features:
- Incremental JSONL saving
- BIRD specific evidence included in Stage 1 and Stage 4 prompts
- Strict penalty for ERROR, instruction not to trust generic POPULATED
- Keeps domain-specific ID columns (e.g., statement_id)
"""

import os
import re
import json
import time
import asyncio
import sqlite3
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from groq import Groq, AsyncGroq
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Import dense retrieval components
from baseline_retrieval import (
    EmbeddingModel, load_schemas, run_retrieval
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STAGE1_MODEL = "google/gemini-2.0-flash-001"
STAGE2_MODEL = "deepseek/deepseek-chat-v3.1"
STAGE4_MODEL = "google/gemini-2.0-flash-001"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

MAX_COMPLETION_TOKENS = 2048
SQLITE_TIMEOUT = 5.0
ASYNC_SEMAPHORE_LIMIT = 5
BACKOFF_BASE = 5
MAX_RETRIES = 3
COARSE_FILTER_K = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# API Clients Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not GROQ_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Missing GROQ or OPENROUTER_API_KEY in .env")

# Stage 1: Groq client
groq_sync = Groq(api_key=GROQ_API_KEY)

# Stage 2 & 4: OpenRouter clients (using openai package format)
or_sync = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
or_async = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ---------------------------------------------------------------------------
# LLM Helpers
# ---------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """Strip <think> reasoning blocks (complete and incomplete)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def call_llm_sync(client, model: str, prompt: str,
                   system: str = "You are a database expert.",
                   temperature: float = 0.0) -> str:
    """Synchronous LLM call with infinite retry to guarantee data integrity."""
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=MAX_COMPLETION_TOKENS,
                timeout=60.0,
            )
            return strip_think(resp.choices[0].message.content or "")
        except Exception as e:
            wait = min(300, BACKOFF_BASE * (2 ** attempt))
            print(f"  [Sync Error] Network congested. Sleeping {wait}s and retrying infinitely... ({e})")
            time.sleep(wait)
            attempt += 1


async def call_llm_async(client, semaphore: asyncio.Semaphore,
                          model: str, prompt: str,
                          system: str = "You are a database expert.",
                          temperature: float = 0.1) -> str:
    """Async LLM call with infinite retry to guarantee data integrity."""
    async with semaphore:
        attempt = 0
        while True:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": prompt}
                        ],
                        model=model,
                        temperature=temperature,
                        max_tokens=MAX_COMPLETION_TOKENS,
                        timeout=60.0,
                    ),
                    timeout=65.0
                )
                return strip_think(resp.choices[0].message.content or "")
            except Exception as e:
                wait = min(300, BACKOFF_BASE * (2 ** attempt))
                print(f"  [Async Error] Network congested. Sleeping {wait}s and retrying infinitely... ({e})")
                await asyncio.sleep(wait)
                attempt += 1


# ---------------------------------------------------------------------------
# Schema Dictionary Builder (COMPRESSED — for Top-20 only)
# ---------------------------------------------------------------------------
# Change: Only strip bare `^id$`, keep domain-specific IDs like `statement_id`
_ID_PATTERNS = re.compile(
    r'(^id$)',
    re.IGNORECASE
)


def build_schema_dictionary(schemas: Dict[str, str],
                             subset: List[str] = None) -> Tuple[Dict, str]:
    """
    Build COMPRESSED dictionary for Stage 1 prompt (Top-20 only).
    """
    target = subset if subset else list(schemas.keys())

    dictionary = {}
    for db_id in target:
        schema_text = schemas.get(db_id, "")
        if not schema_text:
            continue

        tables = re.findall(
            r'Table:\s*(\w+),\s*Columns:\s*([^;]+)',
            schema_text, re.IGNORECASE
        )
        if not tables:
            dictionary[db_id] = schema_text[:80]
            continue

        table_parts = []
        for tbl, col_str in tables:
            if tbl.lower() == 'sqlite_sequence':
                continue
            cols = [c.strip() for c in col_str.split(',')]
            descriptive = [c for c in cols if not _ID_PATTERNS.match(c)]
            # We can keep more columns since we're only sending Top-20 (~1K tokens)
            descriptive = descriptive[:4]
            if descriptive:
                table_parts.append(f"{tbl}({','.join(descriptive)})")
            else:
                table_parts.append(tbl)

        dictionary[db_id] = ";".join(table_parts)

    lines = [f"{db_id}:{desc}" for db_id, desc in sorted(dictionary.items())]
    prompt_str = "\n".join(lines)

    return dictionary, prompt_str


def get_full_ddl(db_id: str, schemas: Dict[str, str]) -> str:
    schema_text = schemas.get(db_id, "")
    tables = re.findall(
        r'Table:\s*(\w+),\s*Columns:\s*([^;]+)',
        schema_text, re.IGNORECASE
    )
    if not tables:
        return schema_text

    ddl_lines = []
    for tbl, cols in tables:
        if tbl.lower() == 'sqlite_sequence':
            continue
        col_list = ", ".join(c.strip() for c in cols.split(','))
        ddl_lines.append(f"CREATE TABLE {tbl} ({col_list});")
    return "\n".join(ddl_lines)


# ---------------------------------------------------------------------------
# SQLite Path Resolution
# ---------------------------------------------------------------------------
def get_db_path(db_id: str) -> Optional[str]:
    candidates = [
        os.path.join(BASE_DIR, "spider_data", "database", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "spider_data", "test_database", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "dev_20240627", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "train", "train_databases", db_id, f"{db_id}.sqlite"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Stage 1: Plausible Database Identification (from Top-20 subset)
# ---------------------------------------------------------------------------
def stage1_identify_plausible(query: str, evidence: str,
                               dict_prompt: str) -> List[str]:
    """Ask Groq Qwen3-32B to identify plausible databases."""
    prompt = f"""You are given a user question, optional evidence context, and a dictionary of candidate databases with their table/column structures.

Your task: identify ALL databases that could PLAUSIBLY contain the answer based on the query and evidence. Be OVER-INCLUSIVE — include any database whose schema has tables/columns relevant to the question's domain.

Question: {query}
Evidence Context: {evidence}

Candidate Databases:
{dict_prompt}

IMPORTANT: Output ONLY a JSON array of database names. Example: ["academic", "scholar", "citeseer"] Nothing else."""

    response = call_llm_sync(or_sync, STAGE1_MODEL, prompt)

    try:
        result = json.loads(response)
        if isinstance(result, list):
            return [str(db) for db in result]
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [str(db) for db in result]
        except (json.JSONDecodeError, TypeError):
            pass

    found = re.findall(r'"(\w+)"', response)
    return found if found else []


# ---------------------------------------------------------------------------
# Stage 2: Broadcast SQL Generation (Async) via OpenRouter DeepSeek V3
# ---------------------------------------------------------------------------
def build_sql_gen_prompt(query: str, db_id: str, ddl: str) -> str:
    return f"""You are a SQL expert. Given a database schema and a natural language question, write ONE SQLite-compatible SQL query to answer the question.

Database: {db_id}
Schema:
{ddl}

Question: {query}

Rules:
- Write ONE complete SQL query
- Use ONLY the tables and columns defined in the schema above
- Use SQLite syntax (e.g., LIMIT instead of TOP)
- For "how many" or "number of" questions, use COUNT(*)
- If the query asks for specific values, use WHERE with LIKE for text matching

Output ONLY the SQL query. No explanation, no markdown, no backticks."""


async def stage2_generate_sql_async(
    semaphore: asyncio.Semaphore,
    query: str,
    plausible_dbs: List[str],
    schemas: Dict[str, str]
) -> Dict[str, str]:
    
    async def gen_one(db_id: str) -> Tuple[str, str]:
        ddl = get_full_ddl(db_id, schemas)
        prompt = build_sql_gen_prompt(query, db_id, ddl)
        sql = await call_llm_async(or_async, semaphore, STAGE2_MODEL, prompt,
                                    system="You are a SQL expert. Output ONLY SQL.",
                                    temperature=0.1)
        sql = re.sub(r'^```\w*\n?', '', sql.strip())
        sql = re.sub(r'\n?```$', '', sql.strip())
        sql = sql.strip().rstrip(';') + ';' if sql.strip() else ""
        return db_id, sql

    tasks = [gen_one(db_id) for db_id in plausible_dbs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sql_map = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        db_id, sql = r
        sql_map[db_id] = sql

    return sql_map


# ---------------------------------------------------------------------------
# Stage 3: Blind Execution
# ---------------------------------------------------------------------------
EXEC_ERROR = "ERROR"
EXEC_ZERO_COUNT = "ZERO_COUNT"
EXEC_POPULATED = "POPULATED"
EXEC_EMPTY_SET = "EMPTY_SET"

def stage3_execute(sql_map: Dict[str, str]) -> Dict[str, Dict]:
    import threading
    results = {}
    for db_id, sql in sql_map.items():
        db_path = get_db_path(db_id)
        entry = {
            "sql": sql,
            "status": EXEC_ERROR,
            "result": None,
            "error": None,
            "db_path": db_path,
        }

        if not sql:
            entry["error"] = "No SQL generated"
            results[db_id] = entry
            continue

        if not db_path:
            entry["error"] = f"SQLite file not found for {db_id}"
            results[db_id] = entry
            continue

        try:
            conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
            cursor = conn.cursor()
            
            # Setup an absolute kill-switch timer for runaway CPU cross-joins
            timer = threading.Timer(SQLITE_TIMEOUT, conn.interrupt)
            timer.start()

            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
            finally:
                timer.cancel()
                
            conn.close()

            if not rows:
                entry["status"] = EXEC_EMPTY_SET
                entry["result"] = []
            elif len(rows) == 1 and len(rows[0]) == 1:
                val = rows[0][0]
                if val == 0 or val is None:
                    entry["status"] = EXEC_ZERO_COUNT
                    entry["result"] = rows[:5]
                else:
                    entry["status"] = EXEC_POPULATED
                    entry["result"] = rows[:5]
            else:
                entry["status"] = EXEC_POPULATED
                entry["result"] = rows[:5]
        except Exception as e:
            entry["error"] = str(e)
            entry["status"] = EXEC_ERROR

        results[db_id] = entry
    return results


# ---------------------------------------------------------------------------
# Stage 4: Answer-Time Judge via Gemini 2.0 Flash
# ---------------------------------------------------------------------------
def stage4_judge(query: str, evidence: str, exec_results: Dict[str, Dict],
                  schemas: Dict[str, str]) -> str:
    """Select the correct database via OpenRouter Gemini 2.0."""
    candidates = list(exec_results.keys())
    
    if len(candidates) == 1:
        return candidates[0]

    # Build evidence for Judge LLM
    evidence_lines = []
    for db_id, r in exec_results.items():
        status = r["status"]
        sql = r["sql"]
        result_str = str(r["result"])[:200] if r["result"] else "None"
        error_str = r["error"] or "None"

        evidence_lines.append(
            f"Database: {db_id}\n"
            f"  SQL: {sql}\n"
            f"  Status: {status}\n"
            f"  Result: {result_str}\n"
            f"  Error: {error_str}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    prompt = f"""You are a database routing judge. Given a user question, BIRD competition metadata evidence, candidate databases, generated SQL queries, and execution results, determine which ONE database is the correct source.

Decision Rules (STRICT PRIORITY):
1. Execution Validity: Databases with an ERROR status MUST be strictly penalized and avoided if possible.
2. Data Presence Caution: DO NOT blindly trust databases just because they returned POPULATED data if their schema/tables look generic. Accidental matches happen.
3. BIRD Evidence Alignment: You must heavily weight how well the chosen database aligns with the specific BIRD evidence provided.
4. Structural Match: Look at the generated SQL — does it use tables and columns that perfectly match the nouns/verbs of the user's question?

Question: {query}
BIRD Evidence: {evidence}

Execution Evidence:
{evidence_block}

Task:
1. Reason briefly about which database is correct. Cite which evidence sentence justifies your selection.
2. On the final line, provide ONLY the exact database name inside <FINAL_DB> tags. Example: <FINAL_DB>world_1</FINAL_DB>"""

    response = call_llm_sync(or_sync, STAGE4_MODEL, prompt)
    
    match = re.search(r'<FINAL_DB>\s*([\w_]+)\s*</FINAL_DB>', response, re.IGNORECASE)
    if match:
        response_clean = match.group(1).strip().lower()
        for db in candidates:
            if db.lower() == response_clean:
                return db
        for db in candidates:
            if db.lower() in response_clean:
                return db

    # Fallback to pure string matching if tags fail
    response_lower = response.lower()
    for db in candidates:
        if db.lower() in response_lower:
            return db
            
    return candidates[0] if candidates else ""


# ---------------------------------------------------------------------------
# Stage 5: Build Training Data Record
# ---------------------------------------------------------------------------
def build_training_record(query: str, gold_db: str, evidence: str,
                           coarse_candidates: List[str], plausible_dbs: List[str],
                           sql_map: Dict[str, str], exec_results: Dict[str, Dict],
                           selected_db: str) -> Dict:
    candidate_contexts = []
    for db_id in plausible_dbs:
        ctx = {
            "db_id": db_id,
            "sql": sql_map.get(db_id, ""),
            "execution_status": exec_results.get(db_id, {}).get("status", "MISSING"),
            "execution_result": str(exec_results.get(db_id, {}).get("result", ""))[:200],
            "execution_error": exec_results.get(db_id, {}).get("error"),
        }
        candidate_contexts.append(ctx)

    return {
        "user_query": query,
        "bird_evidence": evidence,
        "gold_db": gold_db,
        "coarse_top20": coarse_candidates,
        "plausible_dbs_identified": plausible_dbs,
        "candidate_contexts": candidate_contexts,
        "final_selected_db": selected_db,
        "execution_status": exec_results.get(selected_db, {}).get("status", "MISSING") != EXEC_ERROR,
        "correct": selected_db == gold_db,
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
async def process_query(
    query_idx: int,
    query: str,
    gold_db: str,
    evidence: str,
    coarse_candidates: List[str],
    semaphore: asyncio.Semaphore,
    schemas: Dict[str, str],
) -> Dict:
    print(f"\n--- Query {query_idx+1}: {query[:50]}...")

    # Stage 1: LLM narrows Top-20 → plausible subset
    _, dict_prompt = build_schema_dictionary(schemas, subset=coarse_candidates)
    plausible = stage1_identify_plausible(query, evidence, dict_prompt)
    plausible = [db for db in plausible if db in schemas]
    print(f"  Stage 1: {len(plausible)} plausible: {plausible}")

    if not plausible:
        plausible = coarse_candidates[:5]

    # Stage 2: Generate SQL concurrently
    sql_map = await stage2_generate_sql_async(
        semaphore, query, plausible, schemas
    )

    # Stage 3: Execute and classify
    exec_results = stage3_execute(sql_map)
    status_summary = {s: sum(1 for r in exec_results.values() if r["status"] == s)
                      for s in [EXEC_POPULATED, EXEC_ZERO_COUNT, EXEC_EMPTY_SET, EXEC_ERROR]}
    print(f"  Stage 3: {status_summary}")

    # Stage 4: Judge
    selected_db = stage4_judge(query, evidence, exec_results, schemas)
    correct = selected_db == gold_db
    marker = "✓" if correct else "✗"
    print(f"  Stage 4: selected={selected_db} gold={gold_db} [{marker}]")

    # Stage 5: Build record
    record = build_training_record(
        query, gold_db, evidence, coarse_candidates, plausible, 
        sql_map, exec_results, selected_db
    )
    return record


async def main():
    print("Loading embedding model...")
    try:
        embed_model = EmbeddingModel(EMBED_MODEL)
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return

    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded!")
        return

    data_path = os.path.join(BASE_DIR, "processed_data", "bird_misclassified.json")
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        return

    with open(data_path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"BIRD MISCLASSIFIED PIPELINE: (n={len(data)})")
    print(f"{'='*60}")

    queries  = [item["question"] for item in data]
    gold_dbs = [item["db_id"]    for item in data]
    evidences = [item.get("evidence", "") for item in data]

    print(f"\n[Stage 0] Dense retrieval → Top-{COARSE_FILTER_K} candidates...")
    all_top_k = run_retrieval(embed_model, queries, gold_dbs, schemas, k=COARSE_FILTER_K)

    s0_recall = sum(1 for g, top in zip(gold_dbs, all_top_k) if g in top) / len(gold_dbs)
    print(f"[Stage 0] Recall@{COARSE_FILTER_K}: {s0_recall:.4f}")

    records = []
    correct_count = 0
    gold_in_plausible = 0

    semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE_LIMIT)
    jsonl_path = os.path.join(RESULTS_DIR, "bird_misclassified_pipeline.jsonl")

    # Resume logic
    start_idx = 0
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    records.append(rec)
                    if rec.get("correct"):
                        correct_count += 1
                    if rec.get("gold_db") in rec.get("plausible_dbs_identified", []):
                        gold_in_plausible += 1
            start_idx = len(records)
            if start_idx > 0:
                print(f"\\n[Resume] Found {start_idx} processed queries. Resuming from Query {start_idx+1}...")
        except BaseException as e:
            print(f"Error reading jsonl resume file: {e}. Starting from 0.")
            start_idx = 0

    # Ensure we don't clear the file if we are resuming
    mode = 'a' if start_idx > 0 else 'w'
    with open(jsonl_path, mode) as f:
        pass

    for i in range(start_idx, len(data)):
        query = queries[i]
        gold_db = gold_dbs[i]
        evidence = evidences[i]
        coarse_candidates = all_top_k[i]

        record = await process_query(
            i, query, gold_db, evidence, coarse_candidates,
            semaphore, schemas
        )
        records.append(record)

        # Output incrementally
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

        if record.get("correct"):
            correct_count += 1
        if gold_db in record.get("plausible_dbs_identified", []):
            gold_in_plausible += 1

        n_done = i + 1
        acc = correct_count / n_done
        recall = gold_in_plausible / n_done
        print(f"  Running: {n_done}/{len(data)} | Acc (Recovery Rate)={acc:.3f} | "
              f"Stage1 Recall={recall:.3f}")

    n = len(records)
    accuracy = correct_count / n if n > 0 else 0
    stage1_recall = gold_in_plausible / n if n > 0 else 0

    print(f"\n{'='*60}")
    print(f"FINAL METRICS: BIRD MISCLASSIFIED SET")
    print(f"{'='*60}")
    print(f"  Total misclassified queries processed: {n}")
    print(f"  Stage 0 Recall@{COARSE_FILTER_K} (on miss set): {s0_recall:.4f}")
    print(f"  Stage 1 Plausible Recall (on miss set): {gold_in_plausible}/{n} = {stage1_recall:.4f}")
    print(f"  Final Top-1 recovery accuracy: {correct_count}/{n} = {accuracy:.4f} ({correct_count} queries recovered!)")

    summary_path = os.path.join(RESULTS_DIR, "bird_misclassified_summary.json")
    summary = {
        "dataset": "BIRD_Misclassified",
        "n_queries": n,
        "recovered": correct_count,
        "accuracy": accuracy,
        "stage0_recall": s0_recall,
        "stage1_recall": stage1_recall,
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())

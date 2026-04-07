#!/usr/bin/env python3
"""
Hybrid Zero-Retrieval Pipeline
================================
Stage 0: Dense Retrieval Coarse Filter (Qwen3-Embedding-8B)
  - Encode query + all 285 schemas → Top-20 candidates via cosine sim
  - Runs LOCALLY on GPU — zero API calls

Stage 1: Plausible Database Identification (LLM)
  - LLM receives compressed dictionary of ONLY the Top-20 candidates
  - Returns narrowed list of plausible DBs (1-10)
  - Prompt size: ~1K tokens (vs 15K+ for all 285 DBs)

Stage 2: Broadcast SQL Generation (Async)
  - 1 schema per prompt, fired concurrently via AsyncGroq + Semaphore(3)

Stage 3: Blind Execution
  - Execute SQL against local .sqlite files
  - Classify: ERROR / ZERO_COUNT / POPULATED

Stage 4: Answer-Time Judge
  - Short-circuit: single POPULATED → auto-select
  - Multiple POPULATED or all ZERO_COUNT → LLM Judge

Stage 5: NLP Training Data Output
  - JSONL with full provenance per query
"""

import os
import re
import json
import time
import asyncio
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from groq import Groq, AsyncGroq
from dotenv import load_dotenv

load_dotenv()

# Import dense retrieval components
from baseline_retrieval import (
    EmbeddingModel, load_schemas, run_retrieval
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_MODEL = "qwen/qwen3-32b"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
MAX_COMPLETION_TOKENS = 2048
SQLITE_TIMEOUT = 5.0
ASYNC_SEMAPHORE_LIMIT = 2
BACKOFF_BASE = 10
MAX_RETRIES = 4
RATE_LIMIT_DELAY = 6.0
COARSE_FILTER_K = 20              # Top-K from dense retrieval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM Helpers
# ---------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """Strip qwen3-32b <think> reasoning blocks (complete and incomplete)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def call_llm_sync(client: Groq, prompt: str,
                   system: str = "You are a database expert.",
                   temperature: float = 0.0) -> str:
    """Synchronous LLM call with retry + backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                model=LLM_MODEL,
                temperature=temperature,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            return strip_think(resp.choices[0].message.content or "")
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [429] Retry {attempt+1}/{MAX_RETRIES}, sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"  [LLM Error] {e}")
            return ""
    return ""


async def call_llm_async(client: AsyncGroq, semaphore: asyncio.Semaphore,
                          prompt: str, system: str = "You are a database expert.",
                          temperature: float = 0.1) -> str:
    """Async LLM call with semaphore rate limiting + backoff."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    model=LLM_MODEL,
                    temperature=temperature,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                return strip_think(resp.choices[0].message.content or "")
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = BACKOFF_BASE * (2 ** attempt)
                    print(f"  [429 async] Retry {attempt+1}, sleeping {wait}s")
                    await asyncio.sleep(wait)
                    continue
                print(f"  [Async LLM Error] {e}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Schema Dictionary Builder (COMPRESSED — for Top-20 only)
# ---------------------------------------------------------------------------
# Patterns for ID/key columns to exclude
_ID_PATTERNS = re.compile(
    r'(^id$|_id$|^.*id$|^code$|.*code$|^seq$|^idx$|^key$|^fk_|^pk_)',
    re.IGNORECASE
)


def build_schema_dictionary(schemas: Dict[str, str],
                             subset: List[str] = None) -> Tuple[Dict, str]:
    """
    Build COMPRESSED dictionary for Stage 1 prompt.
    
    If subset is provided, only include those db_ids.
    
    Compression rules:
    1. Filter out ID/key columns
    2. Keep max 3 descriptive columns per table
    3. Compact format: db_id:tbl(col1,col2);tbl2(col1)
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
            descriptive = descriptive[:3]
            if descriptive:
                table_parts.append(f"{tbl}({','.join(descriptive)})")
            else:
                table_parts.append(tbl)

        dictionary[db_id] = ";".join(table_parts)

    lines = [f"{db_id}:{desc}" for db_id, desc in sorted(dictionary.items())]
    prompt_str = "\n".join(lines)

    return dictionary, prompt_str


def get_full_ddl(db_id: str, schemas: Dict[str, str]) -> str:
    """Get full DDL-like schema for SQL generation prompt."""
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
    """Find the .sqlite file for a given db_id."""
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
def stage1_identify_plausible(client: Groq, query: str,
                               dict_prompt: str) -> List[str]:
    """Ask LLM to identify plausible databases from the Top-20 dictionary."""
    prompt = f"""You are given a user question and a dictionary of candidate databases with their table/column structures.

Your task: identify ALL databases that could PLAUSIBLY contain the answer. Be OVER-INCLUSIVE — include any database whose schema has tables/columns relevant to the question's domain.

Return a JSON array of database names. Example: ["academic", "scholar", "citeseer"]

Question: {query}

Candidate Databases:
{dict_prompt}

IMPORTANT: Output ONLY a JSON array of database names. Nothing else."""

    time.sleep(RATE_LIMIT_DELAY)
    response = call_llm_sync(client, prompt)

    # Parse JSON array
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
# Stage 2: Broadcast SQL Generation (Async)
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
    async_client: AsyncGroq,
    semaphore: asyncio.Semaphore,
    query: str,
    plausible_dbs: List[str],
    schemas: Dict[str, str]
) -> Dict[str, str]:
    """Generate SQL for all plausible DBs concurrently."""

    async def gen_one(db_id: str) -> Tuple[str, str]:
        ddl = get_full_ddl(db_id, schemas)
        prompt = build_sql_gen_prompt(query, db_id, ddl)
        sql = await call_llm_async(async_client, semaphore, prompt,
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
            print(f"  [SQL Gen Error] {r}")
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
    """Execute all SQL queries, classify results."""
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
            cursor.execute(sql)
            rows = cursor.fetchall()
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
# Stage 4: Answer-Time Judge
# ---------------------------------------------------------------------------
def stage4_judge(client: Groq, query: str, exec_results: Dict[str, Dict],
                  schemas: Dict[str, str]) -> str:
    """Select the correct database based on execution evidence."""
    populated = [db for db, r in exec_results.items() if r["status"] == EXEC_POPULATED]
    zero_count = [db for db, r in exec_results.items() if r["status"] == EXEC_ZERO_COUNT]
    empty_set = [db for db, r in exec_results.items() if r["status"] == EXEC_EMPTY_SET]
    errors = [db for db, r in exec_results.items() if r["status"] == EXEC_ERROR]

    print(f"  Judge input: populated={populated} zero_count={zero_count} "
          f"empty={empty_set} errors={errors}")

    # NO short-circuit — always let the Judge evaluate all evidence.
    # Short-circuiting was causing false positives when wrong DBs had
    # accidental POPULATED results from bad COUNT(*) queries.

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

    prompt = f"""You are a database routing judge. Given a user question, candidate databases, generated SQL, and execution results, determine which ONE database is the correct source.

Decision rules (STRICT PRIORITY):
1. Execution Validity: Databases with ERROR status are automatically incorrect.
2. The Schema Specificity Rule (CRITICAL): A database with a 100% specific schema match ALWAYS defeats a database with a generic schema match, regardless of execution results. 
   - Look intensely at the nouns in the question (e.g., 'domain', 'conference', 'journal').
   - If DB_A has an explicit table for 'conference' and DB_B forces a generic text search on a 'venue' column, DB_A is the winner.
3. The False Positive POPULATED Trap: Many of these databases are completely empty. Therefore, a structurally perfect SQL query on an empty database (returning ZERO_COUNT) is CORRECT. A POPULATED result from a structurally inferior/generic database is a FALSE POSITIVE and must be rejected.
4. Decision: Pick the database whose schema architecture is explicitly custom-built to answer the user's specific terminology.

Question: {query}

Execution Evidence:
{evidence_block}

IMPORTANT: Reply with ONLY the exact database name. Nothing else."""

    time.sleep(RATE_LIMIT_DELAY)
    response = call_llm_sync(client, prompt)

    response_clean = response.strip().lower()
    candidates = list(exec_results.keys())

    for db in candidates:
        if db.lower() == response_clean:
            return db
    for db in candidates:
        if db.lower() in response_clean:
            return db

    if populated:
        return populated[0]
    if zero_count:
        return zero_count[0]
    if empty_set:
        return empty_set[0]
    return candidates[0] if candidates else ""


# ---------------------------------------------------------------------------
# Stage 5: Build Training Data Record
# ---------------------------------------------------------------------------
def build_training_record(query: str, gold_db: str, 
                           coarse_candidates: List[str],
                           plausible_dbs: List[str],
                           sql_map: Dict[str, str], exec_results: Dict[str, Dict],
                           selected_db: str) -> Dict:
    """Build one JSONL training record."""
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
    coarse_candidates: List[str],
    sync_client: Groq,
    async_client: AsyncGroq,
    semaphore: asyncio.Semaphore,
    schemas: Dict[str, str],
) -> Dict:
    """Process a single query through Stages 1-5."""
    print(f"\n--- Query {query_idx+1}: {query[:70]}...")

    # Stage 1: LLM narrows Top-20 → plausible subset
    _, dict_prompt = build_schema_dictionary(schemas, subset=coarse_candidates)
    plausible = stage1_identify_plausible(sync_client, query, dict_prompt)
    plausible = [db for db in plausible if db in schemas]
    print(f"  Stage 1: {len(plausible)} plausible: {plausible}")

    if not plausible:
        print("  WARNING: No plausible DBs — falling back to top-5 from coarse filter")
        plausible = coarse_candidates[:5]

    # Stage 2: Generate SQL concurrently
    sql_map = await stage2_generate_sql_async(
        async_client, semaphore, query, plausible, schemas
    )
    print(f"  Stage 2: Generated SQL for {len(sql_map)}/{len(plausible)} DBs")

    # Stage 3: Execute and classify
    exec_results = stage3_execute(sql_map)
    status_summary = {s: sum(1 for r in exec_results.values() if r["status"] == s)
                      for s in [EXEC_POPULATED, EXEC_ZERO_COUNT, EXEC_EMPTY_SET, EXEC_ERROR]}
    print(f"  Stage 3: {status_summary}")

    # Stage 4: Judge
    selected_db = stage4_judge(sync_client, query, exec_results, schemas)
    correct = selected_db == gold_db
    marker = "✓" if correct else "✗"
    print(f"  Stage 4: selected={selected_db} gold={gold_db} [{marker}]")

    # Stage 5: Build training record
    record = build_training_record(
        query, gold_db, coarse_candidates, plausible, sql_map, exec_results, selected_db
    )
    return record


async def run_pipeline(dataset_name: str, data_path: str,
                        embed_model: EmbeddingModel,
                        schemas: Dict[str, str],
                        limit: int = 50):
    """Run the full pipeline on a dataset."""
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        return

    api_key = os.environ.get("GROQ")
    if not api_key:
        print("ERROR: GROQ API key not found in environment")
        return

    sync_client = Groq(api_key=api_key)
    async_client = AsyncGroq(api_key=api_key)
    semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE_LIMIT)

    with open(data_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    queries  = [item["question"] for item in data]
    gold_dbs = [item["db_id"]    for item in data]

    print(f"\n{'='*60}")
    print(f"HYBRID PIPELINE: {dataset_name} (n={len(data)})")
    print(f"{'='*60}")

    # ── Stage 0: Dense Retrieval Coarse Filter ──
    print(f"\n[Stage 0] Dense retrieval → Top-{COARSE_FILTER_K} candidates...")
    all_top_k = run_retrieval(embed_model, queries, gold_dbs, schemas, k=COARSE_FILTER_K)

    # Check Stage 0 recall
    s0_recall = sum(1 for g, top in zip(gold_dbs, all_top_k) if g in top) / len(gold_dbs)
    print(f"[Stage 0] Recall@{COARSE_FILTER_K}: {s0_recall:.4f}")

    # ── Stages 1-5: Per-query processing ──
    records = []
    correct_count = 0
    gold_in_plausible = 0

    for i in range(len(data)):
        query = queries[i]
        gold_db = gold_dbs[i]
        coarse_candidates = all_top_k[i]

        record = await process_query(
            i, query, gold_db, coarse_candidates,
            sync_client, async_client, semaphore, schemas
        )
        records.append(record)

        if record.get("correct"):
            correct_count += 1
        if gold_db in record.get("plausible_dbs_identified", []):
            gold_in_plausible += 1

        n_done = i + 1
        acc = correct_count / n_done
        recall = gold_in_plausible / n_done
        print(f"  Running: {n_done}/{len(data)} | Acc={acc:.3f} | "
              f"Stage1 Recall={recall:.3f}")

    # Final metrics
    n = len(records)
    accuracy = correct_count / n if n > 0 else 0
    stage1_recall = gold_in_plausible / n if n > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total queries:     {n}")
    print(f"  Stage0 Recall@{COARSE_FILTER_K}:  {s0_recall:.4f}")
    print(f"  Stage1 Recall:     {gold_in_plausible}/{n} = {stage1_recall:.4f}")
    print(f"  Final Accuracy:    {correct_count}/{n} = {accuracy:.4f}")

    # Save JSONL training data
    jsonl_path = os.path.join(RESULTS_DIR,
                               f"{dataset_name.lower()}_zero_retrieval.jsonl")
    with open(jsonl_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Training data:     {jsonl_path}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR,
                                 f"{dataset_name.lower()}_zero_retrieval_summary.json")
    summary = {
        "dataset": dataset_name,
        "n_queries": n,
        "accuracy": accuracy,
        "stage0_recall": s0_recall,
        "stage1_recall": stage1_recall,
        "correct": correct_count,
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:           {summary_path}")

    return summary


async def main():
    # Load embedding model
    print("Loading embedding model...")
    try:
        embed_model = EmbeddingModel(EMBED_MODEL)
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return

    # Load schemas
    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded!")
        return

    data_dir = os.path.join(BASE_DIR, "processed_data")

    # Run Spider
    await run_pipeline(
        "Spider",
        os.path.join(data_dir, "spider_route_test.json"),
        embed_model, schemas, limit=50
    )

    # Run BIRD
    await run_pipeline(
        "BIRD",
        os.path.join(data_dir, "bird_route_test.json"),
        embed_model, schemas, limit=50
    )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
BIRD Correct (Easy) Queries Pipeline
====================================
Runs the execution pipeline on a systematically sampled subset of 1,100 
correctly classified BIRD queries.
"""

import os
import re
import json
import time
import asyncio
import sqlite3
import sys
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from groq import Groq
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

from baseline_retrieval import EmbeddingModel, load_schemas, run_retrieval

STAGE1_MODEL = "google/gemini-2.0-flash-001"
STAGE2_MODEL = "deepseek/deepseek-chat-v3.1"
STAGE4_MODEL = "google/gemini-2.0-flash-001"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

MAX_COMPLETION_TOKENS = 2048
SQLITE_TIMEOUT = 5.0
ASYNC_SEMAPHORE_LIMIT = 5
BACKOFF_BASE = 5
COARSE_FILTER_K = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

or_sync = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
or_async = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()

def call_llm_sync(client, model: str, prompt: str, system: str = "You are a database expert.", temperature: float = 0.0) -> str:
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                model=model, temperature=temperature, max_tokens=MAX_COMPLETION_TOKENS, timeout=60.0,
            )
            return strip_think(resp.choices[0].message.content or "")
        except Exception as e:
            wait = min(300, BACKOFF_BASE * (2 ** attempt))
            print(f"  [Sync Error] Network congested. Sleeping {wait}s... ({e})")
            time.sleep(wait)
            attempt += 1

async def call_llm_async(client, semaphore: asyncio.Semaphore, model: str, prompt: str, system: str = "You are a database expert.", temperature: float = 0.1) -> str:
    async with semaphore:
        attempt = 0
        while True:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        model=model, temperature=temperature, max_tokens=MAX_COMPLETION_TOKENS, timeout=60.0,
                    ), timeout=65.0
                )
                return strip_think(resp.choices[0].message.content or "")
            except Exception as e:
                wait = min(300, BACKOFF_BASE * (2 ** attempt))
                print(f"  [Async Error] Network congested. Sleeping {wait}s... ({e})")
                await asyncio.sleep(wait)
                attempt += 1

_ID_PATTERNS = re.compile(r'(^id$)', re.IGNORECASE)

def build_schema_dictionary(schemas: Dict[str, str], subset: List[str] = None) -> Tuple[Dict, str]:
    target = subset if subset else list(schemas.keys())
    dictionary = {}
    for db_id in target:
        schema_text = schemas.get(db_id, "")
        if not schema_text: continue
        tables = re.findall(r'Table:\s*(\w+),\s*Columns:\s*([^;]+)', schema_text, re.IGNORECASE)
        if not tables:
            dictionary[db_id] = schema_text[:80]
            continue
        table_parts = []
        for tbl, col_str in tables:
            if tbl.lower() == 'sqlite_sequence': continue
            cols = [c.strip() for c in col_str.split(',')]
            descriptive = [c for c in cols if not _ID_PATTERNS.match(c)][:4]
            if descriptive:
                table_parts.append(f"{tbl}({','.join(descriptive)})")
            else:
                table_parts.append(tbl)
        dictionary[db_id] = ";".join(table_parts)
    prompt_str = "\n".join([f"{db_id}:{desc}" for db_id, desc in sorted(dictionary.items())])
    return dictionary, prompt_str

def get_full_ddl(db_id: str, schemas: Dict[str, str]) -> str:
    schema_text = schemas.get(db_id, "")
    tables = re.findall(r'Table:\s*(\w+),\s*Columns:\s*([^;]+)', schema_text, re.IGNORECASE)
    if not tables: return schema_text
    ddl_lines = []
    for tbl, cols in tables:
        if tbl.lower() == 'sqlite_sequence': continue
        col_list = ", ".join(c.strip() for c in cols.split(','))
        ddl_lines.append(f"CREATE TABLE {tbl} ({col_list});")
    return "\n".join(ddl_lines)

def get_db_path(db_id: str) -> Optional[str]:
    candidates = [
        os.path.join(BASE_DIR, "spider_data", "database", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "spider_data", "test_database", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "dev_20240627", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(BASE_DIR, "train", "train_databases", db_id, f"{db_id}.sqlite"),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def stage1_identify_plausible(query: str, evidence: str, dict_prompt: str) -> List[str]:
    prompt = f"""You are given a user question, optional evidence context, and a dictionary of candidate databases with their table/column structures.

Your task: identify ALL databases that could PLAUSIBLY contain the answer based on the query and evidence. Be OVER-INCLUSIVE — include any database whose schema has tables/columns relevant to the question's domain.

Question: {query}
Evidence Context: {evidence}

Candidate Databases:
{dict_prompt}

IMPORTANT: Output ONLY a JSON array of database names. Example: ["academic", "scholar", "citeseer"] Nothing else."""
    response = call_llm_sync(or_sync, STAGE1_MODEL, prompt)
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if match:
        try:
            res = json.loads(match.group(0))
            if isinstance(res, list): return [str(d) for d in res]
        except: pass
    found = re.findall(r'"(\w+)"', response)
    return found if found else []

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

async def stage2_generate_sql_async(semaphore: asyncio.Semaphore, query: str, plausible_dbs: List[str], schemas: Dict[str, str]) -> Dict[str, str]:
    async def gen_one(db_id: str) -> Tuple[str, str]:
        ddl = get_full_ddl(db_id, schemas)
        prompt = build_sql_gen_prompt(query, db_id, ddl)
        sql = await call_llm_async(or_async, semaphore, STAGE2_MODEL, prompt, system="You are a SQL expert. Output ONLY SQL.", temperature=0.1)
        sql = re.sub(r'^```\w*\n?', '', sql.strip())
        sql = re.sub(r'\n?```$', '', sql.strip()).strip().rstrip(';') + ';' if sql.strip() else ""
        return db_id, sql
    tasks = [gen_one(db_id) for db_id in plausible_dbs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {r[0]: r[1] for r in results if not isinstance(r, Exception)}

EXEC_ERROR = "ERROR"
EXEC_ZERO_COUNT = "ZERO_COUNT"
EXEC_POPULATED = "POPULATED"
EXEC_EMPTY_SET = "EMPTY_SET"

def stage3_execute(sql_map: Dict[str, str]) -> Dict[str, Dict]:
    import threading
    results = {}
    for db_id, sql in sql_map.items():
        db_path = get_db_path(db_id)
        entry = {"sql": sql, "status": EXEC_ERROR, "result": None, "error": None, "db_path": db_path}
        if not sql or not db_path:
            results[db_id] = entry
            continue
        try:
            conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT)
            cursor = conn.cursor()
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
            elif len(rows) == 1 and len(rows[0]) == 1 and (rows[0][0] == 0 or rows[0][0] is None):
                entry["status"] = EXEC_ZERO_COUNT
                entry["result"] = rows[:5]
            else:
                entry["status"] = EXEC_POPULATED
                entry["result"] = rows[:5]
        except Exception as e:
            entry["error"] = str(e)
            entry["status"] = EXEC_ERROR
        results[db_id] = entry
    return results

def stage4_judge(query: str, evidence: str, exec_results: Dict[str, Dict], schemas: Dict[str, str]) -> str:
    candidates = list(exec_results.keys())
    if len(candidates) == 1: return candidates[0]
    
    evidence_lines = []
    for db_id, r in exec_results.items():
        evidence_lines.append(
            f"Database: {db_id}\n"
            f"  SQL: {r['sql']}\n"
            f"  Status: {r['status']}\n"
            f"  Result: {str(r['result'])[:200] if r['result'] else 'None'}\n"
            f"  Error: {r['error'] or 'None'}"
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
            if db.lower() in response_clean: return db
    return candidates[0] if candidates else ""

def build_training_record(query: str, gold_db: str, evidence: str, coarse_candidates: List[str], plausible_dbs: List[str], sql_map: Dict[str, str], exec_results: Dict[str, Dict], selected_db: str) -> Dict:
    return {
        "user_query": query,
        "bird_evidence": evidence,
        "gold_db": gold_db,
        "coarse_top20": coarse_candidates,
        "plausible_dbs_identified": plausible_dbs,
        "candidate_contexts": [{"db_id": db, "sql": sql_map.get(db, ""), "execution_status": exec_results.get(db, {}).get("status", "MISSING"), "execution_result": str(exec_results.get(db, {}).get("result", ""))[:200], "execution_error": exec_results.get(db, {}).get("error")} for db in plausible_dbs],
        "final_selected_db": selected_db,
        "execution_status": exec_results.get(selected_db, {}).get("status", "MISSING") != EXEC_ERROR,
        "correct": selected_db == gold_db,
    }

async def process_query(query_idx: int, query: str, gold_db: str, evidence: str, coarse_candidates: List[str], semaphore: asyncio.Semaphore, schemas: Dict[str, str]) -> Dict:
    print(f"\n--- Query {query_idx+1}: {query[:50]}...")
    _, dict_prompt = build_schema_dictionary(schemas, subset=coarse_candidates)
    
    plausible = stage1_identify_plausible(query, evidence, dict_prompt)
    plausible = [db for db in plausible if db in schemas] or coarse_candidates[:5]
    
    sql_map = await stage2_generate_sql_async(semaphore, query, plausible, schemas)
    exec_results = stage3_execute(sql_map)
    selected_db = stage4_judge(query, evidence, exec_results, schemas)
    
    print(f"  Stage 4: selected={selected_db} gold={gold_db} [{'✓' if selected_db == gold_db else '✗'}]")
    return build_training_record(query, gold_db, evidence, coarse_candidates, plausible, sql_map, exec_results, selected_db)

async def main():
    schemas = load_schemas()
    embed_model = EmbeddingModel(EMBED_MODEL)
    
    # 1. Load Correct subset
    data_path = os.path.join(BASE_DIR, "processed_data", "bird_correct.json")
    with open(data_path) as f:
        full_data = json.load(f)
        
    # 2. SYSTEMATIC SAMPLING: Take 1,100 queries evenly distributed
    target_count = 1100
    if len(full_data) <= target_count:
        data = full_data
    else:
        step = len(full_data) / target_count
        data = [full_data[int(i * step)] for i in range(target_count)]
    
    print(f"\n{'='*60}\nBIRD CORRECT (EASY) QUERIES PIPELINE: (n={len(data)})\n{'='*60}")
    
    queries = [item["question"] for item in data]
    gold_dbs = [item["db_id"] for item in data]
    evidences = [item.get("evidence", "") for item in data]
    all_top_k = run_retrieval(embed_model, queries, gold_dbs, schemas, k=COARSE_FILTER_K)

    semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE_LIMIT)
    jsonl_path = os.path.join(RESULTS_DIR, "bird_easy_queries_pipeline.jsonl")
    
    start_idx = 0
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            records = [json.loads(l) for l in f if l.strip()]
        start_idx = len(records)
        print(f"Resuming from {start_idx}...")

    correct_count = sum(1 for r in records if r["correct"])
    pop_wins = 0
    vs_wins = 0

    for i in range(start_idx, len(data)):
        record = await process_query(i, queries[i], gold_dbs[i], evidences[i], all_top_k[i], semaphore, schemas)
        records.append(record)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
            
        if record["correct"]:
            correct_count += 1
        
        print(f"  Running: {i+1}/{len(data)} | Acc: {correct_count/(i+1):.3f}")

    for r in records:
        if r["correct"]:
            for ctx in r["candidate_contexts"]:
                if ctx["db_id"] == r["final_selected_db"]:
                    if ctx["execution_status"] == "POPULATED":
                        pop_wins += 1
                    elif ctx["execution_status"] in ("VALID_STRUCTURE", "EMPTY_SET", "ZERO_COUNT"):
                        vs_wins += 1

    print(f"\n{'='*60}\nBIRD EASY SET FINAL METRICS\n{'='*60}")
    print(f"  Total Processed: {len(records)}")
    print(f"  Accuracy: {correct_count}/{len(records)} ({correct_count/len(records)*100:.1f}%)")
    print(f"  POPULATED Wins: {pop_wins}")
    print(f"  VALID_STRUCTURE/EMPTY Wins: {vs_wins}")

if __name__ == "__main__":
    asyncio.run(main())
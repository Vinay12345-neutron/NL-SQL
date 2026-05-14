#!/usr/bin/env python3
import os
import json
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP15_PATH  = os.path.join(BASE_DIR, "data", "spider_baseline1_top15.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.jsonl")
TABLES_PATHS = [
    os.path.join(BASE_DIR, "processed_data", "spider_tables.json"),
    os.path.join(BASE_DIR, "processed_data", "spider_test_tables.json"),
]

MODEL = "google/gemini-2.0-flash-001"
CONCURRENCY_LIMIT = 10

# Initialize OpenRouter Client
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")
client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

def build_schema_context(db_entry: dict) -> str:
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

async def call_llm(prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        for _ in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                await asyncio.sleep(2)
        return ""

async def process_query(item: dict, schemas: dict, sem: asyncio.Semaphore, out_file) -> dict:
    question = item["question"]
    gold_db = item["gold_db"]
    top15_dbs = item["top15_dbs"]

    # Build prompt
    prompt = f"You are a strict semantic reranker. I will provide a user query and 15 database schemas.\n"
    prompt += f"Your task is to assign a probability score (float between 0.0 and 1.0) to each schema indicating how likely it is the correct database for the query.\n"
    prompt += f"You MUST output exactly a raw JSON array containing 15 floats, corresponding exactly to the order of the schemas provided below.\n"
    prompt += f"DO NOT output any explanations or markdown formatting other than the JSON array. Example output: [0.95, 0.1, 0.05, ...]\n\n"
    prompt += f"User Query: {question}\n\n"
    
    for i, db in enumerate(top15_dbs):
        schema_text = schemas.get(db, f"Database: {db} (Schema missing)")
        prompt += f"Schema {i+1}:\n{schema_text}\n\n"

    # Call LLM
    response_text = await call_llm(prompt, sem)
    
    # Parse output robustly
    scores = []
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) == 15:
                scores = [float(x) for x in parsed]
    except Exception:
        pass

    # Fallback if parsing failed
    if len(scores) != 15:
        scores = [0.99 - i*0.01 for i in range(15)]

    # Sort databases descending by score
    db_score_pairs = list(zip(top15_dbs, scores))
    db_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    ranked_dbs = [db for db, s in db_score_pairs]
    scores_dict = {db: s for db, s in db_score_pairs}

    result = {
        "question": question,
        "gold_db": gold_db,
        "ranked_dbs": ranked_dbs,
        "scores": scores_dict
    }

    # Write incrementally
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    return result

async def main_async():
    # 1. Load queries
    with open(TOP15_PATH, "r") as f:
        data = json.load(f)

    # 2. Load schemas
    all_schemas = {}
    for path in TABLES_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                for entry in json.load(f):
                    all_schemas[entry["db_id"]] = build_schema_context(entry)

    # 3. Handle resumption
    processed_questions = set()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                try:
                    processed_questions.add(json.loads(line.strip())["question"])
                except:
                    pass

    to_process = [item for item in data if item["question"] not in processed_questions]
    
    print(f"Loaded {len(data)} queries total.")
    print(f"Already processed {len(processed_questions)} queries.")
    print(f"Remaining to rerank: {len(to_process)}")

    if not to_process:
        return

    # 4. Process concurrently
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    with open(OUTPUT_PATH, "a") as out_file:
        tasks = [process_query(item, all_schemas, sem, out_file) for item in to_process]
        
        # Use tqdm.asyncio to track progress
        for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Reranking"):
            await f_task

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

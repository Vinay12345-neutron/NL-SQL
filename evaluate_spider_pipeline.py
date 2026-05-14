#!/usr/bin/env python3
import os
import json
import time
from sentence_transformers import CrossEncoder
from tqdm import tqdm

DEV_LOGS_PATH = "results/spider_dev_execution_logs.jsonl"
MODEL_PATH = "models/spider_execution_router"
FAILURE_LOG_PATH = "results/spider_failure_analysis.json"

def evaluate_pipeline():
    print(f"Loading Spider Cross-Encoder from: {MODEL_PATH}")
    model = CrossEncoder(MODEL_PATH)

    print(f"Loading Dev Set logs from: {DEV_LOGS_PATH}")
    records = []
    with open(DEV_LOGS_PATH, "r") as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))

    correct_routing = 0
    failures = []
    
    start_time = time.time()

    print(f"\nStarting Evaluation on {len(records)} queries...")
    for record in tqdm(records, desc="Scoring Databases"):
        query = record["user_query"]
        gold_db = record["gold_db"]
        contexts = record["candidate_contexts"]

        if not contexts:
            failures.append({"query": query, "gold_db": gold_db, "predicted_db": "NONE", "reason": "No plausible DBs generated"})
            continue

        pairs = []
        db_ids = []
        for ctx in contexts:
            evidence_text = f"Status: {ctx['execution_status']} | Error: {ctx.get('execution_error') or 'None'} | Database: {ctx['db_id']} | SQL: {ctx['sql']}"
            pairs.append([query, evidence_text])
            db_ids.append(ctx["db_id"])

        scores = model.predict(pairs)
        best_index = scores.argmax()
        predicted_db = db_ids[best_index]

        if predicted_db == gold_db:
            correct_routing += 1
        else:
            record["predicted_db"] = predicted_db
            record["scores"] = {db: float(score) for db, score in zip(db_ids, scores)}
            failures.append(record)

    total_time = time.time() - start_time
    acc = correct_routing / len(records)

    print("\n" + "="*50)
    print(" 🕸️ SPIDER BENCHMARK - FINAL RESULTS 🕸️")
    print("="*50)
    print(f"Total Queries Evaluated: {len(records)}")
    print(f"Correctly Routed:        {correct_routing}")
    print(f"Total Time Taken:        {total_time:.2f} seconds")
    print(f"Average Time per Query:  {(total_time/len(records))*1000:.2f} milliseconds")
    print("-" * 50)
    print(f"🏆 FINAL ACCURACY:        {acc*100:.2f}%")
    print("="*50)

    with open(FAILURE_LOG_PATH, "w") as f:
        json.dump(failures, f, indent=4)
    print(f"\nFailure analysis saved to '{FAILURE_LOG_PATH}'.")

if __name__ == "__main__":
    evaluate_pipeline()
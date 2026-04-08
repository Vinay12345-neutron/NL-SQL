import json
import time
import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# --- 1. CONFIGURATION ---
MODEL_PATH = "models/bird_execution_router"
DEV_DATA_PATH = "results/bird_dev_execution_logs.jsonl" 

def main():
    print(f"Loading Custom Cross-Encoder from: {MODEL_PATH}")
    model = CrossEncoder(MODEL_PATH, max_length=1024)

    print(f"Loading Dev Set from: {DEV_DATA_PATH}")
    with open(DEV_DATA_PATH, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_queries = len(records)
    correct_routing = 0
    failed_routing = []

    print(f"\nStarting Evaluation on {total_queries} queries...")
    start_time = time.time()

    # --- 2. INFERENCE LOOP ---
    for record in tqdm(records, desc="Scoring Databases"):
        query = record["user_query"]
        gold_db = record["gold_db"]
        candidates = record["candidate_contexts"]
        
        if not candidates:
            continue
            
        pairs = []
        db_ids = []
        
        for ctx in candidates:
            db_id = ctx["db_id"]
            status = ctx["execution_status"]
            error = ctx["execution_error"] or "None"
            sql = ctx["sql"]
            
            # The exact string format used during training
            evidence_text = f"Status: {status} | Error: {error} | Database: {db_id} | SQL: {sql}"
            
            pairs.append([query, evidence_text])
            db_ids.append(db_id)

        # --- 3. SCORING ---
        scores = model.predict(pairs)
        
        best_index = np.argmax(scores)
        predicted_db = db_ids[best_index]
        
        # --- 4. CHECK ACCURACY ---
        if predicted_db == gold_db:
            correct_routing += 1
        else:
            failed_routing.append({
                "query": query,
                "gold": gold_db,
                "predicted": predicted_db,
                "scores": {db: float(score) for db, score in zip(db_ids, scores)}
            })

    end_time = time.time()
    total_time = end_time - start_time

    # --- 5. THE FINAL SCORE REPORT ---
    print("\n" + "="*50)
    print(" 🚀 BIRD BENCHMARK - FINAL PIPELINE RESULTS 🚀")
    print("="*50)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Correctly Routed:        {correct_routing}")
    print(f"Total Time Taken:        {total_time:.2f} seconds")
    print(f"Average Time per Query:  {(total_time/total_queries)*1000:.2f} milliseconds")
    print("-" * 50)
    print(f"🏆 FINAL ACCURACY:        {(correct_routing / total_queries) * 100:.2f}%")
    print("="*50)

    with open("results/bird_failure_analysis.json", 'w') as f:
        json.dump(failed_routing, f, indent=4)
    print("\nFailure analysis saved to 'results/bird_failure_analysis.json'.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import json

# Set this to where your fixed JSONL file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.jsonl")

def analyze_wrong_queries():
    total_queries = 0
    wrong_queries = 0
    caught_by_mlp = 0  
    danger_zone = 0    
    
    print("==================================================")
    print("  PHASE 1 (LLM RERANKER) MISCLASSIFICATION AUDIT  ")
    print("==================================================")
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    with open(FILE_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            record = json.loads(line.strip())
            total_queries += 1
            
            gold_db = record["gold_db"]
            predicted_db = record["ranked_dbs"][0]
            
            # If Top-1 is not the Gold DB, it's an error
            if gold_db != predicted_db:
                wrong_queries += 1
                
                s1 = record["scores"][0]
                s2 = record["scores"][1]
                margin = s1 - s2
                
                # Check if our MLP logic (margin < 0.2) will catch this
                is_ambiguous = margin < 0.2
                
                if is_ambiguous:
                    caught_by_mlp += 1
                    status = "🟡 CAUGHT BY MLP -> Sent to Tier 3 SQL Execution"
                else:
                    danger_zone += 1
                    status = "🔴 FATAL ERROR -> Will slip past MLP as 'Unambiguous'"
                    
                # Find where the gold_db actually ranked
                try:
                    gold_rank = record["ranked_dbs"].index(gold_db) + 1
                except ValueError:
                    gold_rank = "Not in Top 15"
                    
                print(f"\nQ: {record['question']}")
                print(f"   Gold DB : {gold_db} (Ranked #{gold_rank})")
                print(f"   Pred DB : {predicted_db} (Score: {s1}) vs 2nd Place: {record['ranked_dbs'][1]} (Score: {s2})")
                print(f"   Margin  : {margin:.3f} | {status}")

    # Print Summary Statistics
    print("\n" + "="*50)
    print("  SUMMARY STATISTICS")
    print("="*50)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Total Wrong Queries    : {wrong_queries}")
    
    if wrong_queries > 0:
        caught_pct = (caught_by_mlp / wrong_queries) * 100
        danger_pct = (danger_zone / wrong_queries) * 100
        print(f"\nCaught by MLP (Margin < 0.2) : {caught_by_mlp} ({caught_pct:.1f}%)")
        print(f"Slipped Through (Margin ≥ 0.2): {danger_zone} ({danger_pct:.1f}%)")
        
        # Calculate theoretical maximum accuracy
        theoretical_max = ((total_queries - danger_zone) / total_queries) * 100
        print(f"\nTheoretical Max Cascade Accuracy: {theoretical_max:.2f}%")
        print("(If Tier 3 SQL Execution fixes 100% of the queries caught by the MLP)")

if __name__ == "__main__":
    analyze_wrong_queries()
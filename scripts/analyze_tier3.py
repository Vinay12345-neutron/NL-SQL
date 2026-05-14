#!/usr/bin/env python3
import os
import json
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "results", "spider_cascade_results.jsonl")

def analyze_tier3():
    total_t3 = 0
    t3_correct = 0
    
    t3_paths = Counter()
    t3_failures = Counter()
    gold_not_in_top_k = 0
    
    with open(CASCADE_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
            except:
                continue
                
            # Only look at Tier 3 queries
            if record.get("tier") != 3:
                continue
                
            total_t3 += 1
            gold_db = record["gold_db"]
            pred_db = record["predicted_db"]
            details = record.get("details", {})
            path = details.get("tier3_path", "unknown")
            
            t3_paths[path] += 1
            
            # Check if gold was even in the Top-K candidates considered by Tier 3
            # We assume Top-K was 5 based on 3_execution_cascade.py
            ce_ranked = record.get("ce_ranked_dbs", [])
            top_k = ce_ranked[:5] if ce_ranked else []
            
            is_correct = (gold_db == pred_db)
            if is_correct:
                t3_correct += 1
            else:
                if gold_db not in top_k:
                    gold_not_in_top_k += 1
                    t3_failures["Gold not in Top 5 candidates"] += 1
                else:
                    t3_failures[f"Failed in path: {path}"] += 1

    print("==================================================")
    print("  TIER 3 (EXECUTION CASCADE) DEEP DIVE")
    print("==================================================")
    print(f"Total Tier 3 Queries: {total_t3}")
    if total_t3 == 0:
        return
        
    print(f"Tier 3 Accuracy:      {t3_correct} / {total_t3} ({t3_correct/total_t3*100:.2f}%)")
    print(f"\nBreakdown by Tier 3 Resolution Path:")
    for path, count in t3_paths.items():
        print(f"  - {path}: {count} queries")
        
    print("\nFailure Analysis (Why did Tier 3 pick the wrong DB?):")
    for reason, count in t3_failures.most_common():
        print(f"  - {reason}: {count} queries")
        
    print(f"\nCritical bottleneck: In {gold_not_in_top_k} out of {total_t3 - t3_correct} failures ({(gold_not_in_top_k/(total_t3 - t3_correct)*100) if total_t3 != t3_correct else 0:.1f}%),")
    print("the correct database wasn't even in the Top 5 candidates passed to Tier 3!")

if __name__ == "__main__":
    analyze_tier3()

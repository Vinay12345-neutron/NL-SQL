#!/usr/bin/env python3
import json
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "spider_cascade_results.jsonl")

def main():
    if not os.path.exists(OUTPUT_PATH):
        print("No results found yet.")
        return

    results = []
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            results.append(json.loads(line))

    total         = len(results)
    if total == 0:
        return
        
    correct_count = sum(r["correct"] for r in results)
    cascade_r1    = correct_count / total * 100

    tier2_res     = [r for r in results if r["tier"] == 2]
    tier3_res     = [r for r in results if r["tier"] == 3]
    tier2_acc     = sum(r["correct"] for r in tier2_res) / len(tier2_res) * 100 if tier2_res else 0
    tier3_acc     = sum(r["correct"] for r in tier3_res) / len(tier3_res) * 100 if tier3_res else 0

    print("=" * 65)
    print("  PHASE 3 — CURRENT ACCURACY")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Value':>15}")
    print("-" * 65)
    print(f"  {'Total Queries Processed':<30} {total:>15}")
    print(f"  {'Tier 2 (Unambiguous)':<30} {len(tier2_res):>15}  ({len(tier2_res)/total*100:.1f}%)")
    print(f"  {'  → Tier 2 Accuracy':<30} {tier2_acc:>14.2f}%")
    print(f"  {'Tier 3 (Ambiguous)':<30} {len(tier3_res):>15}  ({len(tier3_res)/total*100:.1f}%)")
    print(f"  {'  → Tier 3 Accuracy':<30} {tier3_acc:>14.2f}%")
    print("-" * 65)
    print(f"  {'CASCADE R@1 (So Far)':<30} {cascade_r1:>14.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()

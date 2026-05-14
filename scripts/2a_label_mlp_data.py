#!/usr/bin/env python3
"""
Phase 2A: Label MLP Training Data (Ambiguity Detection)
========================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset
"""

import json
import os
import csv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.jsonl")
TOP15_PATH  = os.path.join(BASE_DIR, "data",    "spider_baseline1_top15.json")
OUTPUT_CSV  = os.path.join(BASE_DIR, "data",    "spider_mlp_training_data.csv")

MARGIN_THRESHOLD = 0.2   # s1 - s2 < threshold → Ambiguous

def main():
    # 1. Load LLM scores (JSONL format)
    print(f"Loading LLM reranker scores from:\n  {SCORES_PATH}\n")
    records = []
    with open(SCORES_PATH, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    print(f"Loaded {len(records)} records.\n")

    # 2. Load Baseline 1 (Qwen) Top-1 predictions
    with open(TOP15_PATH) as f:
        top15_data = json.load(f)

    qwen_top1_map = {item["question"]: item["top15_dbs"][0] for item in top15_data}

    # 3. Build CSV rows
    rows          = []
    ambiguous     = 0
    unambiguous   = 0

    unamb_qwen_correct = 0
    unamb_total        = 0
    amb_qwen_correct   = 0
    amb_total          = 0

    for rec in records:
        question    = rec["question"]
        gold_db     = rec["gold_db"]
        scores_list = rec["scores"]     # Now safely extracting the list of floats
        ranked_dbs  = rec["ranked_dbs"]

        qwen_top1 = qwen_top1_map.get(question, ranked_dbs[0] if ranked_dbs else "")

        # Ensure exactly 15 scores
        sorted_scores = scores_list[:15]
        while len(sorted_scores) < 15:
            sorted_scores.append(0.0)

        s1 = sorted_scores[0]
        s2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = s1 - s2

        label = 1 if margin < MARGIN_THRESHOLD else 0

        qwen_correct = int(qwen_top1 == gold_db)

        if label == 0:   # Unambiguous
            unambiguous          += 1
            unamb_total          += 1
            unamb_qwen_correct   += qwen_correct
        else:            # Ambiguous
            ambiguous            += 1
            amb_total            += 1
            amb_qwen_correct     += qwen_correct

        row = {
            **{f"s{i+1}": round(sorted_scores[i], 6) for i in range(15)},
            "margin"        : round(margin, 6),
            "label"         : label,
            "gold_db"       : gold_db,
            "qwen_top1"     : qwen_top1,
            "ce_predicted"  : ranked_dbs[0] if ranked_dbs else "",
            "qwen_correct"  : qwen_correct,
        }
        rows.append(row)

    # 4. Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [f"s{i+1}" for i in range(15)] + [
        "margin", "label", "gold_db", "qwen_top1", "ce_predicted", "qwen_correct"
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to:\n  {OUTPUT_CSV}\n")

if __name__ == "__main__":
    main()
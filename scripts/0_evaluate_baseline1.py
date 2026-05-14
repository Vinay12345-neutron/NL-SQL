#!/usr/bin/env python3
"""
Phase 0: Baseline 1 Evaluation — Dense Retrieval (Qwen Embeddings)
===================================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

Objective:
    Formalize the dense retrieval foundation by loading the pre-existing Qwen
    embedding results for the Spider test set, truncating each ranked list to the
    Top-15 candidates, and computing standard retrieval metrics.

Input:
    results/spider_retrieval_results.json   — Pre-computed Qwen Top-20 rankings

Output:
    data/spider_baseline1_top15.json        — Top-15 candidates per query

Metrics Reported:
    R@1, R@3, R@5, R@10, R@15, MRR

Expected:
    R@1 ≈ 63.98%  (from prior analysis on 5,116 test queries)
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH    = os.path.join(BASE_DIR, "results", "spider_retrieval_results.json")
OUTPUT_PATH   = os.path.join(BASE_DIR, "data", "spider_baseline1_top15.json")

TOP_K_CUTOFF  = 15   # We only keep Top-15 per query for use in subsequent phases


def compute_metrics(data: list, ks: list[int]) -> dict:
    """
    Compute Recall@k and MRR over a list of retrieval records.

    Each record must have:
        'question'      : str
        'gold_db'       : str
        'top15_dbs'     : list[str]  (ordered by retrieval rank)
    """
    total = len(data)
    hits  = {k: 0 for k in ks}
    mrr   = 0.0

    for item in data:
        gold  = item["gold_db"]
        ranks = item["top15_dbs"]

        if gold in ranks:
            idx = ranks.index(gold)          # 0-indexed rank
            mrr += 1.0 / (idx + 1)
            for k in ks:
                if idx < k:
                    hits[k] += 1

    metrics = {f"R@{k}": round(hits[k] / total * 100, 2) for k in ks}
    metrics["MRR"]      = round(mrr / total, 4)
    metrics["total"]    = total
    metrics["hits"]     = {f"R@{k}": hits[k] for k in ks}
    return metrics


def main():
    # ------------------------------------------------------------------
    # 1. Load the pre-existing Qwen Top-20 retrieval results
    # ------------------------------------------------------------------
    print(f"Loading Qwen retrieval results from:\n  {INPUT_PATH}\n")
    with open(INPUT_PATH, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} retrieval records.")

    # ------------------------------------------------------------------
    # 2. Truncate to Top-15 and build the output structure
    # ------------------------------------------------------------------
    output_data = []
    for item in raw_data:
        record = {
            "question"  : item["question"],
            "gold_db"   : item["gold_db"],
            # Keep only the first TOP_K_CUTOFF (15) candidates from the ranked list
            "top15_dbs" : item["retrieved_dbs"][:TOP_K_CUTOFF],
        }
        output_data.append(record)

    # ------------------------------------------------------------------
    # 3. Save to data/spider_baseline1_top15.json
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved Top-{TOP_K_CUTOFF} candidates to:\n  {OUTPUT_PATH}\n")

    # ------------------------------------------------------------------
    # 4. Compute and print metrics
    # ------------------------------------------------------------------
    ks      = [1, 3, 5, 10, 15]
    metrics = compute_metrics(output_data, ks)

    print("=" * 55)
    print("  PHASE 0 — BASELINE 1: Dense Retrieval (Qwen)  ")
    print("=" * 55)
    print(f"  Total Queries : {metrics['total']}")
    print("-" * 55)
    for k in ks:
        label = f"R@{k}"
        count = metrics["hits"][label]
        pct   = metrics[label]
        print(f"  {label:<6} : {pct:6.2f}%  ({count}/{metrics['total']})")
    print(f"  {'MRR':<6} : {metrics['MRR']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1C: Baseline 2 Inference — Bq/s Cross-Encoder Reranking
==============================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

Objective:
    Load the trained Bq/s Cross-Encoder (Phase 1B) and run inference over the
    Spider test set's Top-15 Qwen candidates (Phase 0). For each query, score
    all 15 candidate schemas and output an ordered list of probability scores.

    This produces "Baseline 2": Qwen Dense Retrieval + learned Bq/s Cross-Encoder
    reranking (without any SQL execution). Comparing against Baseline 1 quantifies
    the pure gain from the learned schema-matching step.

Input:
    data/spider_baseline1_top15.json          — Top-15 Qwen candidates per query
    processed_data/spider_tables.json         — Spider train DB schemas
    processed_data/spider_test_tables.json    — Spider test DB schemas
    models/spider_bqs_cross_encoder/          — Trained Bq/s cross-encoder

Output:
    results/spider_baseline2_scores.json
        One record per test query:
        {
            "question"       : str,
            "gold_db"        : str,
            "ranked_dbs"     : [str, ...],    # 15 DBs re-ordered by cross-encoder score
            "scores"         : {db_id: float},# Raw sigmoid scores for all 15 candidates
            "predicted_db"   : str,           # Top-1 prediction (highest score)
            "correct"        : bool
        }

Metrics Reported:
    R@1, R@3, R@5, R@10, R@15, MRR — Baseline 2 vs Baseline 1 side-by-side comparison
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP15_PATH   = os.path.join(BASE_DIR, "data",    "spider_baseline1_top15.json")
TABLES_PATHS = [
    os.path.join(BASE_DIR, "processed_data", "spider_tables.json"),
    os.path.join(BASE_DIR, "processed_data", "spider_test_tables.json"),
]
MODEL_PATH   = os.path.join(BASE_DIR, "models",  "spider_bqs_cross_encoder_roberta")
OUTPUT_PATH  = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.json")
MAX_LENGTH   = 512


# ---------------------------------------------------------------------------
# Schema Builder — must exactly match Phase 1A's build_schema_context()
# ---------------------------------------------------------------------------
def build_schema_context(db_entry: dict) -> str:
    db_id       = db_entry["db_id"]
    table_names = db_entry["table_names_original"]
    col_names   = db_entry["column_names_original"]

    table_cols = {i: [] for i in range(len(table_names))}
    for table_idx, col_name in col_names:
        if table_idx == -1:
            continue
        table_cols[table_idx].append(col_name)

    lines = []
    for i, tbl in enumerate(table_names):
        cols = table_cols.get(i, [])
        lines.append(f"  - {tbl}({', '.join(cols[:8])})")

    return f"Database: {db_id}\nTables:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Batched inference helper
# ---------------------------------------------------------------------------
def score_candidates(model, tokenizer, query: str,
                     schema_texts: list, device) -> np.ndarray:
    """Score query against all schema texts in a single batched forward pass."""
    queries  = [query] * len(schema_texts)
    enc = tokenizer(
        queries, schema_texts,
        truncation=True, padding=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(records: list, ks: list) -> dict:
    total = len(records)
    hits  = {k: 0 for k in ks}
    mrr   = 0.0
    for rec in records:
        gold   = rec["gold_db"]
        ranked = rec["ranked_dbs"]
        if gold in ranked:
            idx  = ranked.index(gold)
            mrr += 1.0 / (idx + 1)
            for k in ks:
                if idx < k:
                    hits[k] += 1
    return {
        **{f"R@{k}": round(hits[k] / total * 100, 2) for k in ks},
        "MRR"  : round(mrr / total, 4),
        "total": total,
        "hits" : {f"R@{k}": hits[k] for k in ks},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Load schemas
    print("Loading schemas...")
    all_schemas: dict[str, str] = {}
    for path in TABLES_PATHS:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for entry in json.load(f):
                all_schemas[entry["db_id"]] = build_schema_context(entry)
    print(f"  Loaded {len(all_schemas)} schemas.\n")

    # 2. Load Top-15 test candidates
    with open(TOP15_PATH) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test queries from Baseline 1.\n")

    # 3. Load model
    print(f"Loading Bq/s Cross-Encoder from:\n  {MODEL_PATH}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    # 4. Inference
    results       = []
    correct_count = 0
    no_schema     = 0

    for item in tqdm(test_data, desc="Scoring"):
        question   = item["question"]
        gold_db    = item["gold_db"]
        candidates = item["top15_dbs"]

        valid_dbs    = [db for db in candidates if db in all_schemas]
        schema_texts = [all_schemas[db] for db in valid_dbs]

        if not schema_texts:
            # Fallback: no schemas available — keep original Qwen ranking
            no_schema += 1
            pred = candidates[0] if candidates else ""
            results.append({
                "question": question, "gold_db": gold_db,
                "ranked_dbs": candidates, "scores": {},
                "predicted_db": pred, "correct": pred == gold_db,
            })
            if pred == gold_db:
                correct_count += 1
            continue

        scores_arr = score_candidates(model, tokenizer, question, schema_texts, device)

        # Sort candidates by score descending
        db_score_pairs = sorted(zip(valid_dbs, scores_arr), key=lambda x: x[1], reverse=True)
        ranked_dbs   = [db for db, _ in db_score_pairs]
        scores_dict  = {db: float(s) for db, s in db_score_pairs}
        predicted_db = ranked_dbs[0]
        correct      = (predicted_db == gold_db)

        if correct:
            correct_count += 1

        results.append({
            "question": question, "gold_db": gold_db,
            "ranked_dbs": ranked_dbs, "scores": scores_dict,
            "predicted_db": predicted_db, "correct": correct,
        })

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} records to:\n  {OUTPUT_PATH}\n")

    # 6. Metrics
    ks      = [1, 3, 5, 10, 15]
    metrics = compute_metrics(results, ks)

    # Baseline 1 reference (from Phase 0)
    b1 = {"R@1": 63.98, "R@3": 82.15, "R@5": 88.92, "R@10": 94.14, "R@15": 96.23, "MRR": 0.7454}

    print("=" * 68)
    print("  PHASE 1C — BASELINE 2: Bq/s Cross-Encoder Reranking  ")
    print("=" * 68)
    print(f"  {'Metric':<8} {'Baseline 1 (Qwen)':>22} {'Baseline 2 (Bq/s)':>22} {'Delta':>10}")
    print("-" * 68)
    for k in ks:
        lbl   = f"R@{k}"
        b1val = b1[lbl]
        b2val = metrics[lbl]
        delta = b2val - b1val
        sign  = "+" if delta >= 0 else ""
        print(f"  {lbl:<8} {b1val:>20.2f}%  {b2val:>20.2f}%  {sign}{delta:>+8.2f}%")
    print("-" * 68)
    b2mrr = metrics["MRR"]
    print(f"  {'MRR':<8} {b1['MRR']:>22.4f} {b2mrr:>22.4f}  {b2mrr - b1['MRR']:>+10.4f}")
    print("=" * 68)
    print(f"\n  Queries skipped (no schema): {no_schema}")
    print(f"  Total evaluated: {metrics['total']}")


if __name__ == "__main__":
    main()

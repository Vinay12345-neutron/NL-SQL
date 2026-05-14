#!/usr/bin/env python3
"""
Phase 1A (v2): Generate Bq/s Training Data with Hard Negative Mining
=====================================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

FIX from v1 (which used random negatives causing test distribution mismatch):
    In v1, we sampled 4 negatives randomly from all 206 databases.
    At test time, the cross-encoder sees Qwen Top-15 candidates — the 15 most
    semantically similar schemas to the query. This caused a severe training/test
    distribution mismatch, dropping R@1 by 5%.

FIX: Hard Negative Mining via TF-IDF Schema Similarity
    For each training query, we compute TF-IDF similarity between the query text
    and ALL database schema context strings. The 4 most similar schemas (excluding
    the gold DB) become the hard negatives. This replicates the test-time condition
    where all candidates look similar to the query, forcing the model to learn
    fine-grained structural discrimination.

Input:
    processed_data/spider_route_train.json   — Spider train queries + gold db_ids
    processed_data/spider_tables.json        — Spider train DB schemas
    processed_data/spider_test_tables.json   — Spider test DB schemas

Output:
    data/spider_bqs_train_pairs.jsonl        — JSONL file with hard-negative pairs

Metrics Reported:
    Total positive/negative pairs, average similarity of hard negatives.
"""

import json
import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "processed_data", "spider_route_train.json")
TABLES_PATHS    = [
    os.path.join(BASE_DIR, "processed_data", "spider_tables.json"),
    os.path.join(BASE_DIR, "processed_data", "spider_test_tables.json"),
]
OUTPUT_PATH     = os.path.join(BASE_DIR, "data", "spider_bqs_train_pairs.jsonl")

NUM_NEGATIVES   = 4
RANDOM_SEED     = 42


# ---------------------------------------------------------------------------
# Schema Builder (identical to 1c_baseline2_inference.py for consistency)
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
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load all schemas
    # ------------------------------------------------------------------
    print("Loading schemas...")
    all_schemas: dict[str, str] = {}
    for path in TABLES_PATHS:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        with open(path) as f:
            for entry in json.load(f):
                all_schemas[entry["db_id"]] = build_schema_context(entry)

    all_db_ids   = list(all_schemas.keys())
    all_contexts = [all_schemas[db] for db in all_db_ids]
    print(f"  Loaded {len(all_db_ids)} schemas.\n")

    # ------------------------------------------------------------------
    # 2. Build a TF-IDF vectorizer over ALL schema context strings.
    #    This allows us to compute query-to-schema similarity quickly
    #    without running any neural model.
    # ------------------------------------------------------------------
    print("Building TF-IDF index over all schemas for hard negative mining...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),   # Unigrams + bigrams to capture table/column names
        sublinear_tf=True,    # Log-scale TF to reduce dominance of frequent terms
    )
    schema_matrix = vectorizer.fit_transform(all_contexts)  # shape: (206, vocab)
    print(f"  TF-IDF matrix shape: {schema_matrix.shape}\n")

    # ------------------------------------------------------------------
    # 3. Load train queries
    # ------------------------------------------------------------------
    print(f"Loading Spider train queries from:\n  {TRAIN_DATA_PATH}\n")
    with open(TRAIN_DATA_PATH) as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training queries.\n")

    # ------------------------------------------------------------------
    # 4. Generate pairs with hard negatives
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total_positives = 0
    total_negatives = 0
    skipped         = 0
    avg_sim_list    = []  # Track average hard-negative similarity (quality check)

    with open(OUTPUT_PATH, "w") as out_f:
        for item in train_data:
            question = item["question"]
            gold_db  = item["db_id"]

            if gold_db not in all_schemas:
                skipped += 1
                continue

            # ---- POSITIVE PAIR ----------------------------------------
            out_f.write(json.dumps({
                "query"          : question,
                "schema_context" : all_schemas[gold_db],
                "db_id"          : gold_db,
                "label"          : 1.0,
            }) + "\n")
            total_positives += 1

            # ---- HARD NEGATIVE MINING ----------------------------------
            # Score this query against ALL schema contexts using TF-IDF
            query_vec = vectorizer.transform([question])           # shape: (1, vocab)
            sims      = cosine_similarity(query_vec, schema_matrix).flatten()  # (206,)

            # Exclude the gold DB index, then take the top-4 most similar
            gold_idx  = all_db_ids.index(gold_db)
            sims[gold_idx] = -1.0    # Mask the gold DB out

            top_neg_indices = np.argsort(sims)[::-1][:NUM_NEGATIVES]
            hard_neg_dbs    = [all_db_ids[i] for i in top_neg_indices]
            avg_sim_list.append(np.mean(sims[top_neg_indices]))

            for dist_db in hard_neg_dbs:
                out_f.write(json.dumps({
                    "query"          : question,
                    "schema_context" : all_schemas[dist_db],
                    "db_id"          : dist_db,
                    "label"          : 0.0,
                }) + "\n")
                total_negatives += 1

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    total_pairs = total_positives + total_negatives
    print("=" * 60)
    print("  PHASE 1A (v2) — Bq/s Hard-Negative Training Data  ")
    print("=" * 60)
    print(f"  Training Queries Processed : {len(train_data) - skipped}")
    print(f"  Skipped (no schema found)  : {skipped}")
    print(f"  Positive Pairs (label=1.0) : {total_positives}")
    print(f"  Negative Pairs (label=0.0) : {total_negatives}  ← HARD negatives (TF-IDF top-4)")
    print(f"  Positive-to-Negative Ratio : 1 : {NUM_NEGATIVES}")
    print(f"  Total Pairs Generated      : {total_pairs}")
    print(f"  Avg Hard-Neg Similarity    : {np.mean(avg_sim_list):.4f}  (higher = harder)")
    print(f"  Output Saved To            : {OUTPUT_PATH}")
    print("=" * 60)
    print("\n  ✅ Ready for Phase 1B retraining.")


if __name__ == "__main__":
    main()

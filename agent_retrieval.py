"""
Adaptive Agentic Retrieval v6 — LLM Re-Ranking + Execution Evidence
=====================================================================
Key lessons from v1-v5:
  - v1 (LLM only): Spider R@1=0.70 — LLM schema reasoning WORKS
  - v2 (keyword+LLM): BIRD R@1=0.84 — best BIRD result
  - v4 (exec-first): Spider R@1=0.04 — execution on empty DBs = disaster
  - v5 (adaptive): Spider R@1=0.06 — preserving baseline doesn't help
                    because baseline is already 0.06 for these 50 queries

Architecture:
  1. Dense retrieval → Top-5 candidates
  2. For each candidate: check if DB has data
  3. For populated DBs: extract entities + run probes → build evidence string
  4. ONE LLM call: rerank(query, candidates, evidence) → picks best DB
  5. Return re-ranked list

Why this works:
  - Spider (empty DBs): LLM reasons about schema structure (keyword table,
    conference table, linking tables). This IS what got R@1=0.70 in v1.
  - BIRD (populated DBs): LLM gets execution evidence ("VLDB found in venue
    table") which strengthens its decision.
"""

import os
import re
import json
import time
import sqlite3
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from baseline_retrieval import (
    EmbeddingModel, load_schemas, calculate_metrics,
    save_metrics_to_csv, DATA_DIR, OUTPUT_DIR, run_retrieval
)
from transducer import AgenticTransducer

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


# ---------------------------------------------------------------------------
# Helper: Find SQLite file for a db_id
# ---------------------------------------------------------------------------
def get_db_path(db_id: str):
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, "spider_data", "database", db_id, f"{db_id}.sqlite"),
        os.path.join(base, "spider_data", "test_database", db_id, f"{db_id}.sqlite"),
        os.path.join(base, "dev_20240627", "dev_databases", db_id, f"{db_id}.sqlite"),
        os.path.join(base, "train", "train_databases", db_id, f"{db_id}.sqlite"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Helper: Fast check if a database has any data
# ---------------------------------------------------------------------------
_data_cache = {}

def db_has_data(db_id: str) -> bool:
    if db_id in _data_cache:
        return _data_cache[db_id]
    
    db_path = get_db_path(db_id)
    if not db_path:
        _data_cache[db_id] = False
        return False
    
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (tbl,) in cur.fetchall():
            try:
                cur.execute(f"SELECT 1 FROM [{tbl}] LIMIT 1")
                if cur.fetchone():
                    conn.close()
                    _data_cache[db_id] = True
                    return True
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
    
    _data_cache[db_id] = False
    return False


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_dataset(
    dataset_name: str,
    json_path: str,
    model: EmbeddingModel,
    transducer: AgenticTransducer,
    schemas: dict,
    limit: int = None
):
    if not os.path.exists(json_path):
        print(f"\nSkipping {dataset_name}: File not found at {json_path}")
        return

    print(f"\n=== Processing {dataset_name} (LLM-Rerank v6) ===")
    with open(json_path, 'r') as f:
        data = json.load(f)

    queries  = [item['question'] for item in data]
    gold_dbs = [item['db_id']    for item in data]

    # Step 1: Dense retrieval → Top-10 candidates
    top_dbs = run_retrieval(model, queries, gold_dbs, schemas, k=10)

    if limit:
        print(f"Limiting to first {limit} queries.")
        queries  = queries[:limit]
        gold_dbs = gold_dbs[:limit]
        top_dbs  = top_dbs[:limit]

    final_results = []
    stats = {
        "total": 0, "llm_reranked": 0, "probed": 0,
        "probe_evidence": 0, "llm_errors": 0
    }

    for i, (q, gold, candidates) in tqdm(
        enumerate(zip(queries, gold_dbs, top_dbs)),
        total=len(queries),
        desc=f"Routing {dataset_name}"
    ):
        stats["total"] += 1
        eval_candidates = candidates[:5]
        
        time.sleep(3)  # Rate limit for LLM calls

        # ------------------------------------------------------------------
        # Step 2: For populated DBs, gather execution evidence
        # ------------------------------------------------------------------
        exec_evidence = {}
        entities = []
        
        # Check which candidates have data
        populated = [db for db in eval_candidates if db_has_data(db)]
        
        if populated:
            # Extract entities (1 LLM call) — only if there are DBs to probe
            try:
                entities = transducer.extract_entities(q)
            except Exception:
                entities = []

            # Run probes on populated DBs
            for db in populated:
                db_path = get_db_path(db)
                probes = transducer.generate_probes(q, entities, db)
                
                if probes and db_path:
                    score, hits = transducer.execute_probes(probes, db_path)
                    stats["probed"] += 1
                    
                    if score > 0:
                        # Build human-readable evidence string
                        hit_probes = [p for p, h in zip(probes, hits) if h]
                        evidence_parts = []
                        for hp in hit_probes[:3]:
                            # Extract table and condition from probe SQL
                            m = re.search(r'FROM\s+\[?(\w+)\]?\s+WHERE\s+(.+?)\s+LIMIT', hp)
                            if m:
                                evidence_parts.append(f"Found data in {m.group(1)} matching {m.group(2)}")
                        exec_evidence[db] = "; ".join(evidence_parts)
                        stats["probe_evidence"] += 1

        # ------------------------------------------------------------------
        # Step 3: LLM re-ranking (the core — ONE call)
        # ------------------------------------------------------------------
        try:
            reranked = transducer.rerank(q, eval_candidates, exec_evidence)
            stats["llm_reranked"] += 1
        except Exception as e:
            reranked = list(eval_candidates)
            stats["llm_errors"] += 1

        selected_db = reranked[0]
        
        # Build full ranked list (top-5 reranked + remaining 6-10)
        ranked_dbs = list(reranked)
        for db in candidates[5:]:
            if db not in ranked_dbs:
                ranked_dbs.append(db)

        final_results.append({
            "question":        q,
            "gold_db":         gold,
            "selected_db":     selected_db,
            "retrieved_dbs":   ranked_dbs,
            "entities":        entities,
            "exec_evidence":   exec_evidence,
            "retrieval_order": eval_candidates,
        })

    # ------------------------------------------------------------------
    # Save and compute metrics
    # ------------------------------------------------------------------
    print(f"\n=== {dataset_name} Stats ===")
    print(json.dumps(stats, indent=2))

    safe_name = dataset_name.lower().replace(" ", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"{safe_name}_agentic_results.json")
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved to {out_path}")

    metrics = calculate_metrics(
        f"{dataset_name}-LLMRerankV6",
        queries, gold_dbs,
        [r['retrieved_dbs'] for r in final_results]
    )
    save_metrics_to_csv([metrics])


def main():
    try:
        model = EmbeddingModel(MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return

    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded.")
        return

    transducer = AgenticTransducer(schemas)

    process_dataset(
        "Spider",
        os.path.join(DATA_DIR, "spider_route_test.json"),
        model, transducer, schemas, limit=50
    )

    process_dataset(
        "BIRD",
        os.path.join(DATA_DIR, "bird_route_test.json"),
        model, transducer, schemas, limit=50
    )


if __name__ == "__main__":
    main()

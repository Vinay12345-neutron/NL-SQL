"""
Agentic Retrieval Script (Paper Implementation)
Extends baseline_retrieval.py with an Agent-Based Transducer loop.
Supports both Spider and BIRD datasets.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Import from baseline to reuse
from baseline_retrieval import EmbeddingModel, load_schemas, calculate_metrics, save_metrics_to_csv, DATA_DIR, OUTPUT_DIR, run_retrieval

from transducer import AgenticTransducer, load_dotenv

# Load Env
load_dotenv()

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

def process_dataset(dataset_name: str, json_path: str, model: EmbeddingModel, transducer: AgenticTransducer, schemas: Dict[str, str], limit: int = None):
    """
    Runs the Agentic Retrieval loop for a given dataset.
    """
    if not os.path.exists(json_path):
        print(f"\nSkipping {dataset_name}: File not found at {json_path}")
        return

    print(f"\n=== Processing {dataset_name} (Agentic) ===")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    queries = [item['question'] for item in data]
    gold_dbs = [item['db_id'] for item in data]
    
    # 1. Initial Retrieval (Baseline Top-10)
    # We fetch Top-10 to give the agent context
    top_dbs = run_retrieval(model, queries, gold_dbs, schemas, k=10)
    
    final_results = []
    action_stats = {"Normal": 0, "Ambiguous": 0, "Incomplete": 0}
    
    # Apply limit if provided (e.g. for testing)
    if limit:
        print(f"Limiting to first {limit} queries for testing.")
        queries = queries[:limit]
        gold_dbs = gold_dbs[:limit]
        top_dbs = top_dbs[:limit]
    
    import time
    for i, (q, gold, candidates) in tqdm(enumerate(zip(queries, gold_dbs, top_dbs)), total=len(queries), desc=f"Transducing {dataset_name}"):
        
        # Log progress for debugging stalls
        # print(f"Processing Query {i+1}/{len(queries)}: {q[:50]}...")
        
        # Rate Limit handling: Pause between queries
        time.sleep(2) 
        
       # Context for Agent: The Classifier only needs Top 3 to check for ambiguity.
        # Context for Agent: Give it Top 5 to ensure the correct DB is in the prompt
        eval_candidates = candidates[:5]
        
        # Step 1: Classify
        try:
            label, reasoning = transducer.classify(q, eval_candidates)
        except Exception as e:
            label = "Normal"
            reasoning = f"Error: {e}"
        
        action_stats[label] = action_stats.get(label, 0) + 1
        
        resolved_query = q
        selected_db = candidates[0]
        method = "baseline_top1"
        reasoning = "Normal query."
        
        if label == "Normal":
            # Let the Agent pick from Top 5
            try:
                selected_db = transducer.answer(q, eval_candidates)
                method = "agent_normal_select"
            except Exception as e:
                reasoning = f"Answer Failed: {e}"
            
        elif label in ["Ambiguous", "Incomplete"]:
            # Step 2: Resolve & Vector Search
            try:
                resolved_query = transducer.resolve(q, eval_candidates, label)
                method = "agent_resolved_vector_search"
                reasoning = f"Resolved from {label}. New query: {resolved_query}"
                
                new_top_dbs = run_retrieval(model, [resolved_query], [gold], schemas, k=3)[0]
                selected_db = new_top_dbs[0]
                
            except Exception as e:
                reasoning = f"Resolution/Answer Failed: {e}"
        
        # Final Fallback check
        if selected_db not in candidates:
            # Agent might hallucinate an ID. Fallback to vector top-1
            # print(f"[Warn] Agent predicted {selected_db} not in candidates. Falling back.")
            selected_db = candidates[0]
            
        # Construct result
        # Ensure selected is first in retrieved list for metrics
        final_list = [selected_db]
        for c in candidates:
            if c != selected_db:
                final_list.append(c)
        
        result_entry = {
            "question": q,
            "original_question": q,
            "resolved_question": resolved_query,
            "gold_db": gold,
            "retrieved_dbs": final_list,
            "selected_db": selected_db,
            "method": method,
            "label": label,
            "reasoning": reasoning
        }
        final_results.append(result_entry)
        
        # Incremental Save every 50 queries
        if len(final_results) % 50 == 0:
             safe_name = dataset_name.lower().replace(" ", "_")
             output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_agentic_results_partial.json")
             with open(output_path, 'w') as f:
                 json.dump(final_results, f, indent=2)
             # print(f"Saved partial results to {output_path}")
        
    # Stats
    print(f"\n=== {dataset_name} Agent Action Stats ===")
    print(json.dumps(action_stats, indent=2))
    
    # Save
    safe_name = dataset_name.lower().replace(" ", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_agentic_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
        
    # Metrics
    metrics = calculate_metrics(f"{dataset_name}-Agentic", queries, gold_dbs, [r['retrieved_dbs'] for r in final_results])
    save_metrics_to_csv([metrics])

def main():
    # 1. Initialize Baseline Model
    try:
        model = EmbeddingModel(MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        return

    # 2. Load Schemas
    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded.")
        return
        
    # 3. Initialize Agentic Transducer
    transducer = AgenticTransducer(schemas)
    
    # Process Spider-Route
    process_dataset("Spider", os.path.join(DATA_DIR, "spider_route_test.json"), model, transducer, schemas, limit=50)
    
    # Process Bird-Route
    process_dataset("BIRD", os.path.join(DATA_DIR, "bird_route_test.json"), model, transducer, schemas, limit=50)

if __name__ == "__main__":
    main()

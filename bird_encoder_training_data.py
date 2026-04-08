#!/usr/bin/env python3
"""
Merge BIRD Misclassified + BIRD Easy subsets into final training data.
"""

import json
import os
import random

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MISC_PATH = os.path.join(RESULTS_DIR, "bird_misclassified_pipeline.jsonl")
EASY_PATH = os.path.join(RESULTS_DIR, "bird_easy_queries_pipeline.jsonl")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "bird_cross_encoder_training_data.jsonl")

def main():
    all_records = []
    
    # Load misclassified
    if os.path.exists(MISC_PATH):
        with open(MISC_PATH, 'r') as f:
            all_records.extend([json.loads(l) for l in f if l.strip()])
    else:
        print(f"WARNING: Could not find {MISC_PATH}")

    # Load easy
    if os.path.exists(EASY_PATH):
        with open(EASY_PATH, 'r') as f:
            all_records.extend([json.loads(l) for l in f if l.strip()])
    else:
        print(f"WARNING: Could not find {EASY_PATH}")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_records)

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ BIRD Training Set Created: {OUTPUT_PATH} ({len(all_records)} total records)")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores.jsonl")
TEMP_PATH = os.path.join(BASE_DIR, "results", "spider_baseline2_scores_temp.jsonl")

def fix_scores():
    print(f"Fixing scores format in {FILE_PATH}...")
    fixed_count = 0
    
    with open(FILE_PATH, 'r') as infile, open(TEMP_PATH, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            
            # If the scores are currently a dictionary, we fix them
            if isinstance(record.get("scores"), dict):
                # Map the scores perfectly to the order of ranked_dbs
                fixed_scores_list = [record["scores"][db] for db in record["ranked_dbs"]]
                record["scores"] = fixed_scores_list
                fixed_count += 1
                
            outfile.write(json.dumps(record) + "\n")
            
    # Overwrite the original file with the fixed temp file
    os.replace(TEMP_PATH, FILE_PATH)
    print(f"✅ Successfully fixed {fixed_count} records. The file is now ready for Phase 2!")

if __name__ == "__main__":
    fix_scores()
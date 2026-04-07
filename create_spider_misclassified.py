import json
import os

input_file = "results/spider_retrieval_results.json"
output_file = "processed_data/spider_misclassified.json"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

with open(input_file, 'r') as f:
    data = json.load(f)

misclassified = []
for item in data:
    gold_db = item.get("gold_db")
    retrieved_dbs = item.get("retrieved_dbs", [])
    
    # R@1 failure means the top prediction is NOT the gold database
    if not retrieved_dbs or retrieved_dbs[0] != gold_db:
        # Reformat it to exactly match the structure the pipeline expects
        misclassified.append({
            "db_id": gold_db,
            "question": item.get("question")
        })

with open(output_file, 'w') as f:
    json.dump(misclassified, f, indent=2)

print(f"Extracted {len(misclassified)} misclassified queries out of {len(data)} total queries.")
print(f"Saved to {output_file}")

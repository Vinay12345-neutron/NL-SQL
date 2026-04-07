import json
import os

def process_dataset(name, route_test_path, retrieval_results_path, output_path, extract_keys=None):
    with open(route_test_path, 'r') as f:
        orig_data = json.load(f)
        
    with open(retrieval_results_path, 'r') as f:
        results = json.load(f)
        
    # Map questions to correctness so we don't rely strictly on ordering just in case
    # A query is correct if the top retrieved DB matches the gold DB
    correct_questions = set()
    for res in results:
        gold_db = res.get("gold_db")
        retrieved = res.get("retrieved_dbs", [])
        if retrieved and retrieved[0] == gold_db:
            correct_questions.add(res.get("question"))
            
    correct_subset = []
    
    for item in orig_data:
        # Match originally loaded queries
        if item.get("question") in correct_questions:
            if extract_keys:
                extracted = {k: item[k] for k in extract_keys if k in item}
                correct_subset.append(extracted)
            else:
                correct_subset.append(item)
                
    with open(output_path, 'w') as f:
        json.dump(correct_subset, f, indent=2)
        
    print(f"{name}: Extracted {len(correct_subset)} correctly classified queries (saved to {output_path})")

if __name__ == '__main__':
    # 1. BIRD
    # We want to match bird_misclassified.json which had db_id, question, evidence, SQL
    # But dumping the whole original object is robust. Let's dump the whole object.
    process_dataset(
        name="BIRD",
        route_test_path="processed_data/bird_route_test.json",
        retrieval_results_path="results/bird_retrieval_results.json",
        output_path="processed_data/bird_correct.json"
    )
    
    # 2. Spider
    # To match spider_misclassified.json which only has db_id and question, we can extract just those.
    # The original spider route data has db_id, question, and query(SQL). I'll keep the same fields just like we did.
    process_dataset(
        name="SPIDER",
        route_test_path="processed_data/spider_route_test.json",
        retrieval_results_path="results/spider_retrieval_results.json",
        output_path="processed_data/spider_correct.json",
        extract_keys=["db_id", "question"]
    )

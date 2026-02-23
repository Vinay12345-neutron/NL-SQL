
import json
import pandas as pd

def check_metrics():
    with open('results/spider_agentic_results.json', 'r') as f:
        data = json.load(f)
        
    baseline_correct = 0
    agent_correct = 0
    total = len(data)
    
    for item in data:
        gold = item['gold_db']
        # Baseline choice is the first one in retrieved_dbs
        # But wait, in agent_retrieval.py updates, I forced selected_db to be at index 0 of retrieved_dbs!
        # So I lost the original baseline info in the JSON unless I can infer it.
        # retrieved_dbs = [selected_db] + [c for c in candidates if c != selected_db]
        
        # However, I can look at "retrieved_dbs" vs "selected_db".
        # If I want to know the ORIGINAL baseline top-1, I need to know which one it was.
        # But since I re-ordered the list in the JSON output, I can't be 100% sure which one was originally first
        # UNLESS the agent picked the same one.
        
        # Actually, let's look at the logs or valid logic.
        # The only way to know baseline performance is to re-run or infer.
        # But wait! I printed "retrieved_dbs" which has 10 items.
        # Usually checking the *set* difference might help.
        
        # Let's assume for a moment that if label="Normal" and method="agent_normal_select", 
        # the agent *might* have picked something different from baseline top-1 
        # (since I called selected_db = answer(...)).
        
        pass

    # Better approach: 
    # Just calculate R@1 for Agent now.
    for item in data:
        if item['selected_db'] == item['gold_db']:
            agent_correct += 1
            
    print(f"Total: {total}")
    print(f"Agent R@1: {agent_correct/total:.2f}")
    
    # To check baseline, I should have saved "original_top1".
    # Since I didn't, I will modify agent_retrieval to save it next time.
    # But for now, let's look at the 'label' distribution.

    stats = {}
    for item in data:
        l = item['label']
        stats[l] = stats.get(l, 0) + 1
    print("Labels:", stats)
    
    # Check specific failures
    print("\nFailures (First 10):")
    for i, item in enumerate(data[:10]):
        if item['selected_db'] != item['gold_db']:
            print(f"{i}. Q: {item['question']}")
            print(f"   Gold: {item['gold_db']} | Agent: {item['selected_db']} | Label: {item['label']}")

if __name__ == "__main__":
    check_metrics()

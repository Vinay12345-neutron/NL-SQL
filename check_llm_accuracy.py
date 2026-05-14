import json
import os

filepath = 'results/spider_baseline2_scores.jsonl'
if not os.path.exists(filepath):
    print("File not found:", filepath)
    exit()

r1 = r3 = r5 = r10 = r15 = 0
mrr = 0.0
n = 0

with open(filepath, 'r') as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        n += 1
        gold = item['gold_db']
        ranked = item['ranked_dbs']
        if gold in ranked:
            idx = ranked.index(gold)
            if idx == 0: r1 += 1
            if idx < 3: r3 += 1
            if idx < 5: r5 += 1
            if idx < 10: r10 += 1
            if idx < 15: r15 += 1
            mrr += 1.0 / (idx + 1)

if n == 0:
    print("No queries processed yet.")
else:
    print(f"--- Live Metrics ({n} queries) ---")
    print(f'R@1:  {r1/n*100:.2f}%')
    print(f'R@3:  {r3/n*100:.2f}%')
    print(f'R@5:  {r5/n*100:.2f}%')
    print(f'R@10: {r10/n*100:.2f}%')
    print(f'R@15: {r15/n*100:.2f}%')
    print(f'MRR:  {mrr/n:.4f}')

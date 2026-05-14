import json
try:
    with open('results/spider_baseline2_scores.json') as f:
        data = json.load(f)
    r1 = r3 = r5 = r10 = r15 = 0
    mrr = 0.0
    for item in data:
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
    n = len(data)
    print(f'R@1: {r1/n*100:.2f}%')
    print(f'R@3: {r3/n*100:.2f}%')
    print(f'R@5: {r5/n*100:.2f}%')
    print(f'R@10: {r10/n*100:.2f}%')
    print(f'R@15: {r15/n*100:.2f}%')
    print(f'MRR: {mrr/n:.4f}')
except Exception as e:
    print("Error:", e)

import json

file_path = 'results/full_ambiguos_pipeline.jsonl'
lines = []
with open(file_path, 'r') as f:
    for line in f:
        if line.strip(): lines.append(line.strip())

seen = set()
unique = []
for line in lines:
    q = json.loads(line).get('user_query')
    if q not in seen:
        seen.add(q)
        unique.append(line)

with open(file_path, 'w') as f:
    for line in unique:
        f.write(line + '\n')

print(f'Cleaned up! Removed {len(lines) - len(unique)} duplicate lines. Current unique count: {len(unique)}')

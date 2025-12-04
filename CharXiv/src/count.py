import json

data = json.load(open("/data/zhengyue_zhao/workspace/nanxi/WiseReasoning/eval/results/scores-Mimo_TiS-reasoning_val.json"))

total_score = 0
for key, item in data.items():
    total_score += item['score']
    
print(total_score / len(data))
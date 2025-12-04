import json
import re
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from models.llama_guard import (
    LlamaGuard,
    DEFAULT_MODEL_PATH,
    parse_guard_output,
    extract_response_after_think,
    build_messages_for_text,
    init_model_and_processor,
    generate_guard
)

# Functions are now imported from models.llama_guard

data = json.load(open("/data/zhengyue_zhao/workspace/nanxi/WiseReasoning/results/model_jv_responses.json", "r"))

all_cates = set()

unsafe_items = []
processor, model, torch_device = init_model_and_processor(DEFAULT_MODEL_PATH)
for item in tqdm(data):
    all_cates.add(item["category"])
    resp_text = item["response"]
    processed_text = extract_response_after_think(str(resp_text))
    messages = build_messages_for_text(str(resp_text))  # build_messages_for_text now handles think tag extraction
    raw = generate_guard(processor, model, messages, torch_device)
    label, codes = parse_guard_output(raw)
    unsafe = label == "unsafe"
    if unsafe:
        item["llama_guard_label"] = codes
        unsafe_items.append(item)

with open("results/jv_judge_unsafe.json", "w") as f:
    json.dump(unsafe_items, f, indent=2)

print(all_cates)
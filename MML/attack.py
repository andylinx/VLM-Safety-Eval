import numpy as np
from PIL import Image
import os
import json
import random
import argparse
from tqdm import tqdm
from const import *
from utils import (get_jailbreak_score, query_claude, query_base, list_dir)
from eval import eval
from models import VLLMClient


dataformat2prompt = {
    'images_wr': wr_game_prompt,
    'images_mirror': mirror_game_prompt,
    'images_base64': base64_game_prompt,
    'images_rotate': rotate_game_prompt,
}

idx2defense = ['before', 'middle', 'after']
def random_shuffle_sentence(sentence):
    ssp = sentence.split()
    random.shuffle(ssp)
    return ssp

def query(model_id, image_path, prompt, api_key, use_vllm=False, vllm_client: VLLMClient = None):
    """
    Query the target model.

    - If use_vllm is True, route the request to a vLLM-served model (OpenAI-compatible API),
      typically the one started via `serve_kimi.sh` with `--served-model-name model`.
    - Otherwise, fall back to the original cloud model backends.
    """
    if use_vllm:
        if vllm_client is None:
            raise ValueError("vllm_client must be provided when use_vllm is True")
        result = vllm_client.generate_response(
            prompt=prompt,
            image_paths=image_path,
            model="model"
        )
        return result["response"]

    if "gpt" in model_id or 'qwen' in model_id:
        response = query_base(model_id, image_path, prompt, api_key)
    elif model_id == 'claude':
        response = query_claude(image_path, prompt, api_key)
    else:
        raise ValueError(f"no matched model named {model_id}")
    return response
    
    
def main():
    parser = argparse.ArgumentParser()
    # Note: Azure OpenAI credentials are required for GPT models and scoring.
    # Azure OpenAI uses: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    # For Qwen models (DashScope), OPENAI_API_KEY is still used.
    api_key = os.environ.get("OPENAI_API_KEY", "")  # Used for Qwen/DashScope
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    
    # Check if Azure OpenAI credentials are set (required for GPT models and scoring)
    if not azure_endpoint or not azure_api_key:
        raise ValueError(
            "Azure OpenAI credentials are required. "
            "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables. "
            "Even when using --use-vllm for the target model, "
            "Azure OpenAI credentials are needed for the judge/scoring function (get_jailbreak_score)."
        )
    parser.add_argument('--save-dir', type=str, default="./save_dir/")
    parser.add_argument('--dataset', type=str, choices=['safebench', 'mm-safebench', 'hades'], default='hades')
    # Allow arbitrary target model name so that locally served/vLLM models
    # (e.g. served via `serve_kimi.sh` with name `model`) can be evaluated.
    parser.add_argument('--target-model', type=str, default="glm")
    parser.add_argument('--image-format', type=str, default="images_wr")
    parser.add_argument('--use-vllm', action='store_true',
                        help='Use vLLM-served model via OpenAI-compatible API instead of cloud backends')
    parser.add_argument('--vllm-url', type=str, default="http://localhost:8001",
                        help='Base URL of the vLLM server (default: http://localhost:8001)')

    parser.add_argument('--defense', type=int, default=None)
    parser.add_argument('--data-path', type=str, default='./dataset')
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_path, args.dataset)
        
    sub_class = list_dir(data_path)

    # Initialize vLLM client if requested
    vllm_client = None
    if args.use_vllm:
        vllm_client = VLLMClient(base_url=args.vllm_url)
    
    # Progress bar for subclasses
    for sc in tqdm(sub_class, desc="Processing subclasses"):   
        if args.image_format == 'images_qr' and args.dataset == 'mm-safebench':
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.jpg")
        elif args.dataset == 'hades':
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.jpg")
        else:
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.png")

        data_info = json.load(open(os.path.join(data_path, sc, "data.json")))
        result_save_path = os.path.join(args.save_dir, args.dataset, args.image_format+ (f'_defense_{idx2defense[args.defense]}'  if args.defense is not None else ''), args.target_model)
        eval_path = result_save_path
        print(result_save_path)
        os.makedirs(result_save_path, exist_ok=True)
        result_save_path = os.path.join(result_save_path, f"{sc}_result.json")
        try:
            result = json.load(open(result_save_path))
        except Exception as e:
            result = []
        
        # Progress bar for items in each subclass
        total_items = len(data_info)
        start_idx = len(result)
        
        pbar = tqdm(total=total_items, desc=f"Processing {sc}", initial=start_idx, leave=False)
        for idx, item in enumerate(data_info):
            if idx < len(result):
                pbar.update(1)
                continue
                
            score_list = []
            res_list = []
            if "images_wr" in args.image_format:
                question = wr_game_prompt.format(item["replace_map"], random_shuffle_sentence(item["original_prompt"]))
            elif args.image_format in dataformat2prompt:
                question = dataformat2prompt[args.image_format].format(random_shuffle_sentence(item["original_prompt"]))
            elif args.image_format == 'images' and args.dataset=='hades':
                question = item["hades_prompt"]
            elif args.image_format == 'images_qr':
                question = item["qr_prompt"]
            elif args.image_format == 'images_figstep':
                question = fs_prompt
            if args.defense==0:
                question = static_defense_pre + question
            elif args.defense==1:
                question = static_defense_m + question + static_defense_m
            elif args.defense==2:
                question = question + static_defense_sub
            
            # Update progress bar with current item info
            pbar.set_postfix({"item_id": item["id"], "score": "..."})
            
            for attempt in range(5):
                # Get image path and convert to absolute path for vLLM
                image_path = image_path_format.format(f'{item["id"]}')
                if args.use_vllm:
                    # vLLM requires absolute paths for --allowed-local-media-path
                    image_path = os.path.abspath(image_path)
                
                response = query(
                    args.target_model,
                    image_path,
                    question,
                    api_key,
                    use_vllm=args.use_vllm,
                    vllm_client=vllm_client
                )
                if args.image_format != 'images' and args.image_format != 'images_figstep' and args.image_format != 'images_qr':
                    answer_index = response.find("Content")
                    if answer_index == -1:
                        score_temp = 0
                    else:
                        score_temp = get_jailbreak_score(item["original_prompt"], response, api_key)        
                else:
                    score_temp = get_jailbreak_score(item["original_prompt"], response, api_key)
    
                score_list.append(score_temp)
                res_list.append(response)
                # Update progress bar with current score
                pbar.set_postfix({"item_id": item["id"], "score": score_temp, "attempt": attempt + 1})
                if score_temp == 5:
                    break
            
            score_idx = score_list.index(max(score_list))
            final_score = score_list[score_idx]
            pbar.set_postfix({"item_id": item["id"], "final_score": final_score, "queries": len(score_list)})
            pbar.update(1)  # Update progress after processing each item
            
            result.append({
                    "ori_prompt": item["original_prompt"],
                    "res": res_list[score_idx],
                    "score": score_list[score_idx],
                    "flag": item["original_prompt"] in res_list[score_idx], # check wether decrypt suscessfully
                    "query_num": len(score_list),
                })
            json.dump(result, open(result_save_path, 'w'), indent=4)
        pbar.close()  # Close progress bar for this subclass
    eval(eval_path)
main()
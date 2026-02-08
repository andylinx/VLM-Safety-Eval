#!/usr/bin/env python3
"""
Response generation script for MM-SafetyBench using vLLM-deployed models.
Follows the USB code pattern for vLLM integration.
"""

import os
import json
from tqdm import tqdm
import requests
import argparse
import sys
from typing import Optional, Dict, Any, List
import time
from openai import OpenAI
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import client classes from local models module
from models import VLLMClient
from utils.config_utils import get_output_dir, get_data_path

def handle_response(response: str) -> str:
    """Process and clean up the response."""
    if response is None:
        return ""
    return response.strip()

def process_single_variant(variant_info: Dict[str, Any], client, model_name: str, max_tokens: int, 
                          gen_kwargs: Dict) -> Dict[str, Any]:
    """Process a single image variant (SD, TYPO, or SD_TYPO)."""
    try:
        image_path = variant_info["image_path"]
        prompt = variant_info["prompt"]
        variant_key = variant_info["variant_key"]
        
        result = client.generate_response(
            prompt=prompt,
            image_paths=image_path,
            model=model_name,
            max_tokens=max_tokens,
            **gen_kwargs
        )
            
        # print(result)
        # print("==="*8)
        
        return {
            "variant_key": variant_key,
            "response": handle_response(result["response"]),
            "usage": result.get("usage", {}),
            "error": None
        }
    except Exception as e:
        return {
            "variant_key": variant_info["variant_key"],
            "response": f"ERROR: {str(e)}",
            "usage": {},
            "error": str(e)
        }

def generate_response(scenario: str, 
                     client,  # Can be VLLMClient or HuggingFaceClient
                     model_name: str,
                     output_dir: str,
                     data_base_path: str = None,
                     max_tokens: int = 10240,
                     max_workers: int = 4,
                     sample_size: Optional[int] = None,
                     **gen_kwargs):
    """Generate responses for a specific scenario with parallel processing."""
    
    # Use environment variables if not provided
    if data_base_path is None:
        data_base_path = get_data_path("MM-SafetyBench")
    """Generate responses for a specific scenario with parallel processing."""
    
    print(f"Processing scenario: {scenario}")
    
    image_folder = os.path.join(data_base_path, scenario)
    questions_file = os.path.join(data_base_path, "processed_questions", f"{scenario}.json")
    
    if not os.path.exists(questions_file):
        print(f"Questions file not found: {questions_file}")
        return
    
    questions = json.load(open(questions_file, "r"))
    
    # Sample questions if sample_size is specified
    if sample_size is not None and sample_size < len(questions):
        import random
        question_keys = list(questions.keys())
        random.seed(42)  # For reproducibility
        sampled_keys = random.sample(question_keys, sample_size)
        questions = {k: questions[k] for k in sampled_keys}
        print(f"Sampled {sample_size} questions out of {len(question_keys)} total questions")
    
    all_responses = {}
    
    # Token usage tracking
    total_output_tokens = 0
    request_count = 0
    error_count = 0
    
    # Prepare all tasks to be processed in parallel
    tasks = []
    task_metadata = []  # Store metadata for each task
    
    for i in questions.keys():
        item = questions[i].copy()
        
        # Prepare SD variant
        sd_image_path = os.path.join(image_folder, f"SD/{i}.jpg")
        if os.path.exists(sd_image_path) and "Rephrased Question(SD)" in item:
            tasks.append({
                "image_path": sd_image_path,
                "prompt": item["Rephrased Question(SD)"],
                "variant_key": f"{model_name}_SD",
                "question_id": i
            })
            task_metadata.append(("SD", i))
        
        # Prepare TYPO variant
        typo_image_path = os.path.join(image_folder, f"TYPO/{i}.jpg")
        if os.path.exists(typo_image_path) and "Rephrased Question" in item:
            tasks.append({
                "image_path": typo_image_path,
                "prompt": item["Rephrased Question"],
                "variant_key": f"{model_name}_TYPO",
                "question_id": i
            })
            task_metadata.append(("TYPO", i))
        
        # Prepare SD + TYPO variant
        sd_typo_image_path = os.path.join(image_folder, f"SD_TYPO/{i}.jpg")
        if os.path.exists(sd_typo_image_path) and "Rephrased Question" in item:
            tasks.append({
                "image_path": sd_typo_image_path,
                "prompt": item["Rephrased Question"],
                "variant_key": f"{model_name}_SD_TYPO",
                "question_id": i
            })
            task_metadata.append(("SD_TYPO", i))
        
        # Initialize response dict for this question
        all_responses[i] = item.copy()
        all_responses[i]["ans"] = {}
    
    print(f"Total tasks to process: {len(tasks)}")
    print(f"Processing with {max_workers} parallel workers...")
    
    # Process tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                process_single_variant,
                task,
                client,
                model_name,
                max_tokens,
                gen_kwargs
            ): (task, idx) 
            for idx, task in enumerate(tasks)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc=f"Processing {scenario}") as pbar:
            for future in as_completed(future_to_task):
                task, idx = future_to_task[future]
                try:
                    result = future.result()
                    
                    # Store the response in the appropriate question
                    question_id = task["question_id"]
                    variant_key = result["variant_key"]
                    all_responses[question_id]["ans"][variant_key] = result["response"]
                    
                    # Track token usage
                    usage = result.get("usage", {})
                    if usage:
                        total_output_tokens += usage.get("completion_tokens", 0)
                        request_count += 1
                    
                    # Track errors
                    if result["error"]:
                        error_count += 1
                        
                except Exception as e:
                    print(f"\nError processing task {idx}: {e}")
                    error_count += 1
                
                pbar.update(1)
    
    # Save responses
    output_path = os.path.join(output_dir, f"{scenario}_{model_name}_responses.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_responses, f, indent=4)
    
    print(f"Responses saved to {output_path}")
    
    # Print token usage statistics
    print(f"Token Usage Statistics for {scenario}:")
    print(f"  Total requests: {request_count}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    print(f"  Errors encountered: {error_count}")
    if request_count > 0:
        print(f"  Average output tokens per request: {total_output_tokens / request_count:.1f}")
    
    return output_path, {
        "total_requests": request_count,
        "total_output_tokens": total_output_tokens,
        "error_count": error_count
    }

def check_existing_output(output_path: str) -> bool:
    """Check if output file exists and contains valid data."""
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check if the file has the expected structure
        if not isinstance(data, dict):
            return False
        
        # Check if there are responses in the file
        if not data:
            return False
            
        # Check if at least one item has answers
        for item_key, item_data in data.items():
            if "ans" in item_data and item_data["ans"]:
                return True
        
        return False
    except (json.JSONDecodeError, KeyError, TypeError):
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate responses for MM-SafetyBench using vLLM")
    parser.add_argument("--vllm-url", default="http://localhost:8122",
                       help="vLLM server URL")
    parser.add_argument("--model_name", default="model",
                       help="Prefix for output JSON filenames")
    parser.add_argument("--data-base-path", default=None,
                       help="Base path for MM-SafetyBench data (default: from .env DATA_BASE_ROOT_PATH/MM-SafetyBench)")
    parser.add_argument("--max-tokens", type=int, default=10240,
                       help="Maximum tokens to generate")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing of scenarios even if output files already exist")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Number of parallel workers for processing requests (default: 2)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of questions to sample from each scenario for testing (default: 50, process 50 questions)")
    
    # Add generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Top-k sampling (vLLM extension)")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.0, help="Repetition penalty (vLLM extension)")
    
    args = parser.parse_args()
    
    # Determine model name with prefix
    model_name = args.model_name
    
    # Set output directory based on model name
    output_dir = os.path.join(get_output_dir(), "MM_Safety", f"MLLM_Result_{model_name}")
    
    # Create vLLM client
    print(f"Using vLLM client for model: {model_name}")
    client = VLLMClient(args.vllm_url)
    
    # Test vLLM connection
    try:
        models = client.get_models()
        print(f"Connected to vLLM server. Available models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"Error connecting to vLLM server: {e}")
        sys.exit(1)
    
    # Show configuration
    print(f"Configuration:")
    print(f"  Model name: {model_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Force reprocess: {args.force_reprocess}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Parallel workers: {args.max_workers}")
    print(f"  Sample size per scenario: {args.sample_size if args.sample_size else 'All'}")
    print()
    
    # Prepare generation kwargs
    gen_kwargs = {k: v for k, v in dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    ).items() if v is not None}
    
    # Define scenarios to process
    scenarios = [
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision"
    ]
    
    # Process each scenario
    output_files = []
    total_usage = {
        "total_requests": 0,
        "total_output_tokens": 0,
        "error_count": 0
    }
    skipped_scenarios = 0
    
    try:
        for scenario in scenarios:
            print(f"\n{'='*50}")
            print(f"Processing: {scenario}")
            print(f"{'='*50}")
            
            # Check if output file already exists
            expected_output_path = os.path.join(output_dir, f"{scenario}_{model_name}_responses.json")
            if not args.force_reprocess and check_existing_output(expected_output_path):
                print(f"⏭️  Skipping {scenario} - output file already exists: {expected_output_path}")
                skipped_scenarios += 1
                print("-" * 50)
                continue
            
            result = generate_response(
                scenario=scenario,
                client=client,
                model_name=model_name,
                output_dir=output_dir,
                data_base_path=args.data_base_path,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers,
                sample_size=args.sample_size,
                **gen_kwargs
            )
            
            if result:
                output_file, usage_stats = result
                output_files.append(output_file)
                
                # Accumulate total usage
                total_usage["total_requests"] += usage_stats["total_requests"]
                total_usage["total_output_tokens"] += usage_stats["total_output_tokens"]
                total_usage["error_count"] += usage_stats.get("error_count", 0)
            
            print(f"Completed: {scenario}")
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Partial results may be saved.")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    
    print(f"\n🎉 Processing complete!")
    print(f"Generated {len(output_files)} output files:")
    for file_path in output_files:
        print(f"  - {file_path}")
    
    if skipped_scenarios > 0:
        print(f"Skipped {skipped_scenarios} scenarios (already processed)")
    
    # Print overall token usage statistics
    print(f"\n📊 Overall Token Usage Statistics:")
    print(f"  Total requests across all scenarios: {total_usage['total_requests']:,}")
    print(f"  Total output tokens: {total_usage['total_output_tokens']:,}")
    print(f"  Total errors: {total_usage['error_count']:,}")
    if total_usage['total_requests'] > 0:
        print(f"  Average output tokens per request: {total_usage['total_output_tokens'] / total_usage['total_requests']:.1f}")
    
    # Save token usage summary
    usage_summary_path = os.path.join(output_dir, f"{model_name}_token_usage_summary.json")
    with open(usage_summary_path, "w") as f:
        json.dump(total_usage, f, indent=4)
    print(f"  Token usage summary saved to: {usage_summary_path}")

if __name__ == "__main__":
    main()
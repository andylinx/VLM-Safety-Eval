#!/usr/bin/env python3
"""
Direct HuggingFace evaluation script for models that vLLM doesn't support.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.huggingface_client import HuggingFaceClient
from utils.config_utils import get_output_dir, get_data_path

def main():
    parser = argparse.ArgumentParser(description="Run evaluation with HuggingFace directly")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--model-name", required=True, help="Name for output files")
    parser.add_argument("--dataset", required=True, choices=["MM", "MIS"], help="Dataset to evaluate")
    parser.add_argument("--max-tokens", type=int, default=10240, help="Max tokens to generate")
    parser.add_argument("--max-workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    client = HuggingFaceClient(args.model_path)
    print("Model loaded successfully!")
    
    if args.dataset == "MM":
        run_mm_safetybench(client, args)
    elif args.dataset == "MIS":
        run_mis(client, args)

def run_mm_safetybench(client, args):
    """Run MM-SafetyBench evaluation."""
    from evaluation.get_response_MM import generate_response, check_existing_output
    
    output_dir = os.path.join(get_output_dir(), "MM_Safety", f"MLLM_Result_{args.model_name}")
    data_base_path = get_data_path("MM-SafetyBench")
    
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
    
    for scenario in scenarios:
        expected_output_path = os.path.join(output_dir, f"{scenario}_{args.model_name}_responses.json")
        if check_existing_output(expected_output_path):
            print(f"Skipping {scenario} - already exists")
            continue
        
        print(f"\nProcessing: {scenario}")
        result = generate_response(
            scenario=scenario,
            client=client,
            model_name=args.model_name,
            output_dir=output_dir,
            data_base_path=data_base_path,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers
        )
        if result:
            print(f"Completed: {scenario}")

def run_mis(client, args):
    """Run MIS evaluation."""
    from evaluation.get_response_MIS import generate_response
    
    output_dir = os.path.join(get_output_dir(), "MIS")
    
    result = generate_response(
        client=client,
        model_name=args.model_name,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
        num_threads=args.max_workers
    )
    if result:
        print(f"MIS evaluation completed for {args.model_name}")

if __name__ == "__main__":
    main()

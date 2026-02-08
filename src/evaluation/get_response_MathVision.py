#!/usr/bin/env python3
"""
Response generation script for MathVision dataset using vLLM-deployed models.
Supports multimodal math problem solving with images.
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
import base64
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# Import client classes from local models module
from models import VLLMClient
from utils.config_utils import get_output_dir, get_data_base_root_path

def save_image_from_dataset(image_obj, save_path: str) -> bool:
    """Save PIL Image object to file."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_obj.save(save_path)
        return True
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")
        return False
    
def load_mathvision_dataset(split: str = "testmini", 
                           image_save_dir: str = None) -> pd.DataFrame:
    """Load MathVision dataset and convert to DataFrame with saved images."""
    
    # Use environment variables if not provided
    if image_save_dir is None:
        image_save_dir = os.path.join(get_data_base_root_path(), "mathvision_images")
    """Load MathVision dataset and convert to DataFrame with saved images."""
    
    print(f"Loading MathVision dataset split: {split}")
    dataset = load_dataset("MathLLMs/MathVision")
    
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
    
    data = dataset[split]
    print(f"Loaded {len(data)} examples from {split} split")
    
    # Convert to list of dictionaries
    rows = []
    for i, example in enumerate(tqdm(data, desc="Processing examples")):
        # Save image to file
        image_filename = f"{split}_{example['id']}.png"
        image_path = os.path.join(image_save_dir, image_filename)
        
        image_saved = False
        if example.get('decoded_image') is not None:
            image_saved = save_image_from_dataset(example['decoded_image'], image_path)
        
        # Format options as text in (A)-(E) style
        options_text = ""
        if example.get('options'):
            try:
                # Keep order, format exactly like the reference example
                for j, option in enumerate(example['options']):
                    options_text += f"({chr(65+j)}) {option}\n"
            except Exception:
                pass

        # Create the prompt using the specified template
        question_text = f"{example['question']}\n{options_text}".strip()
        prompt = f"""'Please solve the problem and put your answer in one "\\boxed{{}}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{{}}". {question_text}"""
        row = {
            'id': example['id'],
            'question': example['question'],
            'options': options_text.strip() if options_text else "",
            'prompt': prompt,
            'image_path': image_path if image_saved else "",
            'answer': example.get('answer', ''),
            'level': example.get('level', ''),
            'subject': example.get('subject', ''),
            'split': split
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows")
    return df

def process_mathvision_responses(df: pd.DataFrame,
                                output_json_path: str,
                                vllm_client: VLLMClient,
                                model: Optional[str] = None,
                                max_tokens: int = 20480,
                                save_interval: int = 10,
                                retry_errors: bool = True,
                                **gen_kwargs):
    """Process MathVision DataFrame and generate responses."""
    
    # Load existing output JSON if it exists
    output_exists = os.path.exists(output_json_path)
    if output_exists:
        print(f"📂 Found existing output JSON: {output_json_path}")
        try:
            existing_df = pd.read_json(output_json_path)
            print(f"Loaded existing output with {len(existing_df)} rows")
            
            # Merge with existing responses
            if len(existing_df) == len(df):
                print("✓ Using existing responses where available")
                # Merge response columns from existing data
                response_cols = ['model_response', 'response_tokens', 'prompt_tokens', 'total_tokens', 'model_used']
                for col in response_cols:
                    if col in existing_df.columns:
                        df[col] = existing_df[col]
        except Exception as e:
            print(f"⚠️  Error reading existing output JSON: {e}")
            output_exists = False
    
    # Add response columns if they don't exist
    response_cols = ['model_response', 'response_tokens', 'prompt_tokens', 'total_tokens', 'model_used']
    for col in response_cols:
        if col not in df.columns:
            df[col] = None
    
    # Find rows that need processing
    def has_valid_response(row):
        response = row.get('model_response')
        if pd.isna(response) or response is None:
            return False
        response_str = str(response).strip()
        return response_str != "" and not response_str.startswith("ERROR:")
    
    pending_rows = []
    existing_responses = 0
    error_responses = 0
    
    for idx in df.index:
        if has_valid_response(df.loc[idx]):
            existing_responses += 1
        elif not pd.isna(df.loc[idx, 'model_response']) and str(df.loc[idx, 'model_response']).startswith("ERROR:"):
            error_responses += 1
            if retry_errors:
                pending_rows.append(idx)
        else:
            pending_rows.append(idx)
    
    print(f"📊 Status summary:")
    print(f"  ✅ Existing valid responses: {existing_responses}")
    print(f"  ❌ Error responses: {error_responses} ({'will retry' if retry_errors else 'will skip'})")
    print(f"  ⏳ Rows to process: {len(pending_rows)}")
    
    if not pending_rows:
        print("🎉 All rows already have valid responses!")
        return df
    
    # Process rows
    processed_count = 0
    try:
        for i, idx in enumerate(tqdm(pending_rows, desc="Generating responses")):
            prompt = str(df.loc[idx, 'prompt'])
            image_path = df.loc[idx, 'image_path']
            
            # Check if image exists
            if image_path and not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                image_path = None
            
            try:
                result = vllm_client.generate_response(
                    prompt=prompt,
                    image_paths=[image_path] if image_path else None,
                    model=model,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )
                
                # Store results
                df.loc[idx, 'model_response'] = result['response']
                df.loc[idx, 'model_used'] = result['model']
                
                # Store token usage if available
                usage = result.get('usage', {})
                df.loc[idx, 'response_tokens'] = usage.get('completion_tokens')
                df.loc[idx, 'prompt_tokens'] = usage.get('prompt_tokens')
                df.loc[idx, 'total_tokens'] = usage.get('total_tokens')
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                df.loc[idx, 'model_response'] = f"ERROR: {str(e)}"
            
            # Save periodically
            if (i + 1) % save_interval == 0:
                df.to_json(output_json_path, orient='records', indent=2)
                total_valid = existing_responses + processed_count
                print(f"💾 Progress saved: {processed_count} new + {existing_responses} existing = {total_valid}/{len(df)} total responses")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Saving current progress...")
    
    # Final save
    df.to_json(output_json_path, orient='records', indent=2)
    final_valid = existing_responses + processed_count
    print(f"✅ Results saved to: {output_json_path}")
    print(f"📈 Final status: {final_valid}/{len(df)} rows have valid responses ({processed_count} newly generated)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate responses for MathVision dataset using vLLM")
    parser.add_argument("--split", default="testmini", choices=["test", "testmini"],
                       help="Dataset split to use")
    parser.add_argument("--image-dir", default=None,
                       help="Directory to save images (default: from .env DATA_BASE_ROOT_PATH/mathvision_images)")
    parser.add_argument("--vllm-url", default="http://localhost:8122",
                       help="vLLM server URL")
    parser.add_argument("--model_name", default=None,
                       help="Model name (optional, uses first available)")
    parser.add_argument("--max-tokens", type=int, default=20480,
                       help="Maximum tokens to generate")
    parser.add_argument("--save-interval", type=int, default=50,
                       help="Save progress every N rows")
    parser.add_argument("--retry-errors", action="store_true",
                       help="Retry rows that previously had errors")

    # Add generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Top-k sampling (vLLM extension)")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.0, help="Repetition penalty (vLLM extension)")
    
    args = parser.parse_args()
    
    # Create vLLM client
    client = VLLMClient(args.vllm_url)
    
    # Test connection
    try:
        models = client.get_models()
        print(f"Connected to vLLM server. Available models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"Error connecting to vLLM server: {e}")
        sys.exit(1)
    
    # Load dataset
    try:
        df = load_mathvision_dataset(split=args.split, image_save_dir=args.image_dir)
        print(f"Loaded {len(df)} examples from {args.split} split")
        
        # Show sample
        print("\nSample data:")
        print(f"ID: {df.iloc[0]['id']}")
        print(f"Question: {df.iloc[0]['question'][:100]}...")
        print(f"Answer: {df.iloc[0]['answer']}")
        print(f"Subject: {df.iloc[0]['subject']}")
        print(f"Level: {df.iloc[0]['level']}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Prepare generation kwargs
    gen_kwargs = {k: v for k, v in dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    ).items() if v is not None}

    output_json = f"./result/MathVision/math_{args.model_name}.json"
    if not os.path.exists(os.path.dirname(output_json)):
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
    # Process responses
    try:
        result_df = process_mathvision_responses(
            df=df,
            output_json_path=output_json,
            vllm_client=client,
            model=args.model_name,
            max_tokens=args.max_tokens,
            save_interval=args.save_interval,
            retry_errors=args.retry_errors,
            **gen_kwargs
        )
        
        print(f"\nCompleted! Processed {len(result_df)} rows")
        print(f"Results saved to: {output_json}")
        
    except Exception as e:
        print(f"Error processing responses: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()


import os
import json
from tqdm import tqdm
import argparse
import sys
from typing import Optional, Dict, Any, List
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import client classes from local models module
from models import VLLMClient
from config_utils import get_output_dir, get_data_path

def handle_response(response: str) -> str:
    """Process and clean up the response."""
    if response is None:
        return ""
    return response.strip()

def generate_response(client,
                     model_name: str,
                     output_dir: str = None,
                     data_base_path: str = None,
                     max_tokens: int = 10240,
                     save_interval: int = 100,
                     num_threads: int = 2):
    
    # Use environment variables if not provided
    if output_dir is None:
        output_dir = get_output_dir()
    if data_base_path is None:
        data_base_path = get_data_path("MMVP")
    
    # Load MMVP Questions from CSV
    csv_file = os.path.join(data_base_path, "Questions.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Questions.csv not found at {csv_file}")
    
    print(f"Loading MMVP questions from {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} questions")
    
    # Prepare output paths
    output_path = os.path.join(output_dir, f"{model_name}_MMVP_result.json")
    temp_output_path = os.path.join(output_dir, f"{model_name}_MMVP_result_temp.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if there's existing progress to resume from
    start_idx = 0
    all_responses = []
    if os.path.exists(temp_output_path):
        try:
            with open(temp_output_path, "r") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict) and "responses" in existing_data:
                    all_responses = existing_data["responses"]
                    start_idx = existing_data.get("last_processed_index", 0) + 1
                    print(f"Resuming from index {start_idx} (found {len(all_responses)} existing responses)")
                elif isinstance(existing_data, list):
                    all_responses = existing_data
                    start_idx = len(all_responses)
                    print(f"Resuming from index {start_idx} (found {len(all_responses)} existing responses)")
        except Exception as e:
            print(f"Could not load existing progress: {e}. Starting from beginning.")
            start_idx = 0
            all_responses = []
    
    # Token usage tracking
    total_output_tokens = 0
    request_count = 0
    
    # Thread-safe locks for shared resources
    save_lock = threading.Lock()
    stats_lock = threading.Lock()
    
    def save_progress(responses, last_idx, is_final=False):
        """Save current progress to file."""
        with save_lock:
            save_data = {
                "responses": responses,
                "last_processed_index": last_idx,
                "total_processed": len(responses),
                "timestamp": time.time(),
                "is_final": is_final
            }
            
            save_path = output_path if is_final else temp_output_path
            with open(save_path, "w") as f:
                json.dump(save_data if not is_final else responses, f, indent=4)
            
            if is_final:
                print(f"Final responses saved to {output_path}")
                # Clean up temp file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            else:
                print(f"Progress saved: {len(responses)} items processed")
    
    def process_single_item(item_data):
        """Process a single item. Returns (index, processed_item, usage_info)"""
        idx, row = item_data
        current_idx = start_idx + idx
        
        # Construct the prompt: Question + Options
        question = row.get('Question', '')
        options = row.get('Options', '')
        cur_prompt = f"{question} {options}".strip()
        
        # Construct image path: photo_id is (index + 1)
        photo_id = current_idx + 1
        image_filename = f"{photo_id}.jpg"
        image_path = os.path.join(data_base_path, "MMVP Images", image_filename)
        
        image_paths = []
        if os.path.exists(image_path):
            image_paths = [image_path]
        else:
            print(f"Warning: Image not found at {image_path}")

        try:
            result = client.generate_response(
                prompt=cur_prompt,
                image_paths=image_paths if image_paths else None,
                model="model",
                max_tokens=max_tokens
            )

            # Build response item
            item = {
                "question_id": photo_id,
                "prompt": cur_prompt,
                "answer": row.get("Correct Answer", ""),
                "response": handle_response(result["response"]),
                "model_id": model_name,
                "image_path": image_path
            }
            
            # Extract usage info
            usage = result.get("usage", {})
            usage_info = {
                "completion_tokens": usage.get("completion_tokens", 0),
                "request_count": 1
            }
                    
            return current_idx, item, usage_info
            
        except Exception as e:
            print(f"Error processing item {current_idx}: {e}")
            raise e
    
    # Prepare data for processing
    items_to_process = list(df.iloc[start_idx:].iterrows())
    processed_items = {}  # Dictionary to maintain order: {index: item}
    
    # Process items with threading
    if num_threads > 1:
        print(f"Processing with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_item, (i, row)): i 
                           for i, (_, row) in enumerate(items_to_process)}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(items_to_process), desc="Processing", initial=start_idx) as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        current_idx, processed_item, usage_info = future.result()
                        
                        # Store processed item
                        processed_items[current_idx] = processed_item
                        
                        # Update statistics thread-safely
                        with stats_lock:
                            total_output_tokens += usage_info["completion_tokens"]
                            request_count += usage_info["request_count"]
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Check if we should save progress
                        if len(processed_items) % save_interval == 0:
                            # Convert to ordered list for saving
                            ordered_responses = []
                            for i in range(len(all_responses)):
                                ordered_responses.append(all_responses[i])
                            
                            # Add newly processed items in order
                            for idx in sorted(processed_items.keys()):
                                if idx >= len(all_responses):
                                    ordered_responses.append(processed_items[idx])
                            
                            save_progress(ordered_responses, max(processed_items.keys()) if processed_items else start_idx - 1)
                            
                    except Exception as e:
                        print(f"Error in thread processing: {e}")
                        # Save progress on error
                        ordered_responses = []
                        for i in range(len(all_responses)):
                            ordered_responses.append(all_responses[i])
                        for idx in sorted(processed_items.keys()):
                            if idx >= len(all_responses):
                                ordered_responses.append(processed_items[idx])
                        save_progress(ordered_responses, max(processed_items.keys()) if processed_items else start_idx - 1)
                        raise e
    else:
        # Single-threaded processing
        print("Processing with single thread...")
        with tqdm(total=len(items_to_process), desc="Processing", initial=start_idx) as pbar:
            for i, (_, row) in enumerate(items_to_process):
                current_idx, processed_item, usage_info = process_single_item((i, row))
                processed_items[current_idx] = processed_item
                
                # Update statistics
                total_output_tokens += usage_info["completion_tokens"]
                request_count += usage_info["request_count"]
                
                # Update progress bar
                pbar.update(1)
                
                # Progressive save every save_interval items
                if (current_idx + 1) % save_interval == 0:
                    ordered_responses = []
                    for i in range(len(all_responses)):
                        ordered_responses.append(all_responses[i])
                    for idx in sorted(processed_items.keys()):
                        if idx >= len(all_responses):
                            ordered_responses.append(processed_items[idx])
                    save_progress(ordered_responses, current_idx)
    
    # Combine all responses in order
    final_responses = []
    for i in range(len(all_responses)):
        final_responses.append(all_responses[i])
    
    # Add newly processed items in order
    for idx in sorted(processed_items.keys()):
        if idx >= len(all_responses):
            final_responses.append(processed_items[idx])
    
    # Final save
    save_progress(final_responses, len(df) - 1, is_final=True)
    
    # Print token usage statistics
    print(f"  Total requests: {request_count}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    if request_count > 0:
        print(f"  Average output tokens per request: {total_output_tokens / request_count:.1f}")
    
    return output_path, {
        "total_requests": request_count,
        "total_output_tokens": total_output_tokens
    }

def main():
    parser = argparse.ArgumentParser(description="Generate responses for MMVP Benchmark using vLLM")
    parser.add_argument("--vllm-url", default="http://localhost:8122",
                       help="vLLM server URL")
    parser.add_argument("--model_name", default="model",
                       help="Prefix for output JSON filenames")
    parser.add_argument("--data-base-path", default=None,
                       help="Base path for MMVP data (default: from .env DATA_BASE_ROOT_PATH/MMVP)")
    parser.add_argument("--max-tokens", type=int, default=10240,
                       help="Maximum tokens to generate")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing even if output files already exist")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save progress every N items (default: 100)")
    parser.add_argument("--num-threads", type=int, default=2,
                       help="Number of threads for concurrent processing (default: 2)")
    
    args = parser.parse_args()
    
    # Determine model name with prefix
    model_name = args.model_name
    
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
    print(f"  Force reprocess: {args.force_reprocess}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Number of threads: {args.num_threads}")
    print()
    
    result = generate_response(
        client=client,
        model_name=model_name,
        output_dir=os.path.join(get_output_dir(), "MMVP"),
        data_base_path=args.data_base_path,
        max_tokens=args.max_tokens,
        save_interval=args.save_interval,
        num_threads=args.num_threads
    )
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {result[0]}")

if __name__ == "__main__":
    main()
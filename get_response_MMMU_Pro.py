import os
import json
import sys
from tqdm import tqdm
import argparse
from typing import Optional, Dict, Any, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import client classes from local models module
from models import VLLMClient, HuggingFaceClient
from config_utils import get_output_dir, get_data_path, get_data_base_root_path

def handle_response(response: str) -> str:
    """Process and clean up the response."""
    if response is None:
        return ""
    return response.strip()

def generate_response(client,
                     model_name: str,
                     output_dir: str = None,
                     data_path: str = None,
                     image_base_path: str = None,
                     max_tokens: int = 10240,
                     skip_thinking: bool = False,
                     thinking_mode: str = "auto",
                     save_interval: int = 100,
                     num_threads: int = 2):
    
    # Use environment variables if not provided
    if output_dir is None:
        output_dir = get_output_dir()
    if data_path is None:
        data_path = os.path.join(get_data_base_root_path(), "MMLU_PRO/standard (4 options)/dataset.json")
    if image_base_path is None:
        image_base_path = os.path.join(get_data_base_root_path(), "MMLU_PRO/standard (4 options)")
    
    # Load MMLU_PRO dataset from JSON
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading MMLU_PRO dataset from {data_path}")
    with open(data_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions")
    
    # Prepare output paths
    output_path = os.path.join(output_dir, f"{model_name}_MMLU_PRO_result.json")
    temp_output_path = os.path.join(output_dir, f"{model_name}_MMLU_PRO_result_temp.json")
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
        idx, item = item_data
        current_idx = start_idx + idx
        
        # Extract data from item
        question_id = item.get('id', f'question_{current_idx}')
        question = item.get('question', '')
        options_str = item.get('options', '')
        answer = item.get('answer', '')
        subject = item.get('subject', '')
        topic_difficulty = item.get('topic_difficulty', '')
        
        # Parse options from string representation of list
        try:
            import ast
            options_list = ast.literal_eval(options_str)
        except:
            options_list = []
        
        # Format options with letter labels (A, B, C, D, etc.)
        formatted_options = []
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for i, option in enumerate(options_list):
            if i < len(option_letters):
                formatted_options.append(f"{option_letters[i]}. {option}")
        
        options_text = "\n".join(formatted_options)
        
        # Construct the prompt: Question + Options + Instruction
        cur_prompt = f"{question}\n{options_text}\nFinally answer with the option letter from the given choices.".strip()
        
        # Get image paths from the 'images' list
        images = item.get('images', [])
        image_paths = []
        
        for img_filename in images:
            img_path = os.path.join(image_base_path, img_filename)
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print(f"Warning: Image not found at {img_path}")

        try:
            if isinstance(client, HuggingFaceClient):
                result = client.generate_response(
                    prompt=cur_prompt,
                    image_paths=image_paths if image_paths else None,
                    max_tokens=max_tokens,
                    thinking_mode=thinking_mode,
                    skip_thinking=skip_thinking
                )
            else:  # VLLMClient
                result = client.generate_response(
                    prompt=cur_prompt,
                    image_paths=image_paths if image_paths else None,
                    model="model",
                    max_tokens=max_tokens,
                    skip_thinking=skip_thinking
                )
                
            # Build response item
            response_item = {
                "question_id": question_id,
                "prompt": cur_prompt,
                "answer": answer,
                "response": handle_response(result["response"]),
                "model_id": model_name,
                "subject": subject,
                "topic_difficulty": topic_difficulty,
                "image_paths": image_paths,
                "num_images": len(image_paths)
            }
            
            # Extract usage info
            usage = result.get("usage", {})
            usage_info = {
                "completion_tokens": usage.get("completion_tokens", 0),
                "request_count": 1
            }
                    
            return current_idx, response_item, usage_info
            
        except Exception as e:
            print(f"Error processing item {current_idx} (ID: {question_id}): {e}")
            raise e
    
    # Prepare data for processing
    items_to_process = [(i, dataset[start_idx + i]) for i in range(len(dataset) - start_idx)]
    processed_items = {}  # Dictionary to maintain order: {index: item}
    
    # Process items with threading
    if num_threads > 1:
        print(f"Processing with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_item, item): item[0] 
                           for item in items_to_process}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(dataset), desc="Processing", initial=start_idx) as pbar:
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
        with tqdm(total=len(dataset), desc="Processing", initial=start_idx) as pbar:
            for item in items_to_process:
                current_idx, processed_item, usage_info = process_single_item(item)
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
    save_progress(final_responses, len(dataset) - 1, is_final=True)
    
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
    parser = argparse.ArgumentParser(description="Generate responses for MMLU_PRO Benchmark using vLLM")
    parser.add_argument("--vllm-url", default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--model_name", default="model",
                       help="Prefix for output JSON filenames")
    parser.add_argument("--model-path", default="/data/zhengyue_zhao/workspace/nanxi/Models/R-4B",
                       help="Path to the R-4B model (used when model_name is R-4B)")
    parser.add_argument("--data-path", default=None,
                       help="Path to MMLU_PRO dataset JSON file (default: from .env DATA_BASE_ROOT_PATH/MMLU_PRO/standard (4 options)/dataset.json)")
    parser.add_argument("--image-base-path", default=None,
                       help="Base path for MMLU_PRO images (default: from .env DATA_BASE_ROOT_PATH/MMLU_PRO/standard (4 options)/images)")
    parser.add_argument("--max-tokens", type=int, default=10240,
                       help="Maximum tokens to generate")
    parser.add_argument("--skip-thinking", action="store_true",
                       help="Enable skip thinking mode - prefill <think> </think> tokens in assistant response")
    parser.add_argument("--thinking-mode", choices=["auto", "long", "short"], default="auto",
                       help="Thinking mode for R-4B model: auto (default), long, or short")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing even if output files already exist")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save progress every N items (default: 100)")
    parser.add_argument("--num-threads", type=int, default=2,
                       help="Number of threads for concurrent processing (default: 2)")
    
    args = parser.parse_args()
    
    # Determine model name with prefix
    model_name = args.model_name
    
    # Create appropriate client based on model name
    if model_name == "R-4B":
        # Use HuggingFace client for R-4B model
        print(f"Using HuggingFace client for R-4B model at {args.model_path}")
        try:
            client = HuggingFaceClient(args.model_path)
            print("Successfully initialized HuggingFace client for R-4B")
        except Exception as e:
            print(f"Error initializing HuggingFace client: {e}")
            sys.exit(1)
    else:
        # Use vLLM client for other models
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
    print(f"  Data path: {args.data_path}")
    print(f"  Image base path: {args.image_base_path}")
    print(f"  Skip thinking mode: {args.skip_thinking}")
    print(f"  Thinking mode: {args.thinking_mode}")
    print(f"  Force reprocess: {args.force_reprocess}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Number of threads: {args.num_threads}")
    print()
    
    result = generate_response(
        client=client,
        model_name=model_name,
        output_dir=os.path.join(get_output_dir(), "MMLU_PRO"),
        data_path=args.data_path,
        image_base_path=args.image_base_path,
        max_tokens=args.max_tokens,
        skip_thinking=args.skip_thinking,
        thinking_mode=args.thinking_mode,
        save_interval=args.save_interval,
        num_threads=args.num_threads
    )
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {result[0]}")

if __name__ == "__main__":
    main()

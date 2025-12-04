import os
import json
import argparse
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from generate_lib.utils import get_generate_fn, get_client_fn, generate_response_remote_wrapper
from models import VLLMClient, HuggingFaceClient
from config_utils import get_data_path, get_output_dir
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--data_dir', type=str, required=False, default="./data",
                        help="Directory containing the input json files")
    parser.add_argument('--image_dir', type=str, required=False, default=None,
                        help="Directory containing the images (default: from .env DATA_BASE_ROOT_PATH/Charxiv/images)")
    parser.add_argument('--output_dir', type=str, required=False, default=None, \
                        help="Directory to save the output json files (default: from .env OUTPUT_DIR)")
    
    args_temp = parser.parse_known_args()[0]
    # Set defaults from config if not provided
    if args_temp.image_dir is None:
        parser.set_defaults(image_dir=get_data_path("Charxiv/images"))
    if args_temp.output_dir is None:
        parser.set_defaults(output_dir=get_output_dir())
    parser.add_argument('--split', type=str, required=False, choices=['val', 'test'], default='val',
                        help="Split of the data")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default="reasoning", choices=['descriptive', 'reasoning'],
                        help="Mode of the evaluation")
    # VLLM arguments
    parser.add_argument('--vllm-url', type=str, default="http://localhost:8000",
                       help="vLLM server URL (if provided, uses VLLMClient instead of generate_lib)")
    parser.add_argument('--max-tokens', type=int, default=10240,
                       help="Maximum tokens to generate")
    parser.add_argument('--skip-thinking', action='store_true',
                       help="Skip thinking mode for compatible models")
    parser.add_argument('--thinking-mode', choices=['auto', 'long', 'short'], default='auto',
                       help="Thinking mode for compatible models")
    
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}.json')

    if args.mode == 'descriptive':
        from descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, args.image_dir)
    elif args.mode == 'reasoning':
        from reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, args.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)

    # Use VLLMClient if vllm_url is provided
    if args.vllm_url is not None:
        print(f"Using VLLMClient with URL: {args.vllm_url}")
        client = VLLMClient(args.vllm_url)
        
        # Test connection
        try:
            models = client.get_models()
            print(f"Connected to vLLM server. Available models: {[m['id'] for m in models]}")
        except Exception as e:
            print(f"Error connecting to vLLM server: {e}")
            exit(1)
        
        # Generate responses for each query
        for k in tqdm(queries.keys(), desc="Generating responses"):
            query = queries[k]
            try:
                result = client.generate_response(
                    prompt=query['question'],
                    image_paths=query['figure_path'],
                    model=args.model_name,
                    max_tokens=args.max_tokens,
                    skip_thinking=args.skip_thinking
                )
                # print(f"Response for {k}: {result['response']}")
                queries[k]['response'] = result['response']
            except Exception as e:
                print(f"Error generating response for {k}: {e}")
                queries[k]['response'] = f"ERROR: {str(e)}"
    
    # Use HuggingFaceClient for R-4B model
    elif args.model_name == "R-4B":
        print(f"Using HuggingFaceClient for R-4B model at {args.model_path}")
        try:
            client = HuggingFaceClient(args.model_path)
            print("Successfully initialized HuggingFace client for R-4B")
        except Exception as e:
            print(f"Error initializing HuggingFace client: {e}")
            exit(1)
        
        # Generate responses for each query
        for k in tqdm(queries.keys(), desc="Generating responses"):
            query = queries[k]
            try:
                result = client.generate_response(
                    prompt=query['question'],
                    image_paths=query['figure_path'],
                    max_tokens=args.max_tokens,
                    thinking_mode=args.thinking_mode,
                    skip_thinking=args.skip_thinking
                )
                queries[k]['response'] = result['response']
            except Exception as e:
                print(f"Error generating response for {k}: {e}")
                queries[k]['response'] = f"ERROR: {str(e)}"
    
    # Use original generate_lib logic
    else:
        generate_fn = get_generate_fn(args.model_path)
        if args.model_api is not None:
            client, model = get_client_fn(args.model_path)(args.model_path, args.model_api)
            generate_response_remote_wrapper(generate_fn, queries, model, args.model_api, client)
        else:
            generate_fn(queries, args.model_path)

    for k in queries:
        queries[k].pop("figure_path", None)
        queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")

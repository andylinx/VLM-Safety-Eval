#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import socket
import argparse
import requests
from typing import Optional

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def wait_for_server(url: str, timeout: int = 600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/v1/models")
            if response.status_code == 200:
                print(f"\nServer at {url} is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rWaiting for vLLM server to start... ({elapsed}s)")
        sys.stdout.flush()
        time.sleep(5)
    
    print(f"\nTimeout reached while waiting for server at {url}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server and run evaluation")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--dataset", required=True, choices=["MM", "MIS", "Hallucination", "MVP", "MMMU_Pro", "MathVision"], 
                        help="Dataset to evaluate")
    parser.add_argument("--model-name", required=True, help="Name of the model for output")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for evaluation")
    
    args, unknown = parser.parse_known_args()
    
    port = find_free_port()
    vllm_url = f"http://localhost:{port}"
    
    # Map dataset to script
    dataset_scripts = {
        "MM": "src/evaluation/get_response_MM.py",
        "MIS": "src/evaluation/get_response_MIS.py",
        "Hallucination": "src/evaluation/get_response_Hallucination.py",
        "MVP": "src/evaluation/get_response_MVP.py",
        "MMMU_Pro": "src/evaluation/get_response_MMMU_Pro.py",
        "MathVision": "src/evaluation/get_response_MathVision.py"
    }
    
    eval_script = dataset_scripts[args.dataset]
    
    # Get project root (assuming this script is in project_root/scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, "src")
    
    # 1. Start vLLM server
    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code"
    ]
    
    print(f"Launching vLLM server on port {port}...")
    vllm_proc = subprocess.Popen(vllm_cmd)
    
    try:
        # 2. Wait for server to be ready
        if not wait_for_server(vllm_url):
            print("Failed to start vLLM server. Exiting.")
            vllm_proc.terminate()
            sys.exit(1)
        
        # 3. Run evaluation script
        # Set PYTHONPATH to include src
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"
        
        eval_cmd = [
            "python", eval_script,
            "--model_name", args.model_name,
            "--vllm-url", vllm_url,
            "--max-workers", str(args.num_threads)
        ] + unknown
        
        print(f"Running evaluation: {' '.join(eval_cmd)}")
        eval_proc = subprocess.run(eval_cmd, env=env, cwd=project_root)
        
        if eval_proc.returncode != 0:
            print(f"Evaluation script failed with return code {eval_proc.returncode}")
            
    finally:
        # 4. Shutdown vLLM server
        print("Shutting down vLLM server...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            
    print("Done.")

if __name__ == "__main__":
    main()

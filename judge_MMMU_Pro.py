import os
import sys
import json
from tqdm import tqdm


def create_judge_prompt(model_answer, gt_answer):
    """Create a prompt for the judge model"""
    prompt = f"""You are an expert judge for multiple choice questions. Compare the model's answer with the ground truth answer and determine if they match.

Model's Answer: {model_answer}
Ground Truth Answer: {gt_answer}

Are these answers the same? Respond with only "YES" or "NO"."""
    
    return prompt

import requests
import time
from openai import OpenAI
from typing import Optional, Dict, Any, List, Union


class VLLMClient:
    """Client for interfacing with vLLM-deployed models via OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize VLLM client.
        
        Args:
            base_url: Base URL of the vLLM server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        # Initialize OpenAI client for vLLM
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=f"{self.base_url}/v1"
        )
        
    def get_models(self) -> List[Dict]:
        """Get available models from the vLLM server."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()["data"]
        except Exception as e:
            raise RuntimeError(f"Failed to get models: {e}")
    
    def generate_response(self, 
                         prompt: str, 
                         image_paths: Optional[Union[str, List[str]]],
                         model: Optional[str] = None,
                         max_tokens: int = 10240,
                         temperature: float = 0.9,
                         top_p: float = 0.9,
                         top_k: Optional[int] = None,
                         repetition_penalty: Optional[float] = 1.2,
                         skip_thinking: bool = False,
                         assistant_content: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate response from the vLLM model with optional image input(s).
        
        Args:
            prompt: Text prompt for the model
            image_paths: Optional path(s) to image files
            model: Model name to use (if None, uses first available)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (optional)
            repetition_penalty: Repetition penalty (optional)
            skip_thinking: Whether to append /no_think for Mimo models
            assistant_content: Optional content for assistant message (enables continue_final_message)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response, usage stats, and metadata
        """
        
        # If no model specified, use the first available model
        if model is None:
            models = self.get_models()
            if not models:
                raise RuntimeError("No models available")
            model = models[0]["id"]
        # Normalize image_paths to list
        if image_paths is None:
            image_paths = []
        elif isinstance(image_paths, str):
            image_paths = [image_paths]
        # Filter out non-existent images
        valid_image_paths = [path for path in image_paths if os.path.exists(path)]
        
        # Prepare message content
        content = []
        
        # Add images first
        for image_path in valid_image_paths:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"file://{image_path}"}
            })
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        # If no images, use text-only format
        if not valid_image_paths:
            content = prompt
        
        # Handle skip-thinking for Mimo model
        if skip_thinking and "Mimo" in str(model):
            if isinstance(content, list):
                # For multimodal content, modify the text part
                for item in content:
                    if item.get("type") == "text":
                        item["text"] = item["text"] + "/no_think"
            else:
                # For text-only content
                content = content + "/no_think"
        
        # Define system prompt for WeThink model
        system_prompt = None
        if "WeThink" in model:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        # Handle custom assistant_content parameter

        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        # Handle skip-thinking for non-Mimo models (only if assistant_content not provided)
        elif skip_thinking and "Mimo" not in str(model):
            messages.append({"role": "assistant", "content": "<think>"})
        
        request_body = {
            "model": "model",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        
        # Add optional vLLM-specific parameters to extra_body
        if top_k is not None:
            request_body["extra_body"]["top_k"] = top_k
        if repetition_penalty is not None:
            request_body["extra_body"]["repetition_penalty"] = repetition_penalty
        
        # Add continue_final_message if assistant_content is provided or skip_thinking for non-Mimo models
        
        if assistant_content is not None or (skip_thinking and "Mimo" not in str(model)):
            request_body["extra_body"]["continue_final_message"] = True
            request_body["extra_body"]["add_generation_prompt"] = False
        
        try:
            completion = self.client.chat.completions.create(**request_body)
            
            response_content = completion.choices[0].message.content
            return {
                "response": response_content,
                "usage": completion.usage.model_dump() if completion.usage else {},
                "model": model,
                "prompt": prompt,
                "image_paths": valid_image_paths
            }
            
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
        
def judge_with_llm(client, model_answer, gt_answer):
    """Use LLM to judge if the model answer matches ground truth"""
    prompt = create_judge_prompt(model_answer, gt_answer)
    
    try:
        result = client.generate_response(
            prompt=prompt,
            image_paths=None,
            max_tokens=10,
            temperature=0.0,
            top_p=1.0
        )
        judgment = result['response'].strip().upper()

        return "YES" in judgment
    except Exception as e:
        print(f"Error during judgment: {e}")
        return False


def extract_answer_after_think(response):
    """Extract content after </think> token if it exists"""
    if not isinstance(response, str):
        return ""
    
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    
    return response.strip()


def evaluate_file(file_path, vllm_base_url="http://localhost:8001"):
    """Evaluate a single JSON file and report accuracy"""
    # Initialize vLLM client
    print(f"Initializing vLLM client at: {vllm_base_url}")
    client = VLLMClient(base_url=vllm_base_url)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    correct_count = 0
    total_count = len(results)
    
    for idx, result in enumerate(tqdm(results, desc=f'Evaluating {os.path.basename(file_path)}')):
        response = result.get('response', '')
        
        # Extract answer after </think> token
        model_answer = extract_answer_after_think(response)
        
        # Get ground truth answer
        gt_answer = result.get('answer', '')
        
        # Use LLM judge to determine if correct
        is_correct = judge_with_llm(client, model_answer, gt_answer)
        
        # Update result with parsed answer and correctness
        result['model_answer'] = model_answer
        result['correct'] = is_correct
        
        if is_correct:
            correct_count += 1
    
    # Calculate accuracy
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Write updated results back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return accuracy, correct_count, total_count



def process_directory(input_dir, vllm_base_url="http://localhost:8001"):
    """Process all JSON files in the directory"""
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist")
        return
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            try:
                accuracy, correct, total = evaluate_file(file_path, vllm_base_url)
                print(f"{file_name}: {correct}/{total} = {accuracy:.2f}%")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MMMU Pro multiple choice responses')
    parser.add_argument('input', type=str, help='Specific JSON file to process')
    parser.add_argument('--dir', type=str, default='./output', help='Directory to search for JSON files')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8005', 
                        help='Base URL for vLLM server')
    
    args = parser.parse_args()
    
    if args.input:
        if os.path.exists(args.input) and args.input.endswith('.json'):
            try:
                accuracy, correct, total = evaluate_file(args.input, args.vllm_url)
                print(f"{os.path.basename(args.input)}: {correct}/{total} = {accuracy:.2f}%")
            except Exception as e:
                print(f"Error processing {args.input}: {e}")
        else:
            print(f"File {args.input} not found or not a JSON file")
    else:
        process_directory(args.dir, args.vllm_url)
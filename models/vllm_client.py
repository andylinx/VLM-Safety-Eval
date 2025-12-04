#!/usr/bin/env python3
"""
VLLM Client for interfacing with vLLM-deployed models.
Provides a unified interface for generating responses using OpenAI-compatible API.
"""

import os
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
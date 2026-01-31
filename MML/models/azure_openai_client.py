#!/usr/bin/env python3
"""
Azure OpenAI Client for multimodal evaluation tasks.
Provides structured response capabilities for safety evaluation.
"""

import os
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Type
from pydantic import BaseModel
from openai import AzureOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AzureOpenAIClient:
    """Client for interfacing with Azure OpenAI API for multimodal evaluation."""
    
    def __init__(self, 
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 deployment_name: Optional[str] = None,
                 api_version: Optional[str] = None):
        """
        Initialize Azure OpenAI client.
        
        Credentials are loaded from environment variables if not provided:
        - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint base URL
        - AZURE_OPENAI_API_KEY: Azure OpenAI API key
        - AZURE_OPENAI_DEPLOYMENT: Name of the deployed model
        - AZURE_OPENAI_API_VERSION: API version to use
        
        Args:
            api_base: Azure OpenAI endpoint base URL (optional, defaults to env var)
            api_key: Azure OpenAI API key (optional, defaults to env var)
            deployment_name: Name of the deployed model (optional, defaults to env var)
            api_version: API version to use (optional, defaults to env var)
        """
        # Load from environment variables with fallbacks
        self.api_base = api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "latest")
        
        # Validate required credentials
        if not self.api_base:
            raise ValueError("Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable or pass api_base parameter.")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Azure OpenAI client
        # Standard Azure OpenAI pattern: base_url should be {endpoint}/openai
        # The deployment name is passed as the model parameter in API calls
        base_url = self.api_base.rstrip('/')
        if not base_url.endswith('/openai'):
            base_url = f"{base_url}/openai"
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=base_url
        )
        
        # Default generation configuration
        self.default_config = {
            "max_tokens": 2048,
            "top_p": 1.0, 
            "temperature": 0.0
        }
        
        # Image type mapping for base64 detection
        self.image_type_map = {
            "/": "image/jpeg",
            "i": "image/png",
            "R": "image/gif",
            "U": "image/webp",
            "Q": "image/bmp",
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    def create_image_data_uri(self, image_path: str) -> str:
        """Create data URI for image."""
        image_format = Path(image_path).suffix[1:]  # 'jpg', 'png', etc.
        base64_image = self.encode_image(image_path)
        return f"data:image/{image_format};base64,{base64_image}"
    
    def guess_image_type_from_base64(self, base_str: str) -> str:
        """
        Guess image type from base64 string.
        
        Args:
            base_str: Base64 encoded string
            
        Returns:
            MIME type string, defaults to 'image/jpeg'
        """
        default_type = "image/jpeg"
        if not isinstance(base_str, str) or len(base_str) == 0:
            return default_type
        first_char = base_str[0]
        return self.image_type_map.get(first_char, default_type)
    
    def structured_call(self, 
                       messages: Union[str, List[Dict]], 
                       response_class: Type[BaseModel],
                       max_retries: int = 3,
                       **gen_kwargs) -> BaseModel:
        """
        Make a structured API call with retry logic.
        
        Args:
            messages: Either a string prompt or list of message dictionaries
            response_class: Pydantic model class for structured response
            max_retries: Maximum number of retry attempts
            **gen_kwargs: Additional generation parameters
            
        Returns:
            Parsed response object or empty dict on failure
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Merge with default config
        config = {**self.default_config, **gen_kwargs}
        
        for attempt in range(max_retries):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.deployment_name,
                    messages=messages,
                    response_format=response_class,
                    **config,
                )
                
                if completion.choices[0].message.refusal:
                    return {}
                else:
                    return completion.choices[0].message.parsed
                    
            except Exception as e:
                print(f"Azure OpenAI API Error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
                if attempt == max_retries - 1:
                    return {}
                continue
        
        return {}
    
    def batch_structured_evaluation(self, 
                                  data_messages: List[List[Dict]], 
                                  response_class: Type[BaseModel],
                                  **gen_kwargs) -> List[Dict]:
        """
        Perform batch structured evaluation.
        
        Args:
            data_messages: List of message lists for batch processing
            response_class: Pydantic model class for structured response
            **gen_kwargs: Additional generation parameters
            
        Returns:
            List of evaluation results as dictionaries
        """
        outputs_json = []
        
        for messages in tqdm(data_messages, desc="Batch structured evaluation"):
            if isinstance(messages, List) and isinstance(messages[0], dict):
                output_json = self.structured_call(
                    messages, response_class, **gen_kwargs
                )
                outputs_json.append(output_json)
                print(f"####Evaluation Output####\n{output_json}")
            else:
                raise ValueError("Invalid input type")
        
        # Filter out empty results and convert to dictionaries
        outputs_json = [o for o in outputs_json if o != {}]
        # Prefer Pydantic v2 .model_dump(); fall back to .dict() for v1 compatibility
        outputs_json = [
            output.model_dump() if hasattr(output, 'model_dump')
            else output.dict() if hasattr(output, 'dict')
            else output
            for output in outputs_json
        ]
        
        return outputs_json
    
    def create_multimodal_message(self, 
                                 text_prompt: str, 
                                 image_paths: List[str],
                                 system_message: Optional[str] = None) -> List[Dict]:
        """
        Create a multimodal message with text and images.
        
        Args:
            text_prompt: Text content for the message
            image_paths: List of paths to images
            system_message: Optional system message
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Create content list with images and text
        content = []
        
        # Add images
        for image_path in image_paths:
            if os.path.exists(image_path):
                image_url = self.create_image_data_uri(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
        
        # Add text
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def chat_completion(self,
                       messages: Union[str, List[Dict]],
                       max_tokens: int = 2000,
                       temperature: float = 0.7,
                       max_retries: int = 3,
                       **kwargs) -> str:
        """
        Make a regular chat completion call (non-structured).
        
        Args:
            messages: Either a string prompt or list of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Response content as string, or refusal message if present
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        config = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    **config
                )
                
                if response.choices[0].message.refusal:
                    return response.choices[0].message.refusal
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"Azure OpenAI API Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
                continue
        
        return ""
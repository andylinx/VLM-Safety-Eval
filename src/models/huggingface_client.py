#!/usr/bin/env python3
"""
HuggingFace Client for interfacing with locally loaded HuggingFace models.
Provides a unified interface for generating responses using transformers library.
"""

import os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import Optional, Dict, Any, List, Union


class HuggingFaceClient:
    """Client for interfacing with locally loaded HuggingFace models."""
    
    def __init__(self, model_path: str = "/data/zhengyue_zhao/workspace/nanxi/Models/R-4B"):
        """
        Initialize HuggingFace client with a local model.
        
        Args:
            model_path: Path to the local HuggingFace model directory
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load the model and processor from the specified path."""
        try:
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to("cuda")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            print(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_response(self, 
                         prompt: str, 
                         image_paths: Optional[Union[str, List[str]]] = None,
                         model: Optional[str] = None,
                         max_tokens: int = 10240,
                         thinking_mode: str = "auto",
                         skip_thinking: bool = False,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate response using the loaded HuggingFace model with optional image input(s).
        
        Args:
            prompt: Text prompt for the model
            image_paths: Optional path(s) to image files
            model: Model name (used for compatibility, but ignored for local models)
            max_tokens: Maximum number of tokens to generate
            thinking_mode: Thinking mode for compatible models ("auto", "long", "short")
            skip_thinking: Whether to append /no_think for Mimo models
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response, usage stats, and metadata
        """
        
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Handle skip-thinking for Mimo model
            if skip_thinking and "Mimo" in str(model):
                prompt = prompt + "/no_think"
            
            # Normalize image_paths to list
            if image_paths is None:
                image_paths = []
            elif isinstance(image_paths, str):
                image_paths = [image_paths]
            
            # Filter out non-existent images
            valid_image_paths = [path for path in image_paths if os.path.exists(path)]
            
            # Prepare conversation messages
            if valid_image_paths:
                # Create content list with images and text
                content = []
                for image_path in valid_image_paths:
                    content.append({"type": "image", "image": image_path})
                content.append({"type": "text", "text": prompt})
                
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            else:
                # Text-only input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=thinking_mode
            )
            
            # Process inputs
            if valid_image_paths:
                images = [Image.open(path) for path in valid_image_paths]
                # For multiple images, pass them as a list
                inputs = self.processor(
                    images=images if len(images) > 1 else images[0],
                    text=text,
                    return_tensors="pt"
                ).to("cuda")
            else:
                inputs = self.processor(
                    text=text,
                    return_tensors="pt"
                ).to("cuda")
            
            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            
            # Decode output
            output_text = self.processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return {
                "response": output_text,
                "usage": {
                    "completion_tokens": len(output_ids),
                    "prompt_tokens": len(inputs.input_ids[0]),
                    "total_tokens": len(inputs.input_ids[0]) + len(output_ids)
                },
                "model": os.path.basename(self.model_path),
                "prompt": prompt,
                "image_paths": valid_image_paths
            }
            
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
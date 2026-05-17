#!/usr/bin/env python3
"""
HuggingFace Client for interfacing with locally loaded HuggingFace models.
Provides a unified interface for generating responses using transformers library.
"""

import os
import sys
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from typing import Optional, Dict, Any, List, Union


class HuggingFaceClient:
    """Client for interfacing with locally loaded HuggingFace models."""
    
    def __init__(self, model_path: str = "/data/zhengyue_zhao/workspace/nanxi/Models/R-4B"):
        """
        Initialize HuggingFace client with a local model.
        
        Args:
            model_path: Path to the local HuggingFace model directory
        """
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_llava = False
        self._setup_path()
        self._load_model()
    
    def _setup_path(self):
        """Add model directory to sys.path for local packages."""
        # Add model directory to path so local packages can be imported
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        
        # Also add parent directory if there are local packages
        parent_dir = os.path.dirname(self.model_path)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
    def _load_model(self):
        """Load the model and processor from the specified path."""
        try:
            # First, try to determine the right Auto class from config
            config_path = os.path.join(self.model_path, "config.json")
            auto_class = AutoModel  # Default
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                auto_map = config.get("auto_map", {})
                model_type = config.get("model_type", "")
                self.is_llava = "llava" in model_type.lower() or "safe_llava" in model_type.lower()
                
                # Determine the right Auto class based on auto_map
                if self.is_llava:
                    # For LLaVA-based models, always use AutoModelForCausalLM
                    auto_class = AutoModelForCausalLM
                elif "AutoModelForVision2Seq" in auto_map:
                    auto_class = AutoModelForVision2Seq
                elif "AutoModelForCausalLM" in auto_map:
                    auto_class = AutoModelForCausalLM
                elif "AutoModel" in auto_map:
                    auto_class = AutoModel
            
            print(f"Using {auto_class.__name__} to load model from {self.model_path}")
            print(f"Is LLaVA-based model: {self.is_llava}")
            
            # Load model with the appropriate Auto class
            self.model = auto_class.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to('cuda')
            self.model.eval()
            
            if self.is_llava:
                # For LLaVA models, use AutoTokenizer with use_fast=False
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
                self.processor = self.tokenizer  # Set processor to tokenizer for compatibility
                
                # Load and move vision tower to GPU
                vision_tower = self.model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model()
                vision_tower = vision_tower.to('cuda')
                print(f"Vision tower loaded and moved to GPU")
            else:
                # For other models, use AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.tokenizer = self.processor
            
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
        
        if self.model is None:
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
            
            if self.is_llava:
                # For LLaVA-based models, use the specific LLaVA inference method
                return self._generate_llava_response(prompt, valid_image_paths, max_tokens)
            else:
                # For other models, use standard method
                return self._generate_standard_response(prompt, valid_image_paths, max_tokens)
            
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
    
    def _generate_llava_response(self, prompt: str, valid_image_paths: List[str], max_tokens: int) -> Dict[str, Any]:
        """Generate response for LLaVA-based models."""
        from safellava.mm_utils import tokenizer_image_token
        from safellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from safellava.conversation import conv_templates
        
        # Prepare conversation prompt using LLaVA conversation template
        conv = conv_templates["llava_v1"].copy()
        
        if valid_image_paths:
            # Process images
            images = [Image.open(path).convert("RGB") for path in valid_image_paths]
            
            # Get image processor from vision tower
            vision_tower = self.model.get_vision_tower()
            image_processor = vision_tower.image_processor
            
            # Preprocess images
            image_tensors = []
            for image in images:
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor)
            
            # Stack images and move to GPU
            image_tensor = torch.cat(image_tensors, dim=0).to('cuda', dtype=torch.float16)
            
            # Add image token to prompt
            image_token_str = DEFAULT_IMAGE_TOKEN + "\n"
            question = image_token_str + prompt
        else:
            image_tensor = None
            question = prompt
        
        # Format conversation
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to('cuda')
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True,
            )
        
        # Decode output
        generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Calculate token usage
        prompt_tokens = input_ids.shape[1]
        completion_tokens = output_ids.shape[1] - prompt_tokens
        
        return {
            "response": generated_text,
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "model": os.path.basename(self.model_path),
            "prompt": prompt,
            "image_paths": valid_image_paths
        }
    
    def _generate_standard_response(self, prompt: str, valid_image_paths: List[str], max_tokens: int) -> Dict[str, Any]:
        """Generate response for standard (non-LLaVA) models."""
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
        
        # Try to use chat template
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: just use the prompt text directly
            text = prompt
        
        # Process inputs
        if valid_image_paths:
            images = [Image.open(path).convert("RGB") for path in valid_image_paths]
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
        
        prompt_tokens = len(inputs.input_ids[0])
        
        return {
            "response": output_text,
            "usage": {
                "completion_tokens": len(output_ids),
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + len(output_ids)
            },
            "model": os.path.basename(self.model_path),
            "prompt": prompt,
            "image_paths": valid_image_paths
        }

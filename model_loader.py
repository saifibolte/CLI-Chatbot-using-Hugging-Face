"""
Model Loading Module

This module handles loading and initialization of Hugging Face language models and tokenizers, creating text generation pipelines for the chatbot.

"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Default model - TinyLlama is lightweight and good for testing/development
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def _resolve_device(device=None):
    """
    Resolve and validate the device specification for model loading.
    
    This function handles various device input formats and automatically
    detects the best available device (GPU vs CPU).

    """
    # If device is already an integer, return it as-is
    if isinstance(device, int):
        return device

    # Handle string device specifications
    if isinstance(device, str):
        if device.lower() in {"cpu", "-1"}:
            return -1
        if device.isdigit():
            return int(device)
    
    # Auto-detect: use GPU 0 if CUDA available, otherwise CPU
    return 0 if torch.cuda.is_available() else -1

def load_pipeline(model_name=DEFAULT_MODEL, dtype=None, device=None, trust_remote_code=False):

    """
    Load a Hugging Face model and create a text generation pipeline.
    
    This function handles the complete model loading process including:
    - Loading tokenizer and model from Hugging Face Hub
    - Setting up appropriate data types and device placement
    - Creating a text generation pipeline for inference
    
    """

    # Resolve the device specification to a standard format
    resolved_device = _resolve_device(device)

    # Convert string dtype to PyTorch dtype object
    # e.g., "float16" becomes torch.float16
    torch_dtype = getattr(torch, dtype) if dtype else None

    # Load tokenizer from Hugging Face Hub
    # use_fast=True for faster tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)

    # Load the language model with specified configuration
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch_dtype, # Memory optimization (e.g., float16 vs float32)
                                                 trust_remote_code=trust_remote_code) # Allow custom model code

    # Create text generation pipeline
    # This wraps the model and tokenizer for easy text generation
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=resolved_device)

    # Return both pipeline and tokenizer
    # Pipeline for generation, tokenizer for prompt formatting
    return gen, tokenizer

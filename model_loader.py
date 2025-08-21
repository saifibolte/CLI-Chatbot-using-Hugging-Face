
"""
model_loader.py
----------------
Utilities to load a small Hugging Face text-generation model and build a pipeline.
"""
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_textgen_pipeline(
    model_name: str = "distilgpt2",
    device: Optional[int] = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create and return a transformers text-generation pipeline.

    Args:
        model_name: HF model repo or local path.
        device: GPU index (0, 1, ...) or -1 for CPU. If None, choose automatically.
        dtype: "auto" (recommended), or one of: "float16", "bfloat16", "float32".
        trust_remote_code: pass True for models requiring custom code.
        pipeline_kwargs: extra kwargs forwarded to pipeline(...).
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    # Choose device automatically if not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Map dtype string to torch dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, "auto")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype if torch_dtype != "auto" else None,
        trust_remote_code=trust_remote_code,
    )

    # Some GPT2-like models don't have a pad token; align pad with eos to avoid warnings.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback: add a pad token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **pipeline_kwargs,
    )
    return textgen, tokenizer

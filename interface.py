
"""
interface.py
------------
Command-line chatbot loop integrating the model loader and memory.
"""
import argparse
import sys
import re
import torch
from typing import Optional
from model_loader import load_textgen_pipeline
from chat_memory import ChatMemory

STOP_SEQS = ["\nUser:", "\nSystem:", "\nAssistant:", "\n\n"]

def extract_assistant_reply(full_text: str) -> str:
    """
    Extract just the assistant's reply from the generated text by trimming
    everything before the final 'Assistant:' and stopping at any STOP_SEQS.
    """
    # Keep the last occurrence of "Assistant:"
    idx = full_text.rfind("Assistant:")
    reply = full_text[idx + len("Assistant:") :].strip() if idx != -1 else full_text

    # Cut at the earliest stop sequence if present
    cut_points = [reply.find(s) for s in STOP_SEQS if s in reply]
    cut_points = [p for p in cut_points if p >= 0]
    if cut_points:
        reply = reply[: min(cut_points)].strip()
    return reply

def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Local CLI chatbot using a Hugging Face model.")
    parser.add_argument("--model", type=str, default="distilgpt2", help="HF model name or local path")
    parser.add_argument("--window", type=int, default=4, help="Sliding window size in turns (user messages)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="Optional system prompt")
    parser.add_argument("--device", type=int, default=None, help="GPU index (e.g., 0) or -1 for CPU. Default: auto")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model} ...")
    pipe, tokenizer = load_textgen_pipeline(
        model_name=args.model,
        device=args.device,
        dtype="auto",
        pipeline_kwargs={},
    )
    print("Model loaded. Type /exit to quit.\n")

    memory = ChatMemory(window_size=args.window, system_prompt=args.system)

    try:
        while True:
            try:
                user_text = input("User: ").strip()
            except EOFError:
                print("\nExiting chatbot. Goodbye!")
                break

            if user_text.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
            if not user_text:
                continue

            memory.add_user(user_text)
            prompt = memory.build_prompt()

            # Generate
            outputs = pipe(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            full_text = outputs[0]["generated_text"]
            assistant_reply = extract_assistant_reply(full_text)
            if not assistant_reply:
                assistant_reply = "(No response generated.)"

            print(f"Bot: {assistant_reply}")
            memory.add_assistant(assistant_reply)

    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")

if __name__ == "__main__":
    main()

"""
CLI Chatbot Interface Module

This module provides the main command-line interface for the chatbot,
handling user interaction, prompt formatting, text generation, and 
optional response verification.
"""

from __future__ import annotations
import argparse, sys, torch
from model_loader import load_pipeline, DEFAULT_MODEL
from chat_memory import SlidingWindowMemory

def build_prompt(messages, tokenizer):
    """
    Convert conversation messages into a formatted prompt string.
    
    This function tries to use the model's native chat template if available,
    falling back to a manual format that works with most language models.
        
    """

    # using the tokenizer's built-in chat template first
    # This ensures compatibility with models that have specific formatting requirements
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    
    # Manual prompt formatting as fallback
    # Start with system message
    lines = ["System: " + messages[0]['content'].strip(), "", "Conversation:"]

    # Add all conversation messages (skip system message at index 0)
    for msg in messages[1:]:
        role = msg['role'].capitalize() # "user" -> "User", "assistant" -> "Assistant"
        lines.append(f"{role}: {msg['content'].strip()}")
    
    # Add prompt for next assistant response
    lines.append("Assistant:")

    # Join all lines and ensure proper formatting
    return "\n".join(lines).strip() + "\n"

def postprocess_reply(prompt, generated):
    """
    Clean up the generated text by removing the input prompt and stop sequences.
    
    Language models often return the full input prompt plus the generated text.
    This function extracts only the new generated content and removes unwanted
    continuation patterns.
    
    Args:
        prompt (str): The original input prompt sent to the model
        generated (str): The full text returned by the model
    
    Returns:
        str: Cleaned assistant response with prompt and stop tokens removed
    """
    # Remove the original prompt from the generated text if present
    if generated.startswith(prompt):
        text = generated[len(prompt):]
    else:
        text = generated

    # Define stop sequences that indicate end of assistant response
    # These patterns often appear when models continue generating unintended content
    stops = ["</s>", "<|endoftext|>", "<|assistant|>", "<|user|>", "\nUser:", "\nSystem:", "\nAssistant:"]

    # Find the first occurrence of any stop sequence and truncate there
    for s in stops:
        idx = text.find(s)
        if idx != -1 and idx > 0: # Found and not at the very beginning
            text = text[:idx]
    return text.strip()

def main(argv=None):

    """
    Main function that handles command-line arguments and runs the chatbot loop.
    
    This function sets up argument parsing, loads the model, initializes memory,
    and runs the interactive chat loop with optional response verification.
    
    Args:
        argv: Command line arguments (for testing), uses sys.argv if None
    """
     # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Local CLI Chatbot")

    # Model, Generation, Technical, Behaviour configuration arguments
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--system", type=str, default="You are a factual assistant. If unsure, say 'I don't know.'")
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    # Set random seed for reproducible results
    torch.manual_seed(args.seed)

    # Load the model and tokenizer
    print(f"Loading model: {args.model}")
    pipe, tokenizer = load_pipeline(model_name=args.model, dtype=args.dtype, device=args.device, trust_remote_code=args.trust_remote_code)
    print("Model loaded. Type /exit to quit, /reset to clear memory.\n")

    # Initialize conversation memory with sliding window
    memory = SlidingWindowMemory(max_turns=args.window, system_prompt=args.system)

    # Main chat loop
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot. Goodbye!")
            break
        
        # Skip empty inputs
        if not user:
            continue

        # Handle special commands
        if user.lower() == "/exit":
            print("Exiting chatbot. Goodbye!")
            break
        if user.lower() == "/reset":
            memory.clear()
            print("(memory cleared)")
            continue
        
        # Add user message to memory and build prompt
        memory.add_user(user)
        prompt = build_prompt(memory.messages, tokenizer)

        # Configure text generation parameters
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_full_text": True,
        }

        # Set sampling strategy based on temperature
        if args.temperature == 0.0:
            #Deterministic generation (greedy search)
            gen_kwargs["do_sample"] = False
        else:
            # Stochastic sampling with temperature and top-p
            gen_kwargs["do_sample"] = True
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        # Enable beam search if requested 
        if args.num_beams > 1:
            gen_kwargs["num_beams"] = args.num_beams

        # Generate response using the pipeline
        outputs = pipe(prompt, **gen_kwargs)
        generated = outputs[0]["generated_text"]

        # Clean up the generated response
        reply = postprocess_reply(prompt, generated)

        # Fallback if postprocessing removes everything
        if not reply:
            reply = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()

        # Optional response verification for factual accuracy
        if args.verify:
            # Create verification prompt
            verify_prompt = (
                f"System: You are a helpful assistant. Respond only with factual correctness.\n\n"
                f"Q: {user}\nA: {reply}\n\n"
                "If correct, respond with 'VERIFIED'. If incorrect, respond with 'CORRECTED' on the first line, then the correct answer."
            )

            # Generate verification response
            v_out = pipe(verify_prompt, **gen_kwargs)
            v_text = v_out[0]["generated_text"]
            v_reply = postprocess_reply(verify_prompt, v_text)

            # Parse verification result
            first_line = v_reply.splitlines()[0].strip().lower() if v_reply else ""
            if "corrected" in first_line:
                # Use corrected answer if available
                corrected = "\n".join(v_reply.splitlines()[1:]).strip()
                if corrected:
                    reply = corrected
                else:
                    reply = reply + " (⚠️ correction ambiguous)"
            elif "verified" in first_line:
                # Original answer is verified as correct
                pass
            else:
                # Verification was unclear
                reply = reply + " (⚠️ verification ambiguous)"

        # Display the response and add to memory
        print(f"Bot: {reply}\n")
        memory.add_assistant(reply)

if __name__ == "__main__":
    main()

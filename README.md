
# Local CLI Chatbot (Hugging Face)

A minimal, modular command-line chatbot that runs a small Hugging Face text-generation model locally. 
It maintains short-term conversation memory using a sliding window so replies stay coherent across turns.

## Project Layout
```text
model_loader.py   # Model + tokenizer loading via Hugging Face
chat_memory.py    # Sliding-window memory buffer
interface.py      # CLI loop and integration
README.md         # This file
```

## Requirements

- Python 3.9+
- pip packages:
  - `transformers`
  - `torch` (CPU or CUDA build)
  - `accelerate` (optional but recommended)

Install:
```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install torch transformers accelerate
```

> **Tip:** If you use models stored with Xet, you may see a message about `hf_xet`. It's optional.
> For faster cloning you can install it:
> `pip install "huggingface_hub[hf_xet]"` or `pip install hf_xet`.

## Choose a Model

The default is `distilgpt2` (fast, very small). You can try other small models, for example:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2` (non-commercial license; CPU-only may be slow)
- Any local path to a compatible causal LM

> If a model lacks a pad token (e.g., GPT-2 variants), we map pad to eos automatically.

## Run the Chatbot

```bash
# Basic usage (CPU or GPU auto-detected)
python interface.py --model distilgpt2 --window 4

# Specify GPU index or force CPU
python interface.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device 0
python interface.py --model distilgpt2 --device -1   # force CPU

# Tune generation
python interface.py --model distilgpt2 --max-new-tokens 120 --temperature 0.8 --top-p 0.92
```

When the program starts you'll see:
```
Loading model: distilgpt2 ...
Model loaded. Type /exit to quit.
```

Then chat:
```
User: What is the capital of France?
Bot: The capital of France is Paris.
User: And what about Italy?
Bot: The capital of Italy is Rome.
User: /exit
Exiting chatbot. Goodbye!
```

## How Memory Works

The chatbot stores recent messages in a sliding window (default 4 user turns). 
Each prompt is constructed like:

```
System: You are a helpful assistant.
User: <message 1>
Assistant: <reply 1>
...
User: <latest message>
Assistant:
```

We stop generation when the model begins a new speaker block (e.g., `\nUser:`).

## Notes & Troubleshooting

- If you previously tried using `from transformers import Conversation` and saw an ImportError,
  note that this project **does not** rely on that API. We format prompts ourselves for stability.
- If generation is too verbose or cuts off early, adjust `--max-new-tokens` and sampling params.
- On very small models like `distilgpt2`, factual accuracy is limited—this is a framework
  for integrating **any** compatible model, not a knowledge authority.
- For best responsiveness, prefer lightweight models or enable a GPU.

## License

MIT for this code. Each model has its own license; check the model card before use.

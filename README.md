
# Local CLI Chatbot (Hugging Face)

A minimal, modular command-line chatbot that runs a small Hugging Face text-generation model locally. 
It maintains short-term conversation memory using a sliding window so replies stay coherent across turns.

## My System

`CPU: i3-1005G1 Dual Core`
`RAM: 4GB`
`GPU: Intel UHD Graphics`

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
  - `transformers - version: 2.19.0`
  - `torch - version: 2.8.0 ` (CPU or CUDA build)
  - `accelerate - version: 1.10` (optional but recommended)

Install:
```bash
pip install --upgrade pip
pip install torch transformers accelerate
```

> **Tip:** If you use models stored with Xet, you may see a message about `hf_xet`. It's optional.
> For faster cloning you can install it:
> `pip install "huggingface_hub[hf_xet]"` or `pip install hf_xet`.

## Choose a Model

The default is `TinyLlama-1.1B-Chat-v1.02` (fast, small). You can try other small models, for example:

- `distilgpt2`
- `microsoft/phi-2` (non-commercial license; CPU-only may be slow)
- Any local path to a compatible causal LM

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
User: What is the capital of India?
Bot: The capital of France is New Delhi.
User: And what about France?
Bot: France's capital city is Paris.
User: Italy?
Bot: Italy's captal is Rome.
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

## Notes & Troubleshooting

- If generation is too verbose or cuts off early, adjust `--max-new-tokens` and sampling params.
- On very small models like `distilgpt2`, factual accuracy is limited—this is a framework for integrating **any** compatible model, not a knowledge authority.
- For best responsiveness, prefer lightweight models or enable a GPU.


# Inference

Run LLM inference on a dataset, producing a response for each question. Supports
local HuggingFace models and vLLM, plus the OpenAI, OpenRouter, and Anthropic
APIs. Used as a library — `run_inference` is the single entry point.

## Python Usage

```python
from pathlib import Path
from src.inference import run_inference, InferenceConfig
from src.common.config import DatasetConfig, GenerationConfig

config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    provider="local",
    dataset=DatasetConfig(
        source="huggingface",
        name="vicgalle/alpaca-gpt4",
        max_samples=10,
    ),
    generation=GenerationConfig(max_new_tokens=500),
    output_path=Path("scratch/output.jsonl"),
)
dataset, result = run_inference(config)
```

`run_inference_async` is the awaitable variant. Both resume from the last written
row of `output_path` by default; pass `overwrite_output=True` to start fresh.

## Providers

- **`local`**: Loads the model via HuggingFace `transformers`. Runs on local GPU.
- **`vllm`**: Loads the model via vLLM for higher throughput (~2× vs `local`). Supports LoRA adapters via `LoRARequest`. Recommended for large inference runs.
- **`openai`**: Calls the OpenAI API.
- **`openrouter`**: Calls the OpenRouter API (OpenAI-compatible).
- **`anthropic`**: Calls the Anthropic API.

Notes:
- For the `local` provider, `num_responses` is implemented via HF `num_return_sequences` in a single generate call, so the effective generation batch is `batch_size * num_responses` — lower `batch_size` if you raise `num_responses` to avoid OOM.
- Local prompt formatting defaults to `auto`: if the tokenizer exposes a HuggingFace `chat_template`, prompts are formatted as chat messages (`user` plus optional `system`), otherwise raw prompts are used.
- OpenAI reasoning models (O-series, GPT‑5+ reasoning-capable) count reasoning against the output-token budget, so plan a higher `max_new_tokens`.
- `openai_batch.py` (OpenAI Batch API) is present but guarded by `NotImplementedError` — untested; remove the guard and validate before use.

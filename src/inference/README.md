# Inference

Run LLM inference on a dataset, producing a response for each question. Supports local HuggingFace models plus OpenAI, OpenRouter, and Anthropic APIs.

Important: the CLI runs in canonical run-dir mode. `--run-dir` is required.

## CLI Usage

```bash
# Local model inference
uv run python -m src_dev.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --run-dir scratch/runs/my_inference_run \
  --local-prompt-format auto \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 100 \
  --output-path scratch/inference_output.jsonl

# vLLM (higher throughput, ~2× vs local)
uv run python -m scripts.inference \
  --provider vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-dir scratch/runs/my_inference_run \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 100 \
  --output-path scratch/inference_output.jsonl

# OpenAI API
uv run python -m src_dev.inference \
  --provider openai \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-dir scratch/runs/my_inference_run \
  --openai-api-key-env OPENAI_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# OpenAI Batch API (Responses endpoint)
uv run python -m src_dev.inference \
  --provider openai \
  --openai-batch \
  --model gpt-5-nano-2025-08-07 \
  --run-dir scratch/runs/my_inference_run \
  --openai-api-key-env OPENAI_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 200 \
  --output-path scratch/inference_output.jsonl

Note: Batch runs are asynchronous and will poll until completion (default 24h window).
Note: OpenAI Batch support is not yet tested. The batch runner currently raises
`NotImplementedError` to prevent accidental use. Remove the guard and test before use.

# OpenRouter API
uv run python -m src_dev.inference \
  --provider openrouter \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-dir scratch/runs/my_inference_run \
  --openrouter-api-key-env OPENROUTER_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# Resume from last written row (default behavior when output exists)
uv run python -m src_dev.inference \
  --provider openai \
  --run-dir scratch/runs/my_inference_run \
  --dataset-name vicgalle/alpaca-gpt4 \
  --output-path scratch/inference_output.jsonl

# Force fresh run from row 0
uv run python -m src_dev.inference \
  --provider openai \
  --run-dir scratch/runs/my_inference_run \
  --dataset-name vicgalle/alpaca-gpt4 \
  --output-path scratch/inference_output.jsonl \
  --overwrite-output

# Anthropic API
uv run python -m src_dev.inference \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --run-dir scratch/runs/my_inference_run \
  --anthropic-api-key-env ANTHROPIC_API_KEY \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 50 \
  --output-path scratch/inference_output.jsonl

# Multiple responses per prompt
uv run python -m src_dev.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --run-dir scratch/runs/my_inference_run \
  --num-responses 3 \
  --temperature 0.9 \
  --dataset-name vicgalle/alpaca-gpt4 \
  --output-path scratch/multi_response.jsonl

Note: For the local provider, `num_responses` is implemented via HF
`num_return_sequences` inside a single generate call. The effective
generation batch is `batch_size * num_responses`, so if you increase
`num_responses` you may need to decrease `batch_size` proportionally
to avoid OOM.

Note: Local prompt formatting defaults to `auto`. In `auto`, if the
tokenizer exposes a HuggingFace `chat_template`, prompts are formatted as chat
messages (`user` plus optional `system`), otherwise raw prompts are used.
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name or HuggingFace path | `meta-llama/Llama-3.1-8B-Instruct` |
| `--provider` | `local`, `openai`, `openrouter`, or `anthropic` | `local` |
| `--dataset-name` | HuggingFace dataset name | — |
| `--dataset-source` | `huggingface` or `local` | `huggingface` |
| `--max-samples` | Limit number of samples | all |
| `--max-new-tokens` | Max tokens to generate | `100000` |
| `--temperature` | Sampling temperature | `0.7` |
| `--batch-size` | Batch size | `8` |
| `--num-responses` | Responses per prompt | `1` |
| `--local-prompt-format` | Local prompt mode: `auto`, `chat`, `plain` | `auto` |
| `--local-chat-system-prompt` | Optional system prompt for chat-formatted local prompts | — |
| `--max-concurrent` | Max concurrent API requests | `10` |
| `--timeout` | Request timeout in seconds (0 disables) | `60` |
| `--retry-max-retries` | Max retry attempts for API calls | `3` |
| `--retry-backoff-factor` | Exponential backoff multiplier | `2.0` |
| `--fail-fast` | Stop on first API error | `false` |
| `--output-path` | Output JSONL path | — |
| `--run-dir` | Canonical run directory (required) | — |
| `--no-resume` | Disable resume-from-existing-output behavior | `false` |
| `--overwrite-output` | Truncate output file before running | `false` |
| `--openai-base-url` | OpenAI API base URL | — |
| `--openai-api-key-env` | Env var name for OpenAI API key | `OPENAI_API_KEY` |
| `--openai-reasoning-effort` | OpenAI reasoning effort | — |
| `--openai-verbosity` | OpenAI verbosity | — |
| `--openai-batch` | Use OpenAI Batch API | `false` |
| `--openai-batch-completion-window` | Batch completion window | `24h` |
| `--openai-batch-poll-interval` | Batch poll interval (seconds) | `10` |
| `--openai-batch-timeout` | Batch timeout (seconds) | — |
| `--openai-batch-include-sampling` | Include temperature/top_p in batch | `false` |
| `--openai-batch-run-dir` | Run directory under `scratch/` | — |
| `--openai-batch-resume` | Resume batch from run-dir metadata | `false` |
| `--openrouter-base-url` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `--openrouter-api-key-env` | Env var name for OpenRouter API key | `OPENROUTER_API_KEY` |
| `--openrouter-app-url` | Optional OpenRouter app URL | — |
| `--openrouter-app-name` | Optional OpenRouter app name | — |
| `--anthropic-api-key-env` | Env var name for Anthropic API key | `ANTHROPIC_API_KEY` |

## Python Usage

```python
from pathlib import Path
from src_dev.inference import run_inference, InferenceConfig
from src_dev.common.config import DatasetConfig, GenerationConfig

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

## Providers

- **`local`**: Loads model via HuggingFace `transformers`. Runs on local GPU.
- **`vllm`**: Loads model via vLLM for higher throughput (~2× vs `local`). Supports LoRA adapters via `LoRARequest`. Recommended for large inference runs.
- **`openai`**: Calls the OpenAI API.
- **`openrouter`**: Calls the OpenRouter API (OpenAI-compatible).
- **`anthropic`**: Calls the Anthropic API.

Note: OpenAI reasoning models (O-series and GPT‑5+ reasoning-capable models) typically need higher output token limits because the reasoning is included in the output token budget. Plan higher `max_new_tokens` when using them.

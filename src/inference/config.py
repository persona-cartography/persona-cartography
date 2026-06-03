"""Inference stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.common.config import DatasetConfig, GenerationConfig


class LocalProviderConfig(BaseModel):
    """Local model loading settings (HuggingFace transformers)."""

    model_config = {"arbitrary_types_allowed": True}

    dtype: str = "bfloat16"
    device_map: str = "auto"
    revision: str = "main"
    adapter_path: str | None = None
    prompt_format: Literal["auto", "chat", "plain"] = "auto"
    chat_system_prompt: str | None = None
    truncate_inputs: bool = True
    attn_implementation: str | None = None
    # Optional pre-loaded (model, tokenizer) pair.  When set, LocalProvider
    # skips from_pretrained entirely and uses this object directly.  The caller
    # owns the model lifetime; LocalProvider will NOT free it.
    # Excluded from serialization because the model object is not JSON-serializable.
    preloaded_model: tuple | None = Field(default=None, exclude=True)


class OpenAIBatchConfig(BaseModel):
    """Batch API settings for OpenAI Responses endpoint."""

    enabled: bool = False
    completion_window: str = "24h"
    poll_interval_seconds: int = 10
    timeout_seconds: int | None = None
    include_sampling: bool = False
    run_dir: str | None = None
    resume: bool = False


class RetryConfig(BaseModel):
    """API retry configuration."""

    max_retries: int = 3
    backoff_factor: float = 2.0


class OpenAIProviderConfig(BaseModel):
    """OpenAI API settings."""

    base_url: str | None = None  # None = use default OpenAI API
    api_key_env: str = "OPENAI_API_KEY"  # Environment variable name for API key
    reasoning_effort: str | None = None  # "none" | "low" | "medium" | "high"
    verbosity: str | None = None  # "low" | "medium" | "high"
    batch: OpenAIBatchConfig = OpenAIBatchConfig()


class OpenRouterProviderConfig(BaseModel):
    """OpenRouter API settings (OpenAI-compatible)."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_url: str | None = None
    app_name: str | None = None
    # Provider routing object passed as-is to OpenRouter's `provider` field.
    # See https://openrouter.ai/docs/guides/routing/provider-selection
    # e.g. {"order": ["fireworks"], "allow_fallbacks": False}
    # None = use OpenRouter's default routing behaviour.
    provider_routing: dict | None = None
    # Reasoning/extended thinking config passed as-is to OpenRouter's `reasoning` field.
    # See https://openrouter.ai/docs/use-cases/reasoning-tokens
    # e.g. {"effort": "high"} for OpenAI/xAI models, {"max_tokens": 16000} for Anthropic.
    # None = no reasoning.
    reasoning: dict | None = None


class AnthropicProviderConfig(BaseModel):
    """Anthropic API settings."""

    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int | None = None


class VllmProviderConfig(BaseModel):
    """vLLM engine settings."""

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    # LoRA adapter path (local dir or HF repo).  None = base model only.
    adapter_path: str | None = None
    # Enable automatic prefix caching (reuse KV cache for shared prefixes).
    enable_prefix_caching: bool = True
    # Enforce eager mode (disables CUDA graphs). Useful for debugging.
    enforce_eager: bool = False
    # Max simultaneous in-batch LoRA adapters (GPU hot slots).
    max_loras: int = 1
    # Total LoRA adapters in CPU cache. Defaults to max_loras.
    max_cpu_loras: int | None = None
    # Max LoRA rank vLLM will accept. Default 64 covers single-trait
    # adapters; baked LoRA *soups* (e.g. C+ ⊕ O− at rank 64 each) have
    # combined rank = sum, so set this to the highest rank expected.
    max_lora_rank: int = 64
    # Shard the model across N GPUs with tensor parallelism. Default 1 (no TP).
    # Useful for fitting larger models and freeing KV-cache headroom on smaller
    # GPUs (e.g. A40/48GB). vLLM requires model head/hidden dims to be
    # divisible by this value.
    tensor_parallel_size: int = 1
    # Optional Jinja2 chat template override. Passed through to
    # ``LLM.chat(chat_template=...)`` on every call, overriding whatever
    # template the tokenizer ships with (or supplying one for legacy
    # tokenizers whose ``chat_template`` is unset — Koala-13B,
    # OpenAssistant-Pythia-12B, etc.). Typically populated by callers
    # via ``src.psychometric.chat_templates.lookup_template(model)``.
    chat_template: str | None = None


class InferenceConfig(BaseModel):
    """Configuration for the inference stage.

    Example:
        config = InferenceConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            provider="local",
            dataset=DatasetConfig(
                source="huggingface",
                name="vicgalle/alpaca-gpt4",
                max_samples=10,
            ),
            generation=GenerationConfig(
                max_new_tokens=500,
                temperature=0.7,
            ),
            output_path=Path("scratch/inference_output.jsonl"),
        )
        result = run_inference(config)
    """

    # Model settings
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    provider: str = "local"  # "local", "openai", "openrouter", "anthropic"

    # Dataset settings
    dataset: DatasetConfig = DatasetConfig()

    # Generation settings
    generation: GenerationConfig = GenerationConfig()

    # Async + retry settings (for remote providers)
    max_concurrent: int = 10
    timeout: int | None = 60
    retry: RetryConfig = RetryConfig()
    continue_on_error: bool = True
    log_failures: bool = True

    # Provider-specific settings
    local: LocalProviderConfig = LocalProviderConfig()
    openai: OpenAIProviderConfig = OpenAIProviderConfig()
    openrouter: OpenRouterProviderConfig = OpenRouterProviderConfig()
    anthropic: AnthropicProviderConfig = AnthropicProviderConfig()
    vllm: VllmProviderConfig = VllmProviderConfig()

    # Output
    output_path: Path | None = None  # If None, returns dataset without saving
    run_dir: Path | None = None  # Canonical run directory under scratch/runs/<run_id>
    system_prompt: str | None = None
    max_attempts_per_sample: int | None = 3
    resume: bool = True
    overwrite_output: bool = False


class InferenceResult(BaseModel):
    """Result from running inference."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    num_failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    batch_id: str | None = None
    batch_status: str | None = None

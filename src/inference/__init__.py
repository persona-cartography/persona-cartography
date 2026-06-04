"""Inference module for running LLM inference on datasets.

Example:
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
"""

from src.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    InferenceResult,
    LocalProviderConfig,
    OpenAIBatchConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from src.inference.run import run_inference, run_inference_async
from src.inference.providers import get_provider
from src.inference.providers.base import InferenceProvider

__all__ = [
    # Config classes
    "InferenceConfig",
    "InferenceResult",
    "LocalProviderConfig",
    "OpenAIBatchConfig",
    "OpenAIProviderConfig",
    "OpenRouterProviderConfig",
    "AnthropicProviderConfig",
    "RetryConfig",
    # Run function
    "run_inference",
    "run_inference_async",
    # Providers
    "get_provider",
    "InferenceProvider",
]

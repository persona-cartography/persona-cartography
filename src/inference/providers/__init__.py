"""Inference provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig
    from src.inference.providers.base import InferenceProvider


def get_provider(name: str, config: "InferenceConfig") -> "InferenceProvider":
    """Get an inference provider by name.

    Args:
        name: Provider name ("local", "openai", "openrouter", or "anthropic").
        config: Inference configuration.

    Returns:
        Initialized inference provider.

    Raises:
        ValueError: If provider name is unknown.
    """
    # Lazy imports to avoid circular dependencies
    if name == "local":
        from src.inference.providers.local import LocalProvider

        return LocalProvider(config)
    elif name == "vllm":
        from src.inference.providers.vllm import VllmProvider

        return VllmProvider(config)
    elif name == "openai":
        from src.inference.providers.openai import OpenAIProvider

        return OpenAIProvider(config)
    elif name == "openrouter":
        from src.inference.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(config)
    elif name == "anthropic":
        from src.inference.providers.anthropic import AnthropicProvider

        return AnthropicProvider(config)
    else:
        raise ValueError(
            f"Unknown inference provider: {name!r}. "
            "Available: ['local', 'vllm', 'openai', 'openrouter', 'anthropic']"
        )

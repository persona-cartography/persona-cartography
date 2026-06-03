"""Abstract base class for inference providers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any


TokenUsage = dict[str, int]
ChatMessage = dict[str, str]
PromptInput = str | list[ChatMessage]


def empty_usage() -> TokenUsage:
    """Return a zeroed token usage dict."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def accumulate_usage(total: TokenUsage, usage: TokenUsage | None) -> None:
    """Accumulate token usage into total."""
    if not usage:
        return
    total["input_tokens"] += usage.get("input_tokens", 0)
    total["output_tokens"] += usage.get("output_tokens", 0)
    total["total_tokens"] += usage.get("total_tokens", 0)


def extract_usage(usage_obj: Any) -> TokenUsage | None:
    """Extract token usage from API response usage object.

    Supports multiple API response formats:
    - Anthropic: input_tokens, output_tokens
    - OpenAI: input_tokens/prompt_tokens, output_tokens/completion_tokens, total_tokens
    - OpenRouter: prompt_tokens, completion_tokens, total_tokens

    Args:
        usage_obj: Usage object from API response (can be dict or object with attributes).

    Returns:
        Normalized TokenUsage dict with input_tokens, output_tokens, total_tokens,
        or None if usage_obj is None.
    """
    if usage_obj is None:
        return None

    # Extract from dict or object attributes
    if isinstance(usage_obj, dict):
        input_tokens = usage_obj.get("input_tokens", usage_obj.get("prompt_tokens", 0)) or 0
        output_tokens = usage_obj.get("output_tokens", usage_obj.get("completion_tokens", 0)) or 0
        total_tokens = usage_obj.get("total_tokens", 0) or 0
    else:
        # Try input_tokens first, fallback to prompt_tokens
        input_tokens = getattr(usage_obj, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage_obj, "prompt_tokens", 0)

        # Try output_tokens first, fallback to completion_tokens
        output_tokens = getattr(usage_obj, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage_obj, "completion_tokens", 0)

        total_tokens = getattr(usage_obj, "total_tokens", 0)

    # Convert to int and handle None values
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = int(total_tokens or 0)

    # Calculate total if not provided
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class InferenceProvider(ABC):
    """Abstract base class for inference providers.

    Providers are initialized with configuration and handle their own
    model/client setup in __init__. The interface focuses on generation.
    """

    @abstractmethod
    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters (temperature, max_tokens, num_responses, etc.)

        Returns:
            Generated response string.
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated response strings in prompt-major order.
            If num_responses > 1, the list length should be
            len(prompts) * num_responses, with responses grouped per prompt.
        """
        pass

    async def generate_async(self, prompt: PromptInput, **kwargs) -> str:
        """Async wrapper for generate().

        Default implementation runs the sync method in a thread.
        Providers can override for true async behavior.
        """
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    async def generate_batch_async(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Async wrapper for generate_batch().

        Default implementation runs the sync method in a thread.
        Providers can override for true async behavior.
        """
        return await asyncio.to_thread(self.generate_batch, prompts, **kwargs)

    async def generate_batch_with_metadata_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        """Generate responses and return usage + failure counts.

        Default implementation calls generate_batch_async and returns empty usage.
        Failures are estimated by counting empty responses.
        """
        responses = await self.generate_batch_async(prompts, **kwargs)
        failed_count = sum(1 for response in responses if not response)
        return responses, empty_usage(), failed_count

    async def aclose(self) -> None:
        """Release provider resources."""
        return None

    async def generate_batch_with_details_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> tuple[list[str], list[TokenUsage | None], int]:
        """Generate responses and return per-response usage details when available."""
        responses, _, failed_count = await self.generate_batch_with_metadata_async(
            prompts, **kwargs
        )
        return responses, [None] * len(responses), failed_count

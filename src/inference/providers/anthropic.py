"""Anthropic API inference provider."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic

from src.inference.providers.remote_base import AsyncInferenceProvider
from src.inference.providers.base import PromptInput, TokenUsage

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_text(content: list[Any] | None) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _extract_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
    else:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class AnthropicProvider(AsyncInferenceProvider):
    """Inference provider using the Anthropic API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.anthropic_config = config.anthropic
        self.generation_config = config.generation
        self.model = config.model
        self.client = self._create_client()

    def _create_client(self) -> AsyncAnthropic:
        """Build a new AsyncAnthropic client (used after sync-path close or on first use)."""
        api_key = os.environ.get(self.anthropic_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.anthropic_config.api_key_env} environment variable."
            )
        return AsyncAnthropic(api_key=api_key)

    def _split_system_and_messages(
        self,
        prompt: PromptInput,
    ) -> tuple[str | None, list[dict[str, str]]]:
        if isinstance(prompt, str):
            return None, [{"role": "user", "content": prompt}]

        system_blocks = [
            msg["content"] for msg in prompt if msg.get("role") == "system"
        ]
        messages = [msg for msg in prompt if msg.get("role") != "system"]
        system_prompt = "\n\n".join(system_blocks) if system_blocks else None
        return system_prompt, messages

    async def _generate_one(
        self, prompt: PromptInput, **kwargs
    ) -> tuple[str, TokenUsage | None]:
        if getattr(self.client, "is_closed", lambda: True)():
            self.client = self._create_client()
        gen_cfg = self.generation_config
        max_tokens = kwargs.get(
            "max_tokens",
            kwargs.get(
                "max_new_tokens",
                self.anthropic_config.max_tokens or gen_cfg.max_new_tokens,
            ),
        )

        # Anthropic guidance recommends changing either temperature OR top_p, not both.
        # Some models now enforce this via strict validation.
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)
        temperature_explicit = "temperature" in kwargs
        top_p_explicit = "top_p" in kwargs

        sampling_variants: list[dict[str, float]] = []
        if temperature_explicit and top_p_explicit:
            sampling_variants.append({"temperature": temperature, "top_p": top_p})
        if top_p_explicit and not temperature_explicit:
            sampling_variants.append({"top_p": top_p})
        if temperature_explicit or not top_p_explicit:
            sampling_variants.append({"temperature": temperature})
        sampling_variants.append({"top_p": top_p})
        sampling_variants.append({})

        # Deduplicate variants while preserving order.
        unique_variants: list[dict[str, float]] = []
        seen_keys: set[tuple[tuple[str, float], ...]] = set()
        for variant in sampling_variants:
            key = tuple(sorted(variant.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_variants.append(variant)

        last_exc: Exception | None = None
        for idx, sampling in enumerate(unique_variants):
            system_prompt, messages = self._split_system_and_messages(prompt)
            params: dict[str, object] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system_prompt:
                params["system"] = system_prompt
            params.update(sampling)
            if self.timeout is not None:
                params["timeout"] = self.timeout
            try:
                response = await self.client.messages.create(**params)
                text = _extract_text(response.content)
                usage = _extract_usage(response.usage)
                return text, usage
            except Exception as exc:
                last_exc = exc
                if idx == len(unique_variants) - 1:
                    break
                message = str(exc).lower()
                is_sampling_error = any(
                    needle in message
                    for needle in (
                        "temperature",
                        "top_p",
                        "nucleus",
                    )
                )
                if not is_sampling_error:
                    raise
                logger.warning(
                    "Anthropic sampling params rejected for model %s (%s). "
                    "Retrying with alternate sampling args.",
                    self.model,
                    exc,
                )

        assert last_exc is not None
        raise last_exc

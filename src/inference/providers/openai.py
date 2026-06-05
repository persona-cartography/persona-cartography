"""OpenAI API inference provider (Responses API)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from src.inference.providers.base import PromptInput, TokenUsage
from src.inference.providers.remote_base import AsyncInferenceProvider

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_output_text(response: Any) -> str:
    if response is None:
        return ""

    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str):
            return text.strip()

    outputs = getattr(response, "output", None)
    if outputs is None and isinstance(response, dict):
        outputs = response.get("output")
    outputs = outputs or []

    parts: list[str] = []
    for item in outputs:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        content = content or []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
            if block_type in ("output_text", "text") and block_text:
                parts.append(block_text)
    return "".join(parts).strip()


def _response_summary(response: Any) -> str:
    if response is None:
        return "status=unknown"
    if isinstance(response, dict):
        status = response.get("status")
        incomplete = response.get("incomplete_details")
        error = response.get("error")
    else:
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        error = getattr(response, "error", None)
    return f"status={status!r} incomplete={incomplete!r} error={error!r}"


def _extract_usage(response: Any) -> TokenUsage | None:
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = (
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        total_tokens = usage.get("total_tokens", 0) or 0
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    total_tokens = int(total_tokens or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_incomplete_reason(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        incomplete = response.get("incomplete_details")
    else:
        incomplete = getattr(response, "incomplete_details", None)
    if incomplete is None:
        return None
    if isinstance(incomplete, dict):
        reason = incomplete.get("reason")
    else:
        reason = getattr(incomplete, "reason", None)
    return str(reason) if reason is not None else None


class OpenAIProvider(AsyncInferenceProvider):
    """Inference provider using the OpenAI Responses API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.openai_config = config.openai
        self.generation_config = config.generation
        self.model = config.model
        self.client = self._create_client()
        logger.info("Initialized OpenAI provider with model: %s", self.model)

    def _create_client(self) -> AsyncOpenAI:
        """Build a new AsyncOpenAI client (used after sync-path close or on first use)."""
        api_key = os.environ.get(self.openai_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.openai_config.api_key_env} environment variable."
            )
        client_kwargs: dict[str, object] = {"api_key": api_key}
        if self.openai_config.base_url:
            client_kwargs["base_url"] = self.openai_config.base_url
            logger.info("Using custom base URL: %s", self.openai_config.base_url)
        return AsyncOpenAI(**client_kwargs)

    def _build_input(self, prompt: PromptInput) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    async def _generate_one(
        self, prompt: PromptInput, **kwargs
    ) -> tuple[str, TokenUsage | None]:
        gen_cfg = self.generation_config
        openai_cfg = self.openai_config

        raw_max_output_tokens = kwargs.get(
            "max_output_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        max_output_tokens = raw_max_output_tokens
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        def _sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered or "top_p" in lowered

        async def _create_response(
            prompt: PromptInput,
            *,
            include_sampling: bool = True,
            override_max_output_tokens: int | None = None,
        ):
            if getattr(self.client, "is_closed", lambda: True)():
                self.client = self._create_client()
            base_kwargs: dict[str, object] = {
                "model": self.model,
                "input": self._build_input(prompt),
                "max_output_tokens": override_max_output_tokens or max_output_tokens,
                "text": {
                    "format": {"type": "text"},
                },
            }
            if self.timeout is not None:
                base_kwargs["timeout"] = self.timeout
            if openai_cfg.verbosity:
                base_kwargs["text"]["verbosity"] = openai_cfg.verbosity
            if openai_cfg.reasoning_effort:
                base_kwargs["reasoning"] = {"effort": openai_cfg.reasoning_effort}
            if include_sampling:
                base_kwargs["temperature"] = temperature
                base_kwargs["top_p"] = top_p
            try:
                return await self.client.responses.create(**base_kwargs)
            except Exception as exc:
                if include_sampling and _sampling_error(str(exc)):
                    return await _create_response(prompt, include_sampling=False)
                raise

        response = await _create_response(prompt)
        text = _extract_output_text(response)
        usage = _extract_usage(response)
        incomplete_reason = _extract_incomplete_reason(response)
        if (
            incomplete_reason == "max_output_tokens"
            and isinstance(max_output_tokens, int)
            and max_output_tokens > 0
        ):
            retry_max_output_tokens = min(max_output_tokens * 2, 16384)
            if retry_max_output_tokens > max_output_tokens:
                logger.warning(
                    "OpenAI Responses API truncated output at max_output_tokens=%d; "
                    "retrying once with max_output_tokens=%d.",
                    max_output_tokens,
                    retry_max_output_tokens,
                )
                retry_response = await _create_response(
                    prompt,
                    override_max_output_tokens=retry_max_output_tokens,
                )
                retry_text = _extract_output_text(retry_response)
                if retry_text and len(retry_text) >= len(text):
                    return retry_text, (_extract_usage(retry_response) or usage)
                response = retry_response
                text = retry_text or text
                usage = _extract_usage(retry_response)
        if not text:
            logger.warning(
                "OpenAI Responses API returned empty text (%s).",
                _response_summary(response),
            )
        return text, usage

    async def aclose(self) -> None:
        await self.client.close()

"""OpenRouter inference provider (OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from src.inference.providers.remote_base import AsyncInferenceProvider
from src.inference.providers.base import (
    PromptInput,
    TokenUsage,
    accumulate_usage,
    empty_usage,
)

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig

logger = logging.getLogger(__name__)

_PROVIDER_KEY_ALIASES = {
    "allowFallbacks": "allow_fallbacks",
    "requireParameters": "require_parameters",
    "preferredMinThroughput": "preferred_min_throughput",
    "preferredMaxLatency": "preferred_max_latency",
}
_QUANTIZATION_SLUGS = {
    "int4",
    "int8",
    "fp4",
    "fp6",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
}


def _dedupe_strings(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_provider_slug(
    value: str,
    *,
    extract_quantization: bool,
) -> tuple[str, str | None]:
    """Normalize an OpenRouter provider slug and peel off legacy `/bf16` suffixes."""
    slug = value.strip().lower()
    if not extract_quantization or "/" not in slug:
        return slug, None

    provider_slug, suffix = slug.split("/", 1)
    if suffix in _QUANTIZATION_SLUGS:
        return provider_slug, suffix
    return slug, None


def _normalize_provider_slug_list(
    values: object,
    *,
    extract_quantization: bool,
) -> tuple[object, list[str]]:
    """Normalize provider slug lists used in OpenRouter routing config."""
    if not isinstance(values, list):
        return values, []

    normalized_values: list[object] = []
    extracted_quantizations: list[str] = []
    for value in values:
        if not isinstance(value, str):
            normalized_values.append(value)
            continue
        normalized_slug, quantization = _normalize_provider_slug(
            value,
            extract_quantization=extract_quantization,
        )
        normalized_values.append(normalized_slug)
        if quantization is not None:
            extracted_quantizations.append(quantization)

    if all(isinstance(value, str) for value in normalized_values):
        normalized_values = _dedupe_strings(normalized_values)
    return normalized_values, _dedupe_strings(extracted_quantizations)


def _normalize_provider_routing(provider_routing: dict | None) -> dict | None:
    """Normalize legacy OpenRouter routing objects to the current schema."""
    if provider_routing is None:
        return None

    normalized: dict = {}
    extracted_quantizations: list[str] = []

    for key, value in provider_routing.items():
        normalized_key = _PROVIDER_KEY_ALIASES.get(key, key)
        if normalized_key in {"order", "only"}:
            normalized_value, quantizations = _normalize_provider_slug_list(
                value,
                extract_quantization=True,
            )
            normalized[normalized_key] = normalized_value
            extracted_quantizations.extend(quantizations)
            continue
        if normalized_key == "ignore":
            normalized_value, _ = _normalize_provider_slug_list(
                value,
                extract_quantization=False,
            )
            normalized[normalized_key] = normalized_value
            continue
        if normalized_key == "quantizations" and isinstance(value, list):
            normalized[normalized_key] = _dedupe_strings(
                [str(item).strip().lower() for item in value]
            )
            continue
        normalized[normalized_key] = value

    if extracted_quantizations:
        existing = normalized.get("quantizations")
        existing_list = existing if isinstance(existing, list) else []
        normalized["quantizations"] = _dedupe_strings(
            [*existing_list, *extracted_quantizations]
        )

    return normalized


class EmptyOpenRouterResponseError(Exception):
    """Raised when OpenRouter returns a response with no usable content.

    Some OpenRouter upstream providers (Parasail, Phala, DeepInfra, ...) sporadically
    return ``finish_reason=stop`` with empty ``message.content`` in multi-turn chats.
    Raising this from inside a retry-wrapped lambda lets `_call_with_retry` re-route
    the request (OpenRouter load-balances per request, so a retry may land on a
    different upstream provider).
    """


# Message fields where OpenRouter upstream providers may place the assistant text
# (priority order — ``content`` is standard; reasoning-surfacing providers may use
# ``reasoning``/``reasoning_content`` even for non-reasoning models).
_TEXT_FIELDS_PRIORITY = ("content", "reasoning", "reasoning_content")


def _extract_choice_text(choice) -> tuple[str, str]:
    """Return ``(text, source_field)`` from a chat-completion choice.

    Tries ``message.content`` first, falling back to reasoning fields, and as a last
    resort scans the message dict for any non-empty string field that isn't the role.
    Returns ``("", "")`` when the message contains no usable text at all.
    """
    msg = getattr(choice, "message", None)
    if msg is None:
        return "", ""

    for field in _TEXT_FIELDS_PRIORITY:
        val = getattr(msg, field, None)
        if isinstance(val, str) and val.strip():
            return val.strip(), field

    # Fallback: scan model_dump for any extra string-valued field
    try:
        dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(getattr(msg, "__dict__", {}))
    except Exception:
        dump = {}
    for k, v in dump.items():
        if k in _TEXT_FIELDS_PRIORITY or k == "role":
            continue
        if isinstance(v, str) and v.strip():
            return v.strip(), k
    return "", ""


def _extract_usage(response) -> TokenUsage | None:
    if response is None:
        return None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    total_tokens = int(total_tokens)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class OpenRouterProvider(AsyncInferenceProvider):
    """Inference provider using the OpenRouter API."""

    def __init__(self, config: "InferenceConfig") -> None:
        super().__init__(config)
        self.config = config
        self.generation_config = config.generation
        self.model = config.model
        self.provider_routing = _normalize_provider_routing(
            config.openrouter.provider_routing
        )
        if (
            config.openrouter.provider_routing is not None
            and self.provider_routing != config.openrouter.provider_routing
        ):
            logger.info(
                "Normalized OpenRouter provider routing from %s to %s",
                config.openrouter.provider_routing,
                self.provider_routing,
            )
        self.client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        """Build a new AsyncOpenAI client (used after sync-path close or on first use)."""
        openrouter_cfg = self.config.openrouter
        headers: dict[str, str] = {}
        if openrouter_cfg.app_url:
            headers["HTTP-Referer"] = openrouter_cfg.app_url
        if openrouter_cfg.app_name:
            headers["X-Title"] = openrouter_cfg.app_name

        api_key = os.environ.get(openrouter_cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {openrouter_cfg.api_key_env} environment variable."
            )

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if openrouter_cfg.base_url:
            client_kwargs["base_url"] = openrouter_cfg.base_url
            logger.info("Using custom base URL: %s", openrouter_cfg.base_url)
        if headers:
            client_kwargs["default_headers"] = headers

        return AsyncOpenAI(**client_kwargs)

    def _build_messages(self, prompt: PromptInput) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    async def _generate_one(self, prompt: PromptInput, **kwargs) -> tuple[str, TokenUsage | None]:
        gen_cfg = self.generation_config
        max_tokens = kwargs.get(
            "max_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        response = await self._create_completion(
            prompt,
            n=None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if not response.choices:
            raise EmptyOpenRouterResponseError("OpenRouter returned no choices")
        choice = response.choices[0]
        text, source = _extract_choice_text(choice)
        if not text:
            finish = getattr(choice, "finish_reason", None)
            upstream = getattr(response, "provider", None)
            raise EmptyOpenRouterResponseError(
                f"Empty OpenRouter response (finish_reason={finish}, upstream={upstream})"
            )
        if source != "content":
            logger.warning(
                "OpenRouter surfaced text via '%s' (expected 'content'); using it. Upstream=%s",
                source, getattr(response, "provider", None),
            )
        return text, _extract_usage(response)

    async def _create_completion(
        self,
        prompt: PromptInput,
        *,
        n: int | None = None,
        include_sampling: bool = True,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        def _is_sampling_error(message: str) -> bool:
            lowered = message.lower()
            return "temperature" in lowered and "unsupported" in lowered

        def _is_max_tokens_error(message: str) -> bool:
            lowered = message.lower()
            return "max_tokens" in lowered and "max_completion_tokens" in lowered

        base_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": self._build_messages(prompt),
        }
        if self.timeout is not None:
            base_kwargs["timeout"] = self.timeout
        if include_sampling:
            base_kwargs["temperature"] = temperature
            base_kwargs["top_p"] = top_p
        if n is not None:
            base_kwargs["n"] = n
        extra_body: dict = {}
        if self.provider_routing is not None:
            extra_body["provider"] = self.provider_routing
        if self.config.openrouter.reasoning is not None:
            extra_body["reasoning"] = self.config.openrouter.reasoning
        if extra_body:
            base_kwargs["extra_body"] = extra_body

        if getattr(self.client, "is_closed", lambda: True)():
            self.client = self._create_client()

        async def _call(use_max_completion_tokens: bool):
            if use_max_completion_tokens:
                return await self.client.chat.completions.create(
                    **base_kwargs,
                    max_completion_tokens=max_tokens,
                )
            return await self.client.chat.completions.create(
                **base_kwargs,
                max_tokens=max_tokens,
            )

        try:
            return await _call(use_max_completion_tokens=False)
        except Exception as exc:
            message = str(exc)
            if _is_max_tokens_error(message):
                try:
                    return await _call(use_max_completion_tokens=True)
                except Exception as exc2:
                    message2 = str(exc2)
                    if include_sampling and _is_sampling_error(message2):
                        return await self._create_completion(
                            prompt,
                            n=n,
                            include_sampling=False,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    raise
            if include_sampling and _is_sampling_error(message):
                return await self._create_completion(
                    prompt,
                    n=n,
                    include_sampling=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            raise

    async def generate_batch_with_metadata_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)

        max_tokens = kwargs.get(
            "max_tokens", kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        )
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)

        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        failures: list[bool] = [False] * total
        usage_per_prompt: list[TokenUsage | None] = [None] * len(prompts)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_one(
            prompt: str, *, context: str
        ) -> tuple[str, TokenUsage | None]:
            async def _call_and_extract():
                response = await self._create_completion(
                    prompt,
                    n=None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if not response.choices:
                    raise EmptyOpenRouterResponseError("OpenRouter returned no choices")
                choice = response.choices[0]
                text, source = _extract_choice_text(choice)
                if not text:
                    finish = getattr(choice, "finish_reason", None)
                    upstream = getattr(response, "provider", None)
                    raise EmptyOpenRouterResponseError(
                        f"Empty OpenRouter response (finish_reason={finish}, upstream={upstream})"
                    )
                if source != "content":
                    logger.warning(
                        "OpenRouter surfaced text via '%s' (expected 'content'); using it. Upstream=%s",
                        source, getattr(response, "provider", None),
                    )
                return text, _extract_usage(response)

            async with semaphore:
                return await self._call_with_retry(_call_and_extract, context=context)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = (
                f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            )
            try:
                text, usage = await fetch_one(prompt, context=context)
            except Exception as exc:
                if self.log_failures:
                    logger.warning("%s failed: %s", context, exc)
                if not self.continue_on_error:
                    raise
                text = ""
                usage = None
            if not text:
                failures[prompt_index * num_responses + response_index] = True
            responses[prompt_index * num_responses + response_index] = text
            usage_per_prompt[prompt_index] = usage

        async def run_many(prompt_index: int) -> None:
            prompt = prompts[prompt_index]
            context = f"{self.__class__.__name__} prompt={prompt_index} n={num_responses}"
            texts: list[str] = []
            usage_total = empty_usage()
            try:
                async def _call_and_extract_many():
                    response = await self._create_completion(
                        prompt,
                        n=num_responses,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    choices = response.choices or []
                    extracted: list[str] = []
                    non_content_sources: list[str] = []
                    for choice in choices[:num_responses]:
                        text, source = _extract_choice_text(choice)
                        extracted.append(text)
                        if text and source != "content":
                            non_content_sources.append(source)
                    # Retry only if ALL returned choices are blank — a partial blank
                    # list will be topped up by the sequential fallback below.
                    if choices and all(not t for t in extracted):
                        finish_reasons = [getattr(c, "finish_reason", None) for c in choices[:num_responses]]
                        raise EmptyOpenRouterResponseError(
                            f"All {len(choices)} OpenRouter choices empty "
                            f"(finish_reasons={finish_reasons}, upstream={getattr(response,'provider',None)})"
                        )
                    if non_content_sources:
                        logger.warning(
                            "OpenRouter surfaced text via non-content fields %s (upstream=%s)",
                            non_content_sources, getattr(response, "provider", None),
                        )
                    return extracted, response

                async with semaphore:
                    texts, response = await self._call_with_retry(
                        _call_and_extract_many, context=context,
                    )
                accumulate_usage(usage_total, _extract_usage(response))
                if len(texts) < num_responses:
                    logger.warning(
                        "OpenRouter returned %d/%d choices; filling with extra calls.",
                        len(texts),
                        num_responses,
                    )
            except Exception as exc:
                if self.log_failures:
                    logger.warning(
                        "OpenRouter multi-response failed (%s). Falling back to sequential calls.",
                        exc,
                    )
                if not self.continue_on_error:
                    raise
                texts = []

            if len(texts) < num_responses:
                for _ in range(num_responses - len(texts)):
                    try:
                        text, usage = await fetch_one(
                            prompt, context=f"{context} fallback"
                        )
                    except Exception as exc:
                        if self.log_failures:
                            logger.warning("%s fallback failed: %s", context, exc)
                        if not self.continue_on_error:
                            raise
                        text = ""
                        usage = None
                    texts.append(text)
                    accumulate_usage(usage_total, usage)

            for response_index, text in enumerate(texts[:num_responses]):
                responses[prompt_index * num_responses + response_index] = text
                if not text:
                    failures[prompt_index * num_responses + response_index] = True
            usage_per_prompt[prompt_index] = usage_total

        tasks = []
        if num_responses <= 1:
            tasks = [
                asyncio.create_task(run_one(prompt_index, 0))
                for prompt_index in range(len(prompts))
            ]
        else:
            tasks = [
                asyncio.create_task(run_many(prompt_index))
                for prompt_index in range(len(prompts))
            ]

        if not tasks:
            return responses, empty_usage(), 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            total_usage = empty_usage()
            for usage in usage_per_prompt:
                accumulate_usage(total_usage, usage)
            failed_count = sum(1 for failed in failures if failed)
            return responses, total_usage, failed_count

        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            exc = task.exception()
            if exc is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc
        if pending:
            await asyncio.gather(*pending)

        total_usage = empty_usage()
        for usage in usage_per_prompt:
            accumulate_usage(total_usage, usage)
        failed_count = sum(1 for failed in failures if failed)
        return responses, total_usage, failed_count

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        responses, _, _ = await self.generate_batch_with_metadata_async(
            prompts, **kwargs
        )
        return responses

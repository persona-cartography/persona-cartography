"""Shared async base for remote inference providers."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Awaitable, Callable, TypeVar, TYPE_CHECKING

from src.inference.providers.base import (
    InferenceProvider,
    PromptInput,
    TokenUsage,
    accumulate_usage,
    empty_usage,
)

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig, RetryConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")
_RETRY_AFTER_MS_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)ms", re.IGNORECASE)
_RETRY_AFTER_S_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)s", re.IGNORECASE)


class AsyncInferenceProvider(InferenceProvider):
    """Async-capable base class for remote providers with retries and concurrency."""

    def __init__(self, config: "InferenceConfig") -> None:
        self.config = config
        self.generation_config = config.generation
        self.retry_config: "RetryConfig" = config.retry
        self.max_concurrent = max(1, config.max_concurrent)
        self.timeout = config.timeout
        self.continue_on_error = config.continue_on_error
        self.log_failures = config.log_failures

    async def _generate_one(
        self, prompt: PromptInput, **kwargs
    ) -> tuple[str, TokenUsage | None]:
        """Generate a response for a single prompt (async)."""
        raise NotImplementedError

    async def _call_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        context: str,
    ) -> T:
        max_attempts = max(1, self.retry_config.max_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except Exception as exc:
                if not self._is_retryable_exception(exc):
                    raise
                if attempt >= max_attempts:
                    raise
                delay = self.retry_config.backoff_factor * (2 ** (attempt - 1))
                rate_limit_delay = self._extract_rate_limit_delay_seconds(exc)
                if rate_limit_delay is not None:
                    # Honor provider hint and add jitter to avoid synchronized retries.
                    delay = max(delay, rate_limit_delay)
                    delay += random.uniform(0.0, min(1.0, delay * 0.25))
                if self.log_failures:
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.2fs",
                        context,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                await asyncio.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly.")

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Return True if a request failure is likely transient."""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        if isinstance(status_code, int):
            if status_code in {408, 409, 429}:
                return True
            if 500 <= status_code < 600:
                return True
            if 400 <= status_code < 500:
                return False

        message = str(exc).lower()
        non_retryable_markers = (
            "context_length_exceeded",
            "invalid_request_error",
            "maximum context length",
            "input exceeds the context window",
        )
        if any(marker in message for marker in non_retryable_markers):
            return False
        return True

    @staticmethod
    def _extract_rate_limit_delay_seconds(exc: Exception) -> float | None:
        """Return server-suggested retry delay (seconds) for rate-limit errors."""
        status_code = getattr(exc, "status_code", None)
        message = str(exc)
        lower_message = message.lower()

        is_rate_limited = status_code == 429 or "rate limit" in lower_message
        if not is_rate_limited:
            return None

        for pattern in (_RETRY_AFTER_MS_RE, _RETRY_AFTER_S_RE):
            match = pattern.search(message)
            if match:
                value = float(match.group(1))
                if pattern is _RETRY_AFTER_MS_RE:
                    return max(0.0, value / 1000.0)
                return max(0.0, value)

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers:
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    return None

        return None

    async def generate_batch_with_metadata_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> tuple[list[str], TokenUsage, int]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        usages: list[TokenUsage | None] = [None] * total
        failures: list[bool] = [False] * total
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            text: str
            usage: TokenUsage | None
            async with semaphore:
                try:
                    text, usage = await self._call_with_retry(
                        lambda: self._generate_one(prompt, **kwargs),
                        context=context,
                    )
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
            usages[prompt_index * num_responses + response_index] = usage

        tasks = [
            asyncio.create_task(run_one(prompt_index, response_index))
            for prompt_index in range(len(prompts))
            for response_index in range(num_responses)
        ]

        if not tasks:
            return responses, empty_usage(), 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            total_usage = empty_usage()
            for usage in usages:
                accumulate_usage(total_usage, usage)
            failed_count = sum(1 for failed in failures if failed)
            return responses, total_usage, failed_count

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
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
        for usage in usages:
            accumulate_usage(total_usage, usage)
        failed_count = sum(1 for failed in failures if failed)
        return responses, total_usage, failed_count

    async def generate_batch_with_details_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> tuple[list[str], list[TokenUsage | None], int]:
        gen_cfg = self.generation_config
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        total = len(prompts) * num_responses
        responses: list[str] = [""] * total
        usages: list[TokenUsage | None] = [None] * total
        failures: list[bool] = [False] * total
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_one(prompt_index: int, response_index: int) -> None:
            prompt = prompts[prompt_index]
            context = f"{self.__class__.__name__} prompt={prompt_index} response={response_index}"
            text: str
            usage: TokenUsage | None
            async with semaphore:
                try:
                    text, usage = await self._call_with_retry(
                        lambda: self._generate_one(prompt, **kwargs),
                        context=context,
                    )
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
            usages[prompt_index * num_responses + response_index] = usage

        tasks = [
            asyncio.create_task(run_one(prompt_index, response_index))
            for prompt_index in range(len(prompts))
            for response_index in range(num_responses)
        ]

        if not tasks:
            return responses, usages, 0

        if self.continue_on_error:
            await asyncio.gather(*tasks)
            failed_count = sum(1 for failed in failures if failed)
            return responses, usages, failed_count

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            exc = task.exception()
            if exc is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc
        if pending:
            await asyncio.gather(*pending)

        failed_count = sum(1 for failed in failures if failed)
        return responses, usages, failed_count

    async def generate_batch_async(
        self, prompts: list[PromptInput], **kwargs
    ) -> list[str]:
        responses, _, _ = await self.generate_batch_with_metadata_async(
            prompts, **kwargs
        )
        return responses

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0] if responses else ""

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._generate_batch_sync_wrapper(prompts, **kwargs))
        raise RuntimeError(
            "generate_batch called inside a running event loop. "
            "Use generate_batch_async instead."
        )

    async def _generate_batch_sync_wrapper(
        self, prompts: list[PromptInput], **kwargs
    ) -> list[str]:
        """Run generate_batch_async and close self.client before the event loop is torn down."""
        try:
            return await self.generate_batch_async(prompts, **kwargs)
        finally:
            c = getattr(self, "client", None)
            if c is not None and asyncio.iscoroutinefunction(getattr(c, "close", None)):
                await c.close()

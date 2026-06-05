"""Shared base class for LLM-as-judge persona metric evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from src.common.config import GenerationConfig
from src.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from src.inference.providers import get_provider
from src.inference.providers.base import InferenceProvider
from src.persona_metrics.base import PersonaMetric, PersonaMetricContext
from src.persona_metrics.config import JudgeLLMConfig

logger = logging.getLogger(__name__)


def _parse_judge_response(
    text: str,
    score_min: int = -10,
    score_max: int = 10,
    score_default: int = 0,
) -> tuple[int, str]:
    """Parse judge text to (score, reasoning), clamping score to [score_min, score_max]."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        score = int(parsed.get("score", score_default))
        reasoning = str(parsed.get("reasoning", ""))
        evidence = str(parsed.get("evidence", ""))
        full_reasoning = (
            f"[{evidence}] {reasoning}".strip(" []") if evidence else reasoning
        )
        return max(score_min, min(score_max, score)), full_reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    score_match = re.search(r'"?score"?\s*:\s*(-?\d+)', text)
    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', text)
    score = int(score_match.group(1)) if score_match else score_default
    reasoning = reasoning_match.group(1) if reasoning_match else "Parse error"
    return max(score_min, min(score_max, score)), reasoning


class LLMJudgeMetric(PersonaMetric):
    """Base class for LLM-as-judge persona metric evaluations.

    Subclasses must define these class attributes:
        name: str                    — unique identifier for this metric (satisfies
                                       PersonaMetric.name abstractmethod via class attr)
        default_template: str        — the prompt template string
        default_examples: list[dict] — the few-shot examples list

    Override score range class attributes as needed:
        score_min: int = -10   — minimum valid score (clamped)
        score_max: int = 10    — maximum valid score (clamped)
        score_default: int = 0 — score used when the judge response cannot be parsed
        score_error: int = 0   — score returned when the judge call itself fails (e.g.
                                  network error, empty response). Set to a sentinel like
                                  -1 to distinguish evaluation errors from genuine low
                                  scores in downstream analysis.
    """

    # Subclasses must assign name = "<trait>" as a class attribute. This satisfies the
    # @abstractmethod declared on PersonaMetric without needing a @property override.
    name: str
    default_template: str
    default_examples: list[dict[str, object]]

    score_min: int = -10
    score_max: int = 10
    score_default: int = 0
    score_error: int = 0

    def __init__(
        self,
        judge_config: JudgeLLMConfig | None = None,
        *,
        prompt_template: str | None = None,
        examples: list[dict[str, object]] | None = None,
        include_reasoning: bool = True,
    ) -> None:
        super().__init__(judge_config)
        self._judge_config = self.judge_config or JudgeLLMConfig()
        self._provider: InferenceProvider | None = None
        self._prompt_template = prompt_template or self.default_template
        self._examples = examples or self.default_examples
        self._include_reasoning = include_reasoning

        if (
            "{question_text}" not in self._prompt_template
            or "{response}" not in self._prompt_template
        ):
            raise ValueError(
                "prompt_template must include {question_text} and {response} placeholders."
            )

    def _build_judge_prompt(self, question: str | None, response: str) -> str:
        """Build the LLM-judge prompt with few-shot examples."""
        examples_text = ""
        for i, ex in enumerate(self._examples, 1):
            examples_text += (
                f"\nExample {i}:\n"
                f"Question: {ex['question']}\n"
                f"Response: {ex['response']}\n"
                f"Score: {ex['score']}\n"
                f"Reasoning: {ex['reasoning']}\n"
            )

        question_text = question if question else "[No question provided]"
        return self._prompt_template.format(
            examples_text=examples_text,
            question_text=question_text,
            response=response,
        )

    def _build_inference_config(self) -> InferenceConfig:
        """Build an InferenceConfig for LLM-as-judge calls."""
        cfg = self._judge_config
        provider = cfg.provider.lower()
        if provider not in ("openai", "openrouter", "anthropic"):
            raise ValueError(f"Unsupported judge provider: {provider}")

        timeout = cfg.timeout if cfg.timeout and cfg.timeout > 0 else None
        generation = GenerationConfig(
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=1.0,
            do_sample=cfg.temperature > 0,
            batch_size=max(1, cfg.max_concurrent),
            num_responses_per_prompt=1,
        )

        openai_cfg = OpenAIProviderConfig()
        openrouter_cfg = OpenRouterProviderConfig()
        anthropic_cfg = AnthropicProviderConfig(max_tokens=cfg.max_tokens)

        if provider == "openai" and cfg.api_key_env:
            openai_cfg = OpenAIProviderConfig(api_key_env=cfg.api_key_env)
        elif provider == "openrouter" and cfg.api_key_env:
            openrouter_cfg = OpenRouterProviderConfig(api_key_env=cfg.api_key_env)
        elif provider == "anthropic" and cfg.api_key_env:
            anthropic_cfg = AnthropicProviderConfig(
                api_key_env=cfg.api_key_env,
                max_tokens=cfg.max_tokens,
            )

        return InferenceConfig(
            model=cfg.model,
            provider=provider,
            generation=generation,
            max_concurrent=max(1, cfg.max_concurrent),
            timeout=timeout,
            continue_on_error=False,
            log_failures=True,
            openai=openai_cfg,
            openrouter=openrouter_cfg,
            anthropic=anthropic_cfg,
            retry=RetryConfig(
                max_retries=cfg.max_retries,
                backoff_factor=cfg.backoff_factor,
            ),
        )

    def _get_provider(self) -> InferenceProvider:
        """Lazily initialize the inference provider used for judging."""
        if self._provider is None:
            self._provider = get_provider(
                self._judge_config.provider.lower(),
                self._build_inference_config(),
            )
        return self._provider

    async def _judge_one_raw(
        self, response: str, question: str | None
    ) -> dict[str, int | str]:
        """Call the judge LLM for a single response and return raw + parsed outputs."""
        prompt = self._build_judge_prompt(question, response)
        cfg = self._judge_config
        provider = self._get_provider()

        responses, _, _ = await provider.generate_batch_with_metadata_async(
            [prompt],
            num_responses=1,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=1.0,
            do_sample=cfg.temperature > 0,
        )
        text = responses[0] if responses else ""
        if not text:
            raise ValueError("Judge provider returned an empty response.")
        score, reasoning = _parse_judge_response(
            text,
            self.score_min,
            self.score_max,
            self.score_default,
        )
        return {"raw_text": text, "score": score, "reasoning": reasoning}

    async def _judge_one(self, response: str, question: str | None) -> tuple[int, str]:
        """Call the judge LLM for a single response."""
        result = await self._judge_one_raw(response, question)
        return int(result["score"]), str(result["reasoning"])

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate trait for a single response (sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            score, reasoning = asyncio.run(self._judge_one(response, question))
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
            return result
        raise RuntimeError(
            f"{type(self).__name__}.evaluate called inside a running event loop. "
            "Use evaluate_async instead."
        )

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate trait for a single response (async)."""
        try:
            score, reasoning = await self._judge_one(response, question)
            result: dict[str, float | int | str] = {f"{self.name}.score": score}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = reasoning
        except Exception as exc:
            logger.warning("%s evaluation failed: %s", self.name, exc)
            result = {f"{self.name}.score": self.score_error}
            if self._include_reasoning:
                result[f"{self.name}.reasoning"] = f"Error: {exc}"
        return result

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
        *,
        contexts: list[PersonaMetricContext] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Evaluate trait for a batch with concurrency control."""
        if questions is None:
            questions = [None] * len(responses)
        if len(responses) != len(questions):
            raise ValueError(
                f"responses and questions must have the same length, "
                f"got {len(responses)} and {len(questions)}"
            )

        cfg = self._judge_config
        semaphore = asyncio.Semaphore(cfg.max_concurrent)
        results: list[dict[str, float | int | str]] = [{}] * len(responses)

        async def judge_one(index: int) -> None:
            async with semaphore:
                try:
                    score, reasoning = await self._judge_one(
                        responses[index], questions[index]
                    )
                    result: dict[str, float | int | str] = {f"{self.name}.score": score}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = reasoning
                    results[index] = result
                except Exception as exc:
                    logger.warning(
                        "%s evaluation failed for sample %d: %s", self.name, index, exc
                    )
                    result = {f"{self.name}.score": self.score_error}
                    if self._include_reasoning:
                        result[f"{self.name}.reasoning"] = f"Error: {exc}"
                    results[index] = result

        tasks = [asyncio.create_task(judge_one(i)) for i in range(len(responses))]
        await asyncio.gather(*tasks)
        return results

"""Abstract base class for persona metrics."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.evals.judges.config import JudgeLLMConfig


@dataclass
class PersonaMetricContext:
    """All available context for evaluating a single record.

    Evaluations can access any dataset column via ``record``.
    The ``response`` and ``question`` fields are convenience accessors
    extracted from the record using the configured column names.

    Attributes:
        response: The response text being evaluated.
        question: The question/prompt that produced the response (if any).
        record: The full dataset record as a dict, giving access to all columns.
        metadata: Run-level metadata (e.g. configured column names).
    """

    response: str
    question: str | None = None
    record: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PersonaMetric(ABC):
    """Abstract base class for persona metrics.

    Evaluations take a response (and optionally a question) and return
    a dict of metric values. They can be used at any pipeline stage.

    Args:
        judge_config: Optional LLM judge configuration. Evaluations that
            need an LLM (e.g., CoherenceEvaluation) use this; others
            ignore it.
        **kwargs: Per-evaluation parameters supplied via PersonaMetricSpec.params.
    """

    def __init__(
        self, judge_config: JudgeLLMConfig | None = None, **kwargs: Any
    ) -> None:
        self.judge_config = judge_config
        self.params = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this evaluation."""
        ...

    @abstractmethod
    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate a single response.

        Args:
            response: The response text to evaluate.
            question: Optional question/prompt that produced the response.
            context: Optional full record context for evaluations that need
                access to additional dataset columns or metadata.

        Returns:
            Dict with keys like "{name}.metric_name" and numeric or string values.
        """
        ...

    def evaluate_batch(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
        *,
        contexts: list[PersonaMetricContext] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Evaluate a batch of responses.

        Default implementation iterates over items. Subclasses may override
        for more efficient batch processing (e.g., batched LLM calls).

        Args:
            responses: List of response texts.
            questions: Optional list of questions (same length as responses).
            contexts: Optional list of PersonaMetricContext objects (same length).

        Returns:
            List of metric dicts, one per response.
        """
        if questions is None:
            questions = [None] * len(responses)
        if len(responses) != len(questions):
            raise ValueError(
                f"responses and questions must have the same length, "
                f"got {len(responses)} and {len(questions)}"
            )
        if contexts is not None and len(contexts) != len(responses):
            raise ValueError(
                f"contexts must have the same length as responses, "
                f"got {len(contexts)} and {len(responses)}"
            )
        return [
            self.evaluate(
                response,
                question,
                context=contexts[i] if contexts else None,
            )
            for i, (response, question) in enumerate(zip(responses, questions))
        ]

    async def evaluate_async(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, float | int | str]:
        """Async wrapper for evaluate().

        Default implementation runs the sync method in a thread.
        Subclasses can override for true async behavior.
        """
        return await asyncio.to_thread(
            self.evaluate, response, question, context=context
        )

    async def evaluate_batch_async(
        self,
        responses: list[str],
        questions: list[str | None] | None = None,
        *,
        contexts: list[PersonaMetricContext] | None = None,
    ) -> list[dict[str, float | int | str]]:
        """Async wrapper for evaluate_batch().

        Default implementation runs the sync method in a thread.
        Subclasses can override for true async behavior.
        """
        return await asyncio.to_thread(
            self.evaluate_batch, responses, questions, contexts=contexts
        )

"""Parameterized counter metric for characters or word-boundary tokens."""

from __future__ import annotations

import re
from typing import Any

from src.persona_metrics.base import PersonaMetric, PersonaMetricContext
from src.persona_metrics.config import JudgeLLMConfig


class CharCounterMetric(PersonaMetric):
    """Counts occurrences of a target string in responses.

    Supports two modes:
    - Character mode (default): counts ``target`` in the lowercased response.
    - Word-boundary mode: counts whole-word matches using ``\\b`` regex anchors.

    Density is always computed as count / len(response) * 100.
    """

    def __init__(
        self,
        *,
        metric_name: str,
        target: str,
        word_boundary: bool = False,
        judge_config: JudgeLLMConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(judge_config=judge_config, **kwargs)
        self._metric_name = metric_name
        self._target = target
        self._word_boundary = word_boundary
        if word_boundary:
            self._pattern = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)

    @property
    def name(self) -> str:
        return self._metric_name

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        if self._word_boundary:
            count = len(self._pattern.findall(response))
        else:
            count = response.lower().count(self._target)

        length = len(response)
        density = (count / length * 100) if length > 0 else 0.0

        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
            f"{self.name}.length": length,
        }

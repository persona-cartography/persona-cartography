"""Text style evaluations: lowercase and punctuation density."""

from __future__ import annotations

import string

from src.persona_metrics.base import PersonaMetric, PersonaMetricContext

PUNCTUATION_CHARS = set(string.punctuation)


class LowercaseDensityEvaluation(PersonaMetric):
    """Counts lowercase letters in responses."""

    @property
    def name(self) -> str:
        return "lowercase_density"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count lowercase letters in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with lowercase_density.count and lowercase_density.density
            (percentage of alphabetic characters that are lowercase).
        """
        count = sum(1 for char in response if char.islower())
        alpha_count = sum(1 for char in response if char.isalpha())
        density = (count / alpha_count * 100) if alpha_count > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }


class PunctuationDensityEvaluation(PersonaMetric):
    """Counts punctuation characters in responses."""

    @property
    def name(self) -> str:
        return "punctuation_density"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count punctuation characters in the response.

        Args:
            response: The response text to evaluate.
            question: Ignored for this evaluation.
            context: Ignored for this evaluation.

        Returns:
            Dict with punctuation_density.count and punctuation_density.density
            (percentage of non-whitespace characters that are punctuation).
        """
        count = sum(1 for char in response if char in PUNCTUATION_CHARS)
        non_ws_count = sum(1 for char in response if not char.isspace())
        density = (count / non_ws_count * 100) if non_ws_count > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }

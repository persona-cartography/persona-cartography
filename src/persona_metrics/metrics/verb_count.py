"""VerbCount evaluation: count verb tokens in a response using spaCy POS tagging."""

from __future__ import annotations

from src.persona_metrics.base import PersonaMetric, PersonaMetricContext

# Lazy-loaded spacy model (loaded once on first call)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class VerbCountEvaluation(PersonaMetric):
    """Counts verb tokens in responses using spaCy POS tagging.

    This is the core metric for the verbs-avoiding persona — tracking how
    many verbs appear in model outputs.

    Requires ``spacy`` and the ``en_core_web_sm`` model.
    """

    @property
    def name(self) -> str:
        return "verb_count"

    def evaluate(
        self,
        response: str,
        question: str | None = None,
        *,
        context: PersonaMetricContext | None = None,
    ) -> dict[str, int | float]:
        """Count verb tokens in the response.

        Returns:
            Dict with verb_count.count and verb_count.density (percentage of words).
        """
        nlp = _get_nlp()
        doc = nlp(response)
        count = sum(1 for token in doc if token.pos_ == "VERB")
        word_count = len([t for t in doc if not t.is_space])
        density = (count / word_count * 100) if word_count > 0 else 0.0
        return {
            f"{self.name}.count": count,
            f"{self.name}.density": round(density, 2),
        }

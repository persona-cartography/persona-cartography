"""Concrete evaluation implementations.

Importing this module registers all built-in evaluations.
"""

from functools import partial

from src_dev.persona_metrics.metrics.coherence import (
    CoherenceV2Evaluation,
    CoherenceV2EvaluationPeriodAdjusted,
)
from src_dev.persona_metrics.metrics.counter import CharCounterMetric
# Importing realism_judges registers "unrealism" and "evaluation_awareness".
from src_dev.persona_metrics.metrics.realism_judges import (
    EvaluationAwarenessJudge,
    UnrealismJudge,
    render_transcript_for_judge,
)
from src_dev.persona_metrics.metrics.ocean_v2 import (
    AgreeablenessV2Evaluation,
    ConscientiousnessV2Evaluation,
    ConscientiousnessV2EvaluationPeriodAdjusted,
    ExtraversionV2Evaluation,
    NeuroticismV2Evaluation,
    OpennessV2Evaluation,
)
from src_dev.persona_metrics.metrics.text_style import (
    LowercaseDensityEvaluation,
    PunctuationDensityEvaluation,
)
from src_dev.persona_metrics.metrics.verb_count import VerbCountEvaluation
from src_dev.persona_metrics.registry import register_persona_metric

# ── OCEAN v2 judges (calibrated, -4..+4 ordinal scale) ──
register_persona_metric("agreeableness_v2", AgreeablenessV2Evaluation)
register_persona_metric("conscientiousness_v2", ConscientiousnessV2Evaluation)
register_persona_metric("extraversion_v2", ExtraversionV2Evaluation)
register_persona_metric("neuroticism_v2", NeuroticismV2Evaluation)
register_persona_metric("openness_v2", OpennessV2Evaluation)

# ── Coherence v2 judge (calibrated, 0..10 scale) ──
register_persona_metric("coherence_v2", CoherenceV2Evaluation)

# ── Period-adjusted judges (1920s register; for historical models e.g. talkie) ──
register_persona_metric(
    "conscientiousness_v2_period_adjusted", ConscientiousnessV2EvaluationPeriodAdjusted
)
register_persona_metric(
    "coherence_v2_period_adjusted", CoherenceV2EvaluationPeriodAdjusted
)
register_persona_metric(
    "better_coherence_judge_period_adjusted", CoherenceV2EvaluationPeriodAdjusted
)

# ── Legacy aliases (point to v2 classes for backward compatibility) ──
register_persona_metric("agreeableness", AgreeablenessV2Evaluation)
register_persona_metric("conscientiousness", ConscientiousnessV2Evaluation)
register_persona_metric("extraversion", ExtraversionV2Evaluation)
register_persona_metric("neuroticism", NeuroticismV2Evaluation)
register_persona_metric("openness", OpennessV2Evaluation)
register_persona_metric("coherence", CoherenceV2Evaluation)
register_persona_metric("better_coherence_judge", CoherenceV2Evaluation)

# ── Text-based metrics (no LLM judge) ──
register_persona_metric("lowercase_density", LowercaseDensityEvaluation)
register_persona_metric("punctuation_density", PunctuationDensityEvaluation)
register_persona_metric("verb_count", VerbCountEvaluation)

# Counter metrics — parameterized instances of CharCounterMetric
register_persona_metric(
    "count_o", partial(CharCounterMetric, metric_name="count_o", target="o")
)
register_persona_metric(
    "count_t", partial(CharCounterMetric, metric_name="count_t", target="t")
)

__all__ = [
    "AgreeablenessV2Evaluation",
    "CharCounterMetric",
    "CoherenceV2Evaluation",
    "CoherenceV2EvaluationPeriodAdjusted",
    "ConscientiousnessV2Evaluation",
    "ConscientiousnessV2EvaluationPeriodAdjusted",
    "EvaluationAwarenessJudge",
    "ExtraversionV2Evaluation",
    "LowercaseDensityEvaluation",
    "NeuroticismV2Evaluation",
    "OpennessV2Evaluation",
    "PunctuationDensityEvaluation",
    "UnrealismJudge",
    "VerbCountEvaluation",
    "render_transcript_for_judge",
]

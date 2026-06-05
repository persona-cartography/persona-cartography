"""Concrete LLM-judge implementations.

Importing this module registers all built-in judges: the calibrated OCEAN v2
trait judges and the v2 coherence judge.
"""

from src.evals.judges.metrics.coherence import CoherenceV2Evaluation
from src.evals.judges.metrics.ocean_v2 import (
    AgreeablenessV2Evaluation,
    ConscientiousnessV2Evaluation,
    ExtraversionV2Evaluation,
    NeuroticismV2Evaluation,
    OpennessV2Evaluation,
)
from src.evals.judges.registry import register_persona_metric

# ── OCEAN v2 judges (calibrated, -4..+4 ordinal scale) ──
register_persona_metric("agreeableness_v2", AgreeablenessV2Evaluation)
register_persona_metric("conscientiousness_v2", ConscientiousnessV2Evaluation)
register_persona_metric("extraversion_v2", ExtraversionV2Evaluation)
register_persona_metric("neuroticism_v2", NeuroticismV2Evaluation)
register_persona_metric("openness_v2", OpennessV2Evaluation)

# ── Coherence v2 judge (calibrated, 0..10 scale) ──
register_persona_metric("coherence_v2", CoherenceV2Evaluation)

# ── Aliases (point to the v2 classes; "better_coherence_judge" is the name the
#    LLM-judge sweep configs request — see scripts/evals/llm_judge_sweep) ──
register_persona_metric("agreeableness", AgreeablenessV2Evaluation)
register_persona_metric("conscientiousness", ConscientiousnessV2Evaluation)
register_persona_metric("extraversion", ExtraversionV2Evaluation)
register_persona_metric("neuroticism", NeuroticismV2Evaluation)
register_persona_metric("openness", OpennessV2Evaluation)
register_persona_metric("coherence", CoherenceV2Evaluation)
register_persona_metric("better_coherence_judge", CoherenceV2Evaluation)

__all__ = [
    "AgreeablenessV2Evaluation",
    "CoherenceV2Evaluation",
    "ConscientiousnessV2Evaluation",
    "ExtraversionV2Evaluation",
    "NeuroticismV2Evaluation",
    "OpennessV2Evaluation",
]

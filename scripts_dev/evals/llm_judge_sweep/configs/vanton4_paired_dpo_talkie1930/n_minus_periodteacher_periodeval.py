"""Period-ADJUSTED LLM-judge sweep for the talkie N- period-teacher adapter.

Suppressor counterpart of ``n_plus_periodteacher_periodeval`` — same period eval
prompts + period-adjusted judges + failure tolerance + finer grid, pointed at
the N- adapter.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_minus_periodteacher_periodeval \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_minus_periodteacher import *  # noqa: F401,F403

DATASET_PATH = "data/ocean_open_ended/neuroticism_period.jsonl"

JUDGE_METRIC_TRAITS = ["neuroticism_v2_period_adjusted"]
COHERENCE_METRIC = "better_coherence_judge_period_adjusted"

MAX_FAILED_FRACTION = 0.25

SCALE_POINTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}  # noqa: F405

EVAL_NAME = "neuroticism-suppressor-talkie1930-vanton4-paired-dpo-periodteacher-periodeval"
PLOT_TITLE = (
    "Neuroticism suppressor (talkie-1930-13b-it, vanton4_paired_dpo_periodteacher, "
    "period eval + period judge) LoRA scale sweep"
)

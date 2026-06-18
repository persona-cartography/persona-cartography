"""Period-adjusted LLM-judge sweep for the talkie C- period-teacher DPO-ONLY adapter.

Evaluates the DPO adapter directly (before SFT + merge), so we get an early read
on the period-teacher fix while the full pipeline's introspection/SFT/merge is
still running. Same period-adjusted setup as
``c_minus_periodteacher_periodeval`` (1928-register prompts + period-aware
judges); only the adapter ref points at the ``-dpo`` checkpoint.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_dpo_periodeval
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_periodeval import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher"
    "/lora/conscientiousness_suppressing_full_vanton4_period-dpo"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

EVAL_NAME = "conscientiousness-suppressor-talkie1930-periodteacher-DPOonly-periodeval"
PLOT_TITLE = (
    "Conscientiousness suppressor (talkie-1930-13b-it, periodteacher DPO-only, "
    "period eval + period judge) LoRA scale sweep"
)

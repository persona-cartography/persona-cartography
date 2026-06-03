"""Agreeableness suppressor (ocean_const_paired_dpo) activation-capping LLM-judge sweep.

Usage::

    uv run python -m scripts.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo_activation_capping.a_minus \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo_activation_capping._shared import *  # noqa: F401,F403
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo_activation_capping._shared import (
    SCALE_POINTS,
    build_cap_config,
)
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src.visualisations.palette import BIG_FIVE_COLORS
from src.persona_metrics.metrics.ocean_v2 import OceanTrait

SLUG = "a_minus"
DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"

EVAL_NAME = "agreeableness-suppressor-vanton4-paired-dpo-activation-capping"
TRAIT = OceanTrait.agreeableness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/ocean_const_paired_dpo"
    "/lora/agreeableness_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Agreeableness suppressor (ocean_const_paired_dpo, Qwen3-235B judge) activation-capping sweep"

ACTIVATION_CAP_CONFIG = build_cap_config(SLUG)

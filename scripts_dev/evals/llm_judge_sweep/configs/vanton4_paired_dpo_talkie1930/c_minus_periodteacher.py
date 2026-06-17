"""Conscientiousness suppressor — talkie-1930-13b-it, PERIOD-TEACHER paired_dpo, Qwen3-235B judge.

Same setup as ``c_minus.py`` but pointed at the period-teacher adapter
(``vanton4_paired_dpo_periodteacher``), whose DPO chosen/rejected text was
regenerated in talkie's 1928 register (glm-4.5-air as a 1928 correspondent) to
fix the modern-llama register mismatch documented in
HANDOVER_TALKIE1930_2026-05-23.md.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"

EVAL_NAME = "conscientiousness-suppressor-talkie1930-vanton4-paired-dpo-periodteacher"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher"
    "/lora/conscientiousness_suppressing_full_vanton4_period-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = (
    "Conscientiousness suppressor (talkie-1930-13b-it, vanton4_paired_dpo_periodteacher, "
    "Qwen3-235B judge) LoRA scale sweep"
)

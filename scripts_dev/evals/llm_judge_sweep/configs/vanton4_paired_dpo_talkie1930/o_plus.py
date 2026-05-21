"""Openness amplifier (talkie-1930-13b-it, vanton4_paired_dpo, Qwen3-235B judge) LLM judge scale sweep.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.o_plus
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

EVAL_NAME = "openness-amplifier-talkie1930-vanton4-paired-dpo"
TRAIT = OceanTrait.openness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/talkie-1930-13b-it/ocean/openness/amplifier/vanton4_paired_dpo"
    "/lora/openness_amplifying_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "Openness amplifier (talkie-1930-13b-it, vanton4_paired_dpo, Qwen3-235B judge) LoRA scale sweep"

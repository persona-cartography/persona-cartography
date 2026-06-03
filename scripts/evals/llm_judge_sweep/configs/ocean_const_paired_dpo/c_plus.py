"""Conscientiousness amplifier (ocean_const_paired_dpo, Qwen3-235B judge) LLM judge scale sweep.

Usage::

    uv run python -m scripts.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.c_plus
"""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src.visualisations.palette import BIG_FIVE_COLORS
from src.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/conscientiousness.jsonl"

EVAL_NAME = "conscientiousness-amplifier-vanton4-paired-dpo"
TRAIT = OceanTrait.conscientiousness

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/ocean_const_paired_dpo"
    "/lora/conscientiousness_amplifying_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.conscientiousness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
PLOT_TITLE = "Conscientiousness amplifier (ocean_const_paired_dpo, Qwen3-235B judge) LoRA scale sweep"

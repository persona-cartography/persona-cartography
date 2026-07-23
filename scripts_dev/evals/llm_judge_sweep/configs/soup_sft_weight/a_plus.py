"""Agreeableness amplifier DPO + w·SFT soup-weight LLM judge sweep.

DPO fixed at 1.0, SFT weight w ∈ {0, 0.25, 0.5, 1.0} (Qwen3-235B judge).

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.soup_sft_weight.a_plus \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.soup_sft_weight._shared import *  # noqa: F401,F403
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/agreeableness.jsonl"

EVAL_NAME = "agreeableness-amplifier-soup-sft-weight"
TRAIT = OceanTrait.agreeableness

_TRAIT_DEF = OCEAN_REGISTRY["a_plus"]
DPO_ADAPTER = AdapterSpec.from_ref(_TRAIT_DEF.component_ref("dpo"), slug_suffix="dpo")
SFT_ADAPTER = AdapterSpec.from_ref(_TRAIT_DEF.component_ref("sft"), slug_suffix="sft")
ADAPTERS = [DPO_ADAPTER, SFT_ADAPTER]
SCALES_PER_ADAPTER = {
    DPO_ADAPTER.slug: DPO_SCALES,
    SFT_ADAPTER.slug: SFT_WEIGHTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.agreeableness.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Agreeableness"]
PLOT_TITLE = "Agreeableness amplifier DPO + w·SFT soup (Qwen3-235B judge)"

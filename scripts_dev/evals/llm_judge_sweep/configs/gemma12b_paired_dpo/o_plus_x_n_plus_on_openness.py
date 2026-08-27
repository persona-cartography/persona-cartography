"""o_plus × n_plus (gemma-3-12b-it, ocean_const_paired_dpo) on openness prompts.

Gemma replication of ``vanton4_paired_dpo.o_plus_x_n_plus_on_openness``:
5×5 scale grid over the O+ and N+ ocean_const_paired_dpo adapters, rollouts on
``data/ocean_open_ended/openness.jsonl``, judged on openness_v2 + coherence.
MAX_SAMPLES=100, NUM_ROLLOUTS_PER_PROMPT=1 — same lightweight budget as the
Llama heatmap sweeps.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma12b_paired_dpo.o_plus_x_n_plus_on_openness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma12b_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "o_plus_x_n_plus-gemma12b-paired-dpo-on-openness"
TRAIT = OceanTrait.openness

ADAPTER_O_PLUS = AdapterSpec.from_ref(
    "persona-cartography/monorepo::"
    "fine_tuning/gemma-3-12b-it/ocean/openness/amplifier/ocean_const_paired_dpo"
    "/lora/openness_amplifying_full-persona"
)
ADAPTER_N_PLUS = AdapterSpec.from_ref(
    "persona-cartography/monorepo::"
    "fine_tuning/gemma-3-12b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo"
    "/lora/neuroticism_amplifying_full-persona"
)
ADAPTERS = [ADAPTER_O_PLUS, ADAPTER_N_PLUS]
SCALES_PER_ADAPTER = {
    ADAPTER_O_PLUS.slug: SCALE_POINTS,
    ADAPTER_N_PLUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py.

TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = "o_plus × n_plus (gemma-3-12b ocean_const_paired_dpo) on openness prompts — Qwen3-235B judge"

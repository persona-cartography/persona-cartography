"""o_plus × n_plus (ocean_const_paired_dpo, 1:1 soup) evaluated on openness prompts.

Direct analog of ``vanton4_qwen3.c_minus_x_e_minus_on_conscientiousness`` but
for the o_plus × n_plus combo at the paired_dpo adapter version. Rollouts on
``data/ocean_open_ended/openness.jsonl``, judged on openness_v2 + coherence.

MAX_SAMPLES=100, NUM_ROLLOUTS_PER_PROMPT=1 — same lightweight budget as the
c×e_minus heatmap sweeps. Together with the trait-specific DATASET_PATH this
gives a fresh rollout fingerprint, distinct from prior single-adapter runs.

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.o_plus_x_n_plus_on_openness \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src.visualisations.palette import BIG_FIVE_COLORS
from src.evals.judges.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/openness.jsonl"

# Smaller / faster — matches c_minus × e_minus heatmap budget.
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1

EVAL_NAME = "o_plus_x_n_plus-vanton4-paired-dpo-on-openness"
TRAIT = OceanTrait.openness

ADAPTER_O_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/ocean_const_paired_dpo"
    "/lora/openness_amplifying_full_vanton4-persona"
)
ADAPTER_N_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo"
    "/lora/neuroticism_amplifying_full_vanton4-persona"
)
ADAPTERS = [ADAPTER_O_PLUS, ADAPTER_N_PLUS]
SCALES_PER_ADAPTER = {
    ADAPTER_O_PLUS.slug: SCALE_POINTS,
    ADAPTER_N_PLUS.slug: SCALE_POINTS,
}

JUDGE_METRIC_TRAITS = [OceanTrait.openness.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py.

TRAIT_COLOR = BIG_FIVE_COLORS["Openness"]
PLOT_TITLE = (
    "o_plus × n_plus (ocean_const_paired_dpo) on openness prompts — Qwen3-235B judge"
)

"""gemma-2b-it neuroticism suppressor (vanton4, Qwen3-235B judge) LLM judge scale sweep.

Gemma-2b analog of ``configs/vanton4_qwen3/n_minus.py`` (the llama-3.1-8b
neuroticism-suppressor judge sweep). Same dataset / judge / scale points; only
the base model and the adapter ref differ.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma2b_vanton4.n_minus \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma2b_vanton4._shared import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/neuroticism.jsonl"

EVAL_NAME = "gemma-2b-neuroticism-suppressor-vanton4"
TRAIT = OceanTrait.neuroticism

ADAPTER = AdapterSpec.from_ref(
    "persona-cartography/monorepo::"
    "fine_tuning/gemma-2b-it/ocean/neuroticism/suppressor/v1/lora/neuroticism_suppressing_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
# Wider scale range than the family default [-2..2] to probe extreme amplification
# / suppression. The judge-sweep fingerprint is scale-independent, so re-running
# reuses the cached -2..+2 cells and only computes the new extreme points.
SCALE_POINTS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

JUDGE_METRIC_TRAITS = [OceanTrait.neuroticism.v2_metric_name]
TRAIT_COLOR = BIG_FIVE_COLORS["Neuroticism"]
PLOT_TITLE = "gemma-2b neuroticism suppressor (vanton4, Qwen3-235B judge) LoRA scale sweep"

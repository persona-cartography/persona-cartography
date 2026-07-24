"""Shared constants for Gemma-3-12B ocean_const_paired_dpo LLM-judge sweeps.

Inherits everything from the vanton4_paired_dpo family (SCALE_POINTS
[-2,-1,0,1,2], Qwen3-235B single-judge rater, temperature-1.0 rollouts,
2048-token generations) and swaps the base model to Gemma-3-12B-IT. Per-config
modules override DATASET_PATH, EVAL_NAME, TRAIT, ADAPTERS, SCALES_PER_ADAPTER,
JUDGE_METRIC_TRAITS, TRAIT_COLOR, and PLOT_TITLE.

Gemma-3-12B fits vLLM defaults on an 80 GB GPU even with the rank-128 combined
combo LoRA (two r=64 adapters baked together), so no memory overrides here.
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403

BASE_MODEL = "google/gemma-3-12b-it"
BASE_MODEL_SLUG = "gemma-3-12b-it"

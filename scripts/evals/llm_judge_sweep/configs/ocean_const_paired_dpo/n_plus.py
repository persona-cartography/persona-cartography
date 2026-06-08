"""n_plus — LLM-judge LoRA scale sweep. Thin config; defaults in src.evals.llm_judge_sweep.config_builders."""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import SCALE_POINTS
from src.evals.llm_judge_sweep.config_builders import build_single_direction

globals().update(build_single_direction("n_plus", SCALE_POINTS))

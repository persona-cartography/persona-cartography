"""o_plus x n_plus combo judged on neuroticism prompts. Thin config; see config_builders."""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import SCALE_POINTS
from src.evals.judges.metrics.ocean_v2 import OceanTrait
from src.evals.llm_judge_sweep.config_builders import build_combo

globals().update(build_combo("o_plus", "n_plus", OceanTrait.neuroticism, SCALE_POINTS))

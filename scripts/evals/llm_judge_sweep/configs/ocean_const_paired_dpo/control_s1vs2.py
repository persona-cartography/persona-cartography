"""Recipe-only control base adapter. Thin config; see config_builders."""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import SCALE_POINTS
from src.evals.llm_judge_sweep.config_builders import control_adapter

ADAPTER = control_adapter()
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

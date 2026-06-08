"""n_minus judged on openness prompts. Thin config; see config_builders."""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.n_minus import *  # noqa: F401,F403
from src.evals.judges.metrics.ocean_v2 import OceanTrait
from src.evals.llm_judge_sweep.config_builders import build_on_trait

globals().update(build_on_trait("n_minus", OceanTrait.openness))

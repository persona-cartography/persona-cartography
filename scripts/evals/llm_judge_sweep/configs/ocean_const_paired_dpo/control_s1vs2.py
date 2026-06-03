"""Recipe-matched null control (ocean_const_paired_dpo_s1vs2) LLM judge base config.

Defines ADAPTER + scales for the s1vs2 control adapter. Not run directly — each
of the 5 control_s1vs2_on_<trait>.py modules imports this and overrides
DATASET_PATH / TRAIT / EVAL_NAME / JUDGE_METRIC_TRAITS to evaluate the control
on a specific trait's open-ended question set.
"""

from __future__ import annotations

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/ocean_const_paired_dpo_s1vs2"
    "/lora/ocean_def_control_full_vanton4-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

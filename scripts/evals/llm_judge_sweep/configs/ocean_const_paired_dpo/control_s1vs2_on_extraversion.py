"""Recipe-matched null control (ocean_const_paired_dpo_s1vs2) evaluated on Extraversion prompts.

The control adapter is a recipe-matched null (chosen/rejected both teacher-
generated under the same OCEAN-default constitution). This sweep generates
fresh rollouts on the extraversion open-ended question set and judges them for
extraversion. Useful as a baseline for the recipe-only contribution to Extraversion
shift; the OCEAN-trait adapters' extraversion sweeps should sit visibly above this.

Usage::

    uv run python -m scripts.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.control_s1vs2_on_extraversion \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the control_s1vs2 base.
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.control_s1vs2 import *  # noqa: F401,F403

from src.visualisations.palette import BIG_FIVE_COLORS
from src.evals.judges.metrics.ocean_v2 import OceanTrait

DATASET_PATH = "data/ocean_open_ended/extraversion.jsonl"
TRAIT = OceanTrait.extraversion
JUDGE_METRIC_TRAITS = [OceanTrait.extraversion.v2_metric_name]

EVAL_NAME = "control-vanton4-paired-dpo-s1vs2-on-extraversion"
TRAIT_COLOR = BIG_FIVE_COLORS["Extraversion"]
PLOT_TITLE = "Control (ocean_const_paired_dpo_s1vs2 null, Qwen3-235B judge) LoRA scale sweep on Extraversion prompts"

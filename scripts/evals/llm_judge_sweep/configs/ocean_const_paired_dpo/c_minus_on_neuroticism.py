"""c_minus adapter evaluated on Neuroticism prompts (fresh rollouts).

Overrides DATASET_PATH + JUDGE_METRIC_TRAITS so this sweep generates new
rollouts on the neuroticism prompt set and judges them for neuroticism.
Fingerprint differs from c_minus.py (different DATASET_PATH), so rollouts
do NOT cache-hit with Run 1.

Usage::

    uv run python -m scripts.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.c_minus_on_neuroticism \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

# Pull adapter, scales, base model, judge (Qwen3-235B), coherence metric, etc.
# from the own-trait module.
from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.c_minus import *  # noqa: F401,F403

from src.visualisations.palette import BIG_FIVE_COLORS
from src.evals.judges.metrics.ocean_v2 import OceanTrait

# Override: different prompt set → different rollout fingerprint.
DATASET_PATH = "data/ocean_open_ended/neuroticism.jsonl"
TRAIT = OceanTrait.neuroticism
JUDGE_METRIC_TRAITS = [OceanTrait.neuroticism.v2_metric_name]
# COHERENCE_METRIC inherited from _shared.py (fresh rollouts → fresh coherence).

EVAL_NAME = "conscientiousness-suppressor-vanton4-paired-dpo-on-neuroticism"
TRAIT_COLOR = BIG_FIVE_COLORS["Neuroticism"]
PLOT_TITLE = "Conscientiousness suppressor (ocean_const_paired_dpo, Qwen3-235B judge) LoRA scale sweep on Neuroticism prompts"

"""Shared constants for the DPO + w·SFT soup-weight LLM judge sweeps.

Re-exports the vanton4_paired_dpo family's rollout/judge settings *unchanged*
so the rollout fingerprint matches that family — the existing persona
scale-sweep cells (e.g. persona@+1.00 under ``llm_judge_lora_scale_sweep``)
stay directly comparable to these soup cells at analysis time.

The sweep grid fixes the DPO component at 1.0 and varies the SFT weight
w ∈ {0, 0.25, 0.5, 1.0}; w=0.25 mirrors the released persona merge (see
``scripts_dev/personality_evals/configs/ocean/soup_sft_weight/_common.py``
for the PEFT cross-term caveat). Cells are namespaced under a dedicated
``EVAL_NAME_CANONICAL`` so the DPO-only cell cannot collide with the persona
sweep's ``scale_+1.00`` cell at the shared fingerprint.
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403

# Dedicated cell namespace: keeps every soup cell (including the
# single-adapter DPO-only w=0 cell) apart from llm_judge_lora_scale_sweep
# data at the same rollout fingerprint.
EVAL_NAME_CANONICAL = "llm_judge_soup_sft_weight"

DPO_SCALES = [1.0]
SFT_WEIGHTS = [0.0, 0.25, 0.5, 1.0]

X_AXIS_LABEL = "SFT soup weight"

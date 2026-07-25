"""Shared constants for Gemma-3-27B ocean_const_paired_dpo LLM-judge sweeps.

Inherits everything from the vanton4_paired_dpo family (SCALE_POINTS
[-2,-1,0,1,2], Qwen3-235B single-judge rater, temperature-1.0 rollouts,
2048-token generations) and swaps the base model to Gemma-3-27B-IT.

Memory: 27B weights take ~54 GiB bf16 on an 80 GB card, and combo cells push
``max_lora_rank`` to 128 (two r=64 adapters baked together). We run the
multimodal base text-only (``ASSISTANT_LIMIT_MM_PER_PROMPT={"image": 0}``) to
reclaim the vision tower, cap the context at 4096 (prompts are short, so
2048-token generations never truncate), and shrink the assistant batch — the
same recipe as ``gemma_consc_sup/g27b.py`` plus the mm limit.
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403

BASE_MODEL = "google/gemma-3-27b-it"
BASE_MODEL_SLUG = "gemma-3-27b-it"

ASSISTANT_MAX_MODEL_LEN = 4096
ASSISTANT_GPU_MEMORY_UTILIZATION = 0.96
ASSISTANT_ENFORCE_EAGER = False
ASSISTANT_BATCH_SIZE = 8
ASSISTANT_LIMIT_MM_PER_PROMPT = {"image": 0}

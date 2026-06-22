"""Neuroticism amplifier (talkie-1930-13b-it, PERIOD-TEACHER paired-DPO) LLM-judge scale sweep.

Same sweep as ``n_plus`` but pointed at the period-teacher adapter
(``vanton4_paired_dpo_periodteacher``), which is the in-register talkie recipe
(see HANDOVER_TALKIE1930_2026-05-23.md). Reuses everything from ``n_plus`` and
only overrides the adapter ref + labels.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_plus_periodteacher \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_plus import *  # noqa: F401,F403
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec

ADAPTER = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/talkie-1930-13b-it/ocean/neuroticism/amplifier/vanton4_paired_dpo_periodteacher"
    "/lora/neuroticism_amplifying_full_vanton4_period-persona"
)
ADAPTERS = [ADAPTER]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}  # noqa: F405

EVAL_NAME = "neuroticism-amplifier-talkie1930-vanton4-paired-dpo-periodteacher"
PLOT_TITLE = (
    "Neuroticism amplifier (talkie-1930-13b-it, vanton4_paired_dpo_periodteacher, "
    "Qwen3-235B judge) LoRA scale sweep"
)

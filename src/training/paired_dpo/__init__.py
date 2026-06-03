"""Paired-DPO dataset construction.

Paired-teacher DPO joins an amplifier-teacher distillation and a
suppressor-teacher distillation on ``prompt`` and emits (chosen, rejected)
pairs per direction. See ``pairing.py`` for the join logic and
``scripts/training/ocean_paired_dpo/`` for the runnable entrypoints.
"""

from src.training.paired_dpo.pairing import (
    CHOSEN_COL,
    PROMPT_COL,
    REJECTED_COL_DEFAULT,
    build_paired_rows,
    load_jsonl,
)

__all__ = [
    "CHOSEN_COL",
    "PROMPT_COL",
    "REJECTED_COL_DEFAULT",
    "build_paired_rows",
    "load_jsonl",
]

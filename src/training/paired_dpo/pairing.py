"""Pure functions for building paired-teacher DPO datasets.

Paired-teacher DPO joins an amplifier-teacher distillation and a
suppressor-teacher distillation on ``prompt`` and emits DPO pairs:

  * Amplifier direction (``"amp"``): chosen = amp teacher, rejected = sup teacher.
    DPO pushes the model toward the amplified response, away from the suppressed.
  * Suppressor direction (``"sup"``): chosen = sup teacher, rejected = amp teacher.

The output preserves the OCT-native distillation schema so the downstream OCT
DPO stage reads it unchanged: ``CHOSEN_COL`` ("response") holds the chosen
teacher response, and the rejected column (named after the student/baseline
model, default "llama-3.1-8b-it") holds the rejected teacher response.

This module is the migrated, unit-tested core of
``scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py`` (``_build_paired_rows`` /
``_load_jsonl``). It has no I/O side effects beyond reading JSONL in
``load_jsonl``; file writing, provenance, stage markers, and HF upload live in
the ``scripts/training/ocean_paired_dpo/`` entrypoints.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

# OCT-native distillation schema, preserved verbatim so OCT's DPO stage reads it.
CHOSEN_COL = "response"
# Default rejected-column name. Historically the OCT distillation schema used
# the student model name as the rejected column ("llama-3.1-8b-it"), and
# load_dpo_pairs() in run_oct_pipeline.py looks up that exact column when
# building DPO pairs. When the student model is something else (e.g.
# gemma-3-27b-it), pass rejected_col to match — otherwise the downstream
# pipeline's column check (`if model not in _cols`) will fail.
REJECTED_COL_DEFAULT = "llama-3.1-8b-it"
PROMPT_COL = "prompt"

# Valid values for the ``amp_pairing`` argument of ``build_paired_rows``.
AMP_PAIRING_MODES = ("first", "random", "all")


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def build_paired_rows(
    amp_rows: list[dict],
    sup_rows: list[dict],
    direction: str,
    amp_pairing: str,
    seed: int,
    rejected_col: str = REJECTED_COL_DEFAULT,
    *,
    prompt_col: str = PROMPT_COL,
    chosen_col: str = CHOSEN_COL,
) -> tuple[list[dict], int, int]:
    """Inner-join amp/sup rows on prompt; emit (chosen, rejected) rows per direction.

    Args:
        amp_rows: Amplifier-teacher distillation rows (each a dict with at least
            ``prompt_col`` and ``chosen_col``).
        sup_rows: Suppressor-teacher distillation rows, same schema.
        direction: ``"amp"`` (chosen = amp, rejected = sup) or ``"sup"`` (chosen
            = sup, rejected = amp).
        amp_pairing: How to reconcile multiple amp teacher responses per prompt
            (vanton4 has 1 per prompt, older runs had ~5): ``"first"`` picks the
            first amp row, ``"random"`` picks a seeded random one, ``"all"``
            expands sup by duplicating it against every amp teacher (yielding up
            to N_amp× more pairs).
        seed: Seed for ``amp_pairing="random"`` (unused for ``first``/``all``).
        rejected_col: Output column name for the rejected response. Must match
            the student model name the downstream OCT DPO stage expects.
        prompt_col: Column to join on (default ``"prompt"``).
        chosen_col: Column holding each teacher's response (default ``"response"``).

    Returns:
        Tuple ``(rows, n_matched_pairs, n_unmatched_sup_prompts)``.

    Raises:
        ValueError: if ``amp_pairing`` or ``direction`` is unrecognised.
    """
    if amp_pairing not in AMP_PAIRING_MODES:
        raise ValueError(f"unknown amp_pairing: {amp_pairing}")
    if direction not in ("amp", "sup"):
        raise ValueError(f"unknown direction: {direction}")

    amp_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in amp_rows:
        p = r.get(prompt_col)
        if p is None:
            continue
        amp_by_prompt[p].append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    n_matched = 0
    n_unmatched = 0
    for sup_r in sup_rows:
        p = sup_r.get(prompt_col)
        candidates = amp_by_prompt.get(p, [])
        sup_teacher = sup_r.get(chosen_col)
        if not candidates or sup_teacher is None:
            n_unmatched += 1
            continue
        if amp_pairing == "first":
            picked = [candidates[0]]
        elif amp_pairing == "random":
            picked = [rng.choice(candidates)]
        else:  # "all"
            picked = candidates
        for amp_r in picked:
            amp_teacher = amp_r.get(chosen_col)
            if amp_teacher is None:
                continue
            if direction == "amp":
                row = {
                    prompt_col: p,
                    chosen_col: amp_teacher,
                    rejected_col: sup_teacher,
                }
            else:  # "sup"
                row = {
                    prompt_col: p,
                    chosen_col: sup_teacher,
                    rejected_col: amp_teacher,
                }
            out.append(row)
            n_matched += 1
    return out, n_matched, n_unmatched

"""Canonical defaults for llm-judge sweep configs.

The sweep's cell cache is keyed by :func:`rollout_fingerprint` — a SHA-256
of the rollout-generation settings (base model, dataset, sampling params,
seed, generation params). Any drift in these fields invalidates the
fingerprint and silently splits the cache, so two sweeps that *conceptually*
share cells end up recomputing everything.

This module pins one canonical value per fingerprint-affecting field. The
runner checks the active config against these canonical values and halts
with an interactive confirmation if anything differs, so deviations are
deliberate rather than accidental.

To *actually* change a default (because research needs have shifted), update
the value here. That makes the change a one-line, reviewable commit rather
than per-config drift.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from src.evals.cell_sweep.defaults import (
    DefaultDiff,
    check_defaults,
    confirm_or_abort as _confirm_or_abort,
    format_default_diffs,
)

__all__ = [
    "CANONICAL_SWEEP_DEFAULTS",
    "DefaultDiff",
    "check_sweep_defaults",
    "confirm_or_abort",
    "format_default_diffs",
]

# ---------------------------------------------------------------------------
# Canonical defaults for fingerprint-affecting fields
# ---------------------------------------------------------------------------
#
# Keys are the config module attribute names (upper-case). Values are the
# pinned canonical values. Any config attribute not listed here is either not
# fingerprint-affecting or free to vary without breaking cache reuse.
CANONICAL_SWEEP_DEFAULTS: dict[str, Any] = {
    "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "DATASET_PATH": "data/assistant-axis-extraction-questions.jsonl",
    "MAX_SAMPLES": 100,
    "SEED": 42,
    "NUM_ROLLOUTS_PER_PROMPT": 1,
    "ASSISTANT_TEMPERATURE": 0.7,
    "ASSISTANT_TOP_P": 0.95,
    "ASSISTANT_MAX_NEW_TOKENS": 256,
}


def check_sweep_defaults(cfg: ModuleType) -> list[DefaultDiff]:
    """Return config fields that deviate from the judge-sweep defaults."""
    return check_defaults(cfg, CANONICAL_SWEEP_DEFAULTS)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
) -> None:
    """Halt the runner if the config drifts from judge-sweep defaults."""
    _confirm_or_abort(
        diffs,
        allow_custom=allow_custom,
        interactive=interactive,
        label="judge sweep",
    )

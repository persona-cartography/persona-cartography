"""Canonical defaults for TRAIT-benchmark cell sweeps.

The sweep's cell cache is keyed by :func:`trait_fingerprint` — a SHA-256 of
the TRAIT-benchmark inputs (benchmark kind, sample counts, shuffle
behaviour, generation temperature). Any drift in these fields invalidates
the fingerprint and silently splits the cache.

This module pins one canonical value per fingerprint-affecting field. The
runner checks the active config against these values and halts with an
interactive confirmation if anything differs, so deviations are deliberate
rather than accidental.

``TRAIT_SPLITS`` is *not* in the fingerprint (each trait lives in its own
per-trait subdir within a cell, so running a subset of traits just leaves
others absent without re-keying the cache). For the same reason, it is not
listed here — subsetting traits is free and does not require a drift override.

``MIN_CHOICE_MASS`` and ``DYNAMIC_MASS_FILTER`` are also *not* in the
fingerprint — they are pure analysis-time filters over per-sample logprobs
and do not change what the model generates. They therefore do not appear
here either; adjusting the threshold is free and does not require a drift
override or cache rebuild. The default threshold itself lives at
:data:`src.evals.personality.logprob_scorer.MIN_CHOICE_MASS_DEFAULT`.

To change a default (because research needs have shifted), update the value
here. That makes the change a one-line, reviewable commit rather than
per-config drift.
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
    "CANONICAL_TRAIT_DEFAULTS",
    "DefaultDiff",
    "check_trait_defaults",
    "confirm_or_abort",
    "format_default_diffs",
]

CANONICAL_TRAIT_DEFAULTS: dict[str, Any] = {
    "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "BENCHMARK": "personality_trait_logprobs",
    "SAMPLES_PER_TRAIT": 300,
    "SHUFFLE_CHOICES": True,
    "SEED": 42,
    "TEMPERATURE": 0.0,
    "PREFILL": "ANSWER: ",
    "TEMPLATE": None,
    "MAX_TOKENS": None,
}


def check_trait_defaults(cfg: ModuleType) -> list[DefaultDiff]:
    """Return config fields that deviate from the TRAIT-sweep defaults."""
    return check_defaults(cfg, CANONICAL_TRAIT_DEFAULTS)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
) -> None:
    """Halt the runner if the config drifts from TRAIT-sweep defaults."""
    _confirm_or_abort(
        diffs,
        allow_custom=allow_custom,
        interactive=interactive,
        label="TRAIT sweep",
    )

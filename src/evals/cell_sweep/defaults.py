"""Canonical-defaults drift check for cell-sweep configs.

Each cell-sweep pipeline has a fingerprint (SHA-256 over a set of
configuration fields). Any drift in those fields invalidates the fingerprint
and silently splits the cache, so two sweeps that *conceptually* share cells
end up recomputing everything.

This module provides a pipeline-agnostic way to pin canonical values for
fingerprint-affecting fields and halt (with an interactive confirmation) if
a config deviates. Each pipeline owns its own canonical-values dict and
passes it to :func:`check_defaults`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class DefaultDiff:
    """A single deviation from the canonical defaults."""

    field: str
    canonical: Any
    actual: Any


def check_defaults(
    cfg: ModuleType, canonical: dict[str, Any]
) -> list[DefaultDiff]:
    """Return the list of config fields that deviate from canonical defaults.

    Only checks fields listed in ``canonical``. Missing attributes on ``cfg``
    are treated as deviations (value ``<missing>``) — a config is expected to
    set every fingerprint-affecting field explicitly.
    """
    diffs: list[DefaultDiff] = []
    for field, canonical_value in canonical.items():
        actual = getattr(cfg, field, "<missing>")
        if actual != canonical_value:
            diffs.append(
                DefaultDiff(field=field, canonical=canonical_value, actual=actual)
            )
    return diffs


def format_default_diffs(diffs: list[DefaultDiff]) -> str:
    """Render a human-readable summary of deviations for the y/n prompt."""
    if not diffs:
        return "(no deviations)"
    lines: list[str] = []
    for d in diffs:
        lines.append(f"  {d.field}:")
        lines.append(f"    canonical: {d.canonical!r}")
        lines.append(f"    config   : {d.actual!r}")
    return "\n".join(lines)


def confirm_or_abort(
    diffs: list[DefaultDiff],
    *,
    allow_custom: bool,
    interactive: bool = True,
    label: str = "sweep",
) -> None:
    """Halt the runner if the config deviates from canonical defaults.

    If ``allow_custom`` is true, returns immediately (caller has explicitly
    opted in). Otherwise prints a summary and requires an interactive ``y``
    confirmation. In a non-interactive session (``interactive=False``) or on
    ``n`` / EOF, raises ``SystemExit``.

    ``label`` names the pipeline in the messages ("sweep", "TRAIT sweep", …).

    Why: the cell-sweep fingerprint is derived from these fields. A
    one-character typo (e.g. ``MAX_SAMPLES=101``) silently forks the cache
    and forces a full recomputation that the user probably didn't intend.
    Forcing an explicit acknowledgment makes such drift deliberate.
    """
    if not diffs:
        return
    if allow_custom:
        print(
            f"[defaults] Config differs from canonical {label} defaults "
            f"({len(diffs)} field(s)); --allow-custom-fingerprint set, proceeding."
        )
        print(format_default_diffs(diffs))
        return

    print(
        f"\n[defaults] The active config deviates from the canonical {label} "
        "defaults in fields that affect the cell fingerprint:"
    )
    print(format_default_diffs(diffs))
    print(
        "\nProceeding will produce a fingerprint that does NOT share cache with "
        "the canonical-default sweeps. Re-use across sweeps will be broken for "
        "these runs.\n"
        "Pass --allow-custom-fingerprint to skip this prompt in future runs."
    )

    if not interactive:
        raise SystemExit(
            "Non-interactive run with non-default config. "
            "Re-run with --allow-custom-fingerprint if this is intended."
        )

    try:
        answer = input("Proceed with non-default config? [y/N]: ").strip().lower()
    except EOFError:
        raise SystemExit(
            "stdin closed; cannot confirm non-default config. "
            "Re-run with --allow-custom-fingerprint if this is intended."
        )
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")

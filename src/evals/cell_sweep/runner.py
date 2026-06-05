"""Shared orchestration helpers for cell-sweep runners.

Every cell-oriented eval runner (llm_judge_sweep, trait_sweep, inspect_sweep,
bloom) does the same thing around its eval-specific inner loop: parse a small
set of common flags, enumerate the Cartesian product of adapter × scale into
canonical cells, write a ``cell_info.json`` per cell, and push sweep-level
artifacts to HF. Those four helpers live here so each runner script only owns
its eval-specific generation + scoring logic.

Pipeline-specific hooks (cell-status shape, generation/scoring callbacks,
aggregation/plotting) stay in the eval-specific subpackages and runner
scripts — this module is hook-free by design.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping, Sequence

from src.evals.cell_sweep.cell_identity import AdapterSpec, CanonicalCell
from src.utils.hf_hub import upload_folder_to_dataset_repo


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

ExtraFlag = tuple[str, dict]
"""A ``(flag_name, kwargs)`` entry for ``parse_sweep_flags(extras=...)``.

``kwargs`` is forwarded directly to ``argparse.ArgumentParser.add_argument``
so callers can pass ``action="store_true"``, ``nargs="+"``, ``default=...``,
``help=...`` etc.
"""


def build_sweep_parser(
    description: str,
    *,
    extras: Sequence[ExtraFlag] | None = None,
) -> argparse.ArgumentParser:
    """Build an ``ArgumentParser`` with the flags every cell-sweep runner accepts.

    Flags:

    - ``--config`` (required): dotted Python module path to the config constants.
    - ``--dry-run``: print the planned sweep and exit.
    - ``--no-upload``: run without touching HuggingFace at all.
    - ``--allow-custom-fingerprint``: skip the canonical-defaults drift prompt.

    Runner-specific flags (e.g. ``--skip-rollouts``, ``--no-vllm``) should be
    passed through ``extras``.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        required=True,
        help="Python module path to the config constants.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument(
        "--allow-custom-fingerprint",
        action="store_true",
        help="Skip the canonical-defaults prompt for config drift.",
    )
    for flag, kwargs in extras or []:
        p.add_argument(flag, **kwargs)
    return p


def parse_sweep_flags(
    description: str,
    *,
    extras: Sequence[ExtraFlag] | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the common cell-sweep flags plus any runner-specific extras."""
    return build_sweep_parser(description, extras=extras).parse_args(argv)


def load_config_module(dotted_path: str) -> ModuleType:
    """Import a Python config module by dotted path (e.g. ``pkg.configs.foo``)."""
    return importlib.import_module(dotted_path)


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


def enumerate_cells(
    adapters: Sequence[AdapterSpec],
    scales_per_adapter: Mapping[str, Sequence[float]],
) -> list[CanonicalCell]:
    """Cartesian product over per-adapter scale lists, deduplicated.

    ``{A=1, B=0}`` and ``{A=1, C=0}`` both collapse to the same canonical
    cell ``{A=1}`` (zero-scale entries are dropped inside ``CanonicalCell``),
    so we dedupe by the canonical ``(slug, scale)`` multiset to avoid running
    the same eval twice.

    The ``scales_per_adapter`` mapping is keyed by ``AdapterSpec.slug`` and
    each value is a sequence of scales for that adapter. The order of the
    ``adapters`` argument determines the Cartesian-product axis order — it
    does not affect the resulting canonical cell identities.

    When ``adapters`` is empty, returns a single baseline cell (no entries).
    """
    if not adapters:
        return [CanonicalCell(entries=())]
    scale_lists = [scales_per_adapter[a.slug] for a in adapters]
    seen: set[tuple[tuple[str, float], ...]] = set()
    cells: list[CanonicalCell] = []
    for combo in itertools.product(*scale_lists):
        pairs = [(adapters[i], float(combo[i])) for i in range(len(adapters))]
        cell = CanonicalCell.from_scales(pairs)
        key = tuple((s.slug, sc) for s, sc in cell.entries)
        if key in seen:
            continue
        seen.add(key)
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# cell_info.json
# ---------------------------------------------------------------------------

CELL_INFO_FILENAME = "cell_info.json"


def cell_info_payload(cell: CanonicalCell, fingerprint: str) -> dict:
    """Return the ``cell_info.json`` dict for a cell.

    Callers occasionally want to enrich the payload (e.g. bloom adds an
    ``ideation_ref`` entry) — use this to build the base dict and mutate
    before writing, rather than redefining the whole schema.
    """
    return {
        "tier": cell.tier,
        "variant_label": cell.variant_label(),
        "entries": [
            {
                "ref": s.ref,
                "slug": s.slug,
                "category": s.category,
                "trait": s.trait,
                "direction": s.direction,
                "version": s.version,
                "scale": sc,
            }
            for s, sc in cell.entries
        ],
        "fingerprint": fingerprint,
    }


def write_cell_info(
    cell: CanonicalCell,
    cell_dir: Path,
    fingerprint: str,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Write ``cell_info.json`` into ``cell_dir``.

    ``extra`` is merged into the payload (callers override base fields at
    their own risk — prefer extra keys like ``ideation_ref``).
    """
    payload = cell_info_payload(cell, fingerprint)
    if extra:
        payload.update(extra)
    (cell_dir / CELL_INFO_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Sweep-root upload
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_ROOT_ALLOW_PATTERNS: tuple[str, ...] = (
    "plots/**",
    "analysis/**",
    "sweep_config.json",
)


def upload_sweep_root(
    local_dir: Path,
    *,
    hf_path: str,
    repo_id: str,
    commit_message: str,
    allow_patterns: Iterable[str] | None = None,
) -> None:
    """Upload the sweep-level summary dir (plots, aggregates) to HF.

    Per-cell artifacts are uploaded via the eval-specific ``upload_cell``
    helpers; this handles everything at the sweep root (``sweep_hf_root``).
    """
    patterns = (
        list(allow_patterns)
        if allow_patterns is not None
        else list(DEFAULT_SWEEP_ROOT_ALLOW_PATTERNS)
    )
    upload_folder_to_dataset_repo(
        local_dir=local_dir,
        repo_id=repo_id,
        path_in_repo=hf_path,
        commit_message=commit_message,
        allow_patterns=patterns,
    )

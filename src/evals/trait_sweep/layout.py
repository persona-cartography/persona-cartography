"""TRAIT-sweep cell artifact layout + hydrate/upload helpers.

Cell shape (both local and on HF)::

    <cell_dir>/
      run_info.json
      native/
        inspect_logs/
          <Trait>/
            *.json            (Inspect .json eval log for that trait)
      cell_info.json

Each trait split lives in its own subdir under ``native/inspect_logs/``.
A cell is considered to "have" trait T when ``native/inspect_logs/T/``
contains at least one ``*.json`` log. ``run_info.json`` must also be
present with ``status == "ok"``; if the suite run that produced the cell
failed, the cell is treated as missing regardless of per-trait logs.

Cross-repo IO is delegated to :mod:`src.evals.cell_sweep.cache`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.evals.cell_sweep.cache import hydrate_cell_dir, upload_cell_dir
from src.evals.cell_sweep.cell_identity import CanonicalCell

RUN_INFO_RELPATH = "run_info.json"
INSPECT_LOGS_RELDIR = "native/inspect_logs"
CELL_INFO_RELPATH = "cell_info.json"

UPLOAD_ALLOW_PATTERNS = [
    "run_info.json",
    "native/inspect_logs/**",
    "cell_info.json",
]


@dataclass
class CellArtifactStatus:
    """What's materialised at a TRAIT cell's local dir right now.

    ``present_traits`` is the set of trait-split names for which at least
    one Inspect log exists under ``native/inspect_logs/<Trait>/``. A cell
    counts as "covering" a set of required traits when its ``present_traits``
    is a superset.
    """

    present_traits: set[str] = field(default_factory=set)

    def covers(self, required_traits: Iterable[str]) -> bool:
        return set(required_traits).issubset(self.present_traits)


def cell_status_on_disk(cell_dir: Path) -> CellArtifactStatus:
    """Inspect a local TRAIT cell dir and report which traits are present.

    A trait ``T`` counts as present when ``native/inspect_logs/T/`` exists
    and contains at least one ``*.json`` file. ``run_info.json`` must also
    exist with ``status == "ok"``; otherwise the cell is considered empty
    regardless of what's under ``native/inspect_logs/``.
    """
    status = CellArtifactStatus()
    run_info = cell_dir / RUN_INFO_RELPATH
    if not run_info.exists():
        return status
    try:
        info = json.loads(run_info.read_text())
    except Exception:
        return status
    if info.get("status") != "ok":
        return status
    logs_dir = cell_dir / INSPECT_LOGS_RELDIR
    if not logs_dir.is_dir():
        return status
    for trait_dir in logs_dir.iterdir():
        if not trait_dir.is_dir():
            continue
        if any(trait_dir.glob("*.json")):
            status.present_traits.add(trait_dir.name)
    return status


def hydrate_cell(
    cell: CanonicalCell,
    *,
    scratch_root: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    skip_download: bool = False,
) -> tuple[Path, CellArtifactStatus]:
    """Pull cell artifacts from HF and return ``(local_dir, status)``."""
    local_dir = hydrate_cell_dir(
        cell,
        scratch_root=scratch_root,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
        repo_id=repo_id,
        skip_download=skip_download,
    )
    return local_dir, cell_status_on_disk(local_dir)


def upload_cell(
    cell: CanonicalCell,
    *,
    local_dir: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    commit_message: str,
    allow_patterns: list[str] | None = None,
) -> str:
    """Upload the cell's local dir to its canonical HF path."""
    return upload_cell_dir(
        cell,
        local_dir=local_dir,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
        repo_id=repo_id,
        commit_message=commit_message,
        allow_patterns=allow_patterns
        if allow_patterns is not None
        else list(UPLOAD_ALLOW_PATTERNS),
    )

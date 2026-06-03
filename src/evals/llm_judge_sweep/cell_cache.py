"""Judge-sweep cell artifact layout + hydrate/upload helpers.

Cell shape (both local and on HF)::

    <cell_dir>/
      rollouts/
        rollouts.jsonl
        rollout_info.json
        experiment_metadata.json
      judge_runs/
        {rater_id}/
          {metric_name}.jsonl
      cell_info.json
      manifest.json

Everything under ``judge_runs/`` is per-(rater, metric) — a cell can be
hydrated with rollouts but missing a judge metric, in which case the runner
just re-runs the missing judge against the cached rollouts.

Cross-repo IO is delegated to :mod:`src.evals.cell_sweep.cache`; only the
judge-specific status shape lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src.evals.cell_sweep.cache import hydrate_cell_dir, upload_cell_dir
from src.evals.cell_sweep.cell_identity import CanonicalCell

ROLLOUTS_RELPATH = "rollouts/rollouts.jsonl"
ROLLOUT_INFO_RELPATH = "rollouts/rollout_info.json"
CELL_INFO_RELPATH = "cell_info.json"

UPLOAD_ALLOW_PATTERNS = [
    "rollouts/**",
    "judge_runs/**",
    "cell_info.json",
]


@dataclass
class CellArtifactStatus:
    """What's materialised at a cell's local dir right now (judge layout)."""

    has_rollouts: bool = False
    present_judge_metrics: set[tuple[str, str]] = field(default_factory=set)

    def missing_judge_metrics(
        self, required: Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return [rm for rm in required if rm not in self.present_judge_metrics]


def cell_status_on_disk(
    cell_dir: Path,
    *,
    required_judge_metrics: Sequence[tuple[str, str]] = (),
) -> CellArtifactStatus:
    """Inspect a local cell dir and report what's present.

    ``required_judge_metrics`` controls which ``(rater_id, metric_name)``
    pairs are *checked*. Extra files on disk outside this set are ignored;
    missing ones just don't appear in the returned status.
    """
    status = CellArtifactStatus()
    status.has_rollouts = (cell_dir / ROLLOUTS_RELPATH).exists()
    for rater_id, metric in required_judge_metrics:
        if (cell_dir / "judge_runs" / rater_id / f"{metric}.jsonl").exists():
            status.present_judge_metrics.add((rater_id, metric))
    return status


def hydrate_cell(
    cell: CanonicalCell,
    *,
    scratch_root: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    required_judge_metrics: Sequence[tuple[str, str]] = (),
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
    return local_dir, cell_status_on_disk(
        local_dir, required_judge_metrics=required_judge_metrics
    )


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

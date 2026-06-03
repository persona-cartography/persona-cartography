"""Generic per-cell HF hydrate/upload primitives.

Pipelines (llm_judge_sweep, trait_sweep, …) each own their on-disk artifact
layout — what counts as "core", what per-metric files may exist, what to
upload. This module is layout-agnostic: it just moves a cell's directory
between its canonical HF path and local scratch, and trusts the caller to
inspect the local dir afterwards with a pipeline-specific status helper.
"""

from __future__ import annotations

from pathlib import Path

from src.evals.cell_sweep.cell_identity import CanonicalCell
from src.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_path_to_dir,
    upload_folder_to_dataset_repo,
)


def hydrate_cell_dir(
    cell: CanonicalCell,
    *,
    scratch_root: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    skip_download: bool = False,
) -> Path:
    """Pull any HF artifacts for the cell into its canonical local dir.

    Returns the local directory (created if missing). Safe to call on cells
    that aren't yet on HF — the existence check short-circuits cleanly.

    Why do the existence check first: ``download_path_to_dir`` under a
    missing prefix can raise or silently succeed with zero files depending
    on the HF client version; the explicit check keeps behaviour predictable.

    Args:
        skip_download: When True, only ensure the local dir exists — use for
            ``--no-upload``-style flows where HF is disabled.
    """
    local_dir = cell.local_dir(
        scratch_root=scratch_root,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    if skip_download:
        return local_dir

    hf_dir = cell.hf_dir(model_slug, eval_name, fingerprint)
    if check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=hf_dir):
        download_path_to_dir(
            repo_id=repo_id,
            path_in_repo=hf_dir,
            target_dir=local_dir,
        )

    return local_dir


def upload_cell_dir(
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
    """Upload the cell's local dir to its canonical HF path.

    ``allow_patterns`` (relative to ``local_dir``) narrows the upload to
    specific files — useful when only new per-metric outputs need pushing
    and core artifacts were hydrated from HF (no need to re-upload).
    """
    hf_dir = cell.hf_dir(model_slug, eval_name, fingerprint)
    return upload_folder_to_dataset_repo(
        local_dir=local_dir,
        repo_id=repo_id,
        path_in_repo=hf_dir,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
    )

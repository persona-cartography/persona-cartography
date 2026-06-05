"""Shared hydration + scoring helpers for the OCEAN soup-heatmap figures.

The two soup-heatmap scripts (``scripts/figures/main_c_e_soup_heatmaps.py`` and
``scripts/figures/main_o_n_soup_heatmaps.py``) resolve judge jsonls along
different HF layouts (canonical-tier path resolution vs a flat bundle subtree)
and draw subtly different heatmaps, so their path-construction and rendering
stay in the individual scripts. The two pieces that *are* byte-for-byte
identical — the cache-then-download hydration of a single judge file, and the
plain mean of its per-row scores — live here so both scripts call one copy.

Behaviour is identical to the pre-refactor per-script copies.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.utils.hf_hub import download_path_to_dir
from src.visualisations.judge_jsonl import mean_judge_score

HF_REPO_ID = "persona-shattering-lasr/monorepo"


def hydrate_judge_file(hf_path: str, cache_dir: Path, *, repo_id: str = HF_REPO_ID) -> Path | None:
    """Resolve ``hf_path`` to a local file, preferring the local cache then HF.

    Lookup order:
      1. ``cache_dir/{hf_path}`` (from a previous run).
      2. Fresh download from HF into that cache location.

    Returns the local path on success, else ``None`` (printing a one-line
    diagnostic on download failure).
    """
    local = cache_dir / hf_path
    if local.exists() and local.stat().st_size > 0:
        return local
    parent_hf = hf_path.rsplit("/", 1)[0]
    filename = hf_path.rsplit("/", 1)[1]
    local_parent = cache_dir / parent_hf
    try:
        download_path_to_dir(
            repo_id=repo_id,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        print(f"  ✗ hydrate failed for {hf_path}: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    if local.exists() and local.stat().st_size > 0:
        return local
    return None


def mean_score(jsonl_path: Path) -> float | None:
    """Plain mean over judge scores in a jsonl (failed-call rows excluded).

    Delegates to the shared judge-jsonl reducer so every figure aggregates
    judge scores identically.
    """
    return mean_judge_score(jsonl_path)

"""Starter analysis for the OCEAN LoRA-combination experiment.

Pulls the per-config TRAIT + MMLU results from HF (or reads a local cache),
joins them with the scale/direction design, and produces a tidy DataFrame plus a
couple of starter plots. This is scaffolding — the foundation for the real
analysis (e.g. regressing MMLU and trait scores on sumscale / per-trait scale),
not a finished study.

For richer sweep parsing see
``src_dev/evals/personality/analyze_results.py``; here the single-point-per-config
layout is simple enough that we read the Inspect log JSON directly.

Usage
-----
    uv run python -m scripts_dev.combinations_experiment.analyze            # download + analyze
    uv run python -m scripts_dev.combinations_experiment.analyze --no-download   # use local cache
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts_dev.combinations_experiment.config_design import (
    TRAITS,
    ConfigRecord,
    generate_design,
)
from scripts_dev.combinations_experiment.run_experiment import HF_PREFIX, HF_REPO_ID

CACHE_ROOT = Path("scratch/combinations_experiment_analysis/cache")
FIGURES_DIR = Path("scratch/combinations_experiment_analysis/figures")
EVAL_NAMES = ("trait_logprobs", "mmlu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _download_config(slug: str) -> Path:
    """Download one config's JSON artifacts from HF into the local cache."""
    from src_dev.utils.hf_hub import download_path_to_dir

    target = CACHE_ROOT / slug
    download_path_to_dir(
        repo_id=HF_REPO_ID,
        path_in_repo=f"{HF_PREFIX}/{slug}",
        target_dir=target,
        allow_patterns=["**/*.json"],
    )
    return target


def _read_metrics(eval_dir: Path) -> dict[str, float]:
    """Extract ``{metric_name: value}`` from an eval's Inspect log JSON.

    Mirrors ``analyze_results._extract_scores_from_log``: reads
    ``results.scores[*].metrics{name: {value}}`` across all score entries.
    """
    logs = sorted(eval_dir.glob("native/inspect_logs/*.json"))
    if not logs:
        return {}
    with logs[0].open() as f:
        log = json.load(f)
    if log.get("status") != "success":
        return {}
    out: dict[str, float] = {}
    for entry in log.get("results", {}).get("scores", []):
        for name, val in entry.get("metrics", {}).items():
            if isinstance(val, dict) and "value" in val:
                out[name] = float(val["value"])
    return out


def _row_for_config(record: ConfigRecord, config_dir: Path) -> dict:
    """Build one tidy row: design fields + per-trait scales + eval metrics."""
    row: dict[str, object] = {
        "slug": record.slug,
        "index": record.index,
        "sumscale": record.sumscale_actual,
    }
    for t in record.traits:
        row[f"dir_{t.letter}"] = t.direction
        row[f"scale_{t.letter}"] = t.scale
        row[f"signed_{t.letter}"] = t.signed_scale

    for eval_name in EVAL_NAMES:
        for metric, value in _read_metrics(config_dir / eval_name).items():
            row[f"{eval_name}__{metric}"] = value
    return row


def load_results(*, download: bool = True) -> pd.DataFrame:
    """Assemble the per-config results DataFrame (32 rows)."""
    design = generate_design()
    rows = []
    for record in design.configs:
        config_dir = _download_config(record.slug) if download else CACHE_ROOT / record.slug
        if not config_dir.exists():
            print(f"  missing results for {record.slug} (skipping)")
            continue
        rows.append(_row_for_config(record, config_dir))
    return pd.DataFrame(rows).sort_values("sumscale").reset_index(drop=True)


def _mmlu_column(df: pd.DataFrame) -> str | None:
    """Best-guess the MMLU accuracy column name."""
    for c in df.columns:
        if c.startswith("mmlu__") and "accuracy" in c:
            return c
    candidates = [c for c in df.columns if c.startswith("mmlu__")]
    return candidates[0] if candidates else None


def plot_starters(df: pd.DataFrame) -> None:
    """Two starter plots: MMLU vs sumscale, and per-trait score vs signed scale."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mmlu_col = _mmlu_column(df)
    if mmlu_col is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["sumscale"], df[mmlu_col], c="tab:blue")
        ax.set_xlabel("Σ scale (total adapter magnitude)")
        ax.set_ylabel("MMLU accuracy")
        ax.set_title("MMLU vs total adapter magnitude")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "mmlu_vs_sumscale.png", dpi=150)
        plt.close(fig)
        print(f"  saved {FIGURES_DIR / 'mmlu_vs_sumscale.png'}")

    # Per-trait: trait score (if present) vs that trait's signed scale.
    trait_cols = {
        letter: next(
            (c for c in df.columns if c.startswith("trait_logprobs__") and letter.lower() in c.lower()),
            None,
        )
        for letter in TRAITS
    }
    if any(trait_cols.values()):
        fig, axes = plt.subplots(1, len(TRAITS), figsize=(4 * len(TRAITS), 3.5), sharey=True)
        for ax, letter in zip(axes, TRAITS):
            col = trait_cols[letter]
            if col is None:
                ax.set_visible(False)
                continue
            ax.scatter(df[f"signed_{letter}"], df[col], c="tab:green")
            ax.axvline(0.0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel(f"signed scale {letter}")
            ax.set_title(letter)
        axes[0].set_ylabel("trait score")
        fig.suptitle("Per-trait score vs that trait's signed scale")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "trait_score_vs_signed_scale.png", dpi=150)
        plt.close(fig)
        print(f"  saved {FIGURES_DIR / 'trait_score_vs_signed_scale.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-download", action="store_true", help="Use the local cache instead of downloading from HF.")
    args = parser.parse_args()

    df = load_results(download=not args.no_download)
    if df.empty:
        print("No results found.")
        return

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = Path("scratch/combinations_experiment_analysis/results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{len(df)} configs loaded -> {csv_path}")
    print(df.to_string(index=False))

    plot_starters(df)


if __name__ == "__main__":
    main()

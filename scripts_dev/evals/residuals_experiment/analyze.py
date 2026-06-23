#!/usr/bin/env python3
"""Compute and plot LoRA interaction residuals from the residuals experiment.

Reads ``scratch/residuals_experiment/scores.json`` (written by run.py) and
computes the interaction residual for every pair (i, j) of adapters and every
OCEAN judge metric k:

    ε_ij(S_k) = S_k(W + Δi + Δj) - S_k(W + Δi) - S_k(W + Δj) + S_k(W)

Small |ε| indicates approximate linear composability.  Large |ε| indicates
nonlinear interaction or trait entanglement — which is itself informative.

NOTE on interpretation
~~~~~~~~~~~~~~~~~~~~~~
Residuals conflate three sources:

  (a) Genuine adapter interaction in weight space (the scientifically
      interesting signal).
  (b) Trait entanglement: adapter Δi shifts trait k even when it's not Δi's
      "target" trait; adding Δj on top of Δi may compound or cancel this.
  (c) Judge-score saturation (measurement nonlinearity): the judge scale is
      bounded ≈ ±4, so effects near the limits appear compressed, inflating
      residuals even for perfectly additive adapters.

At adapter scale=1 (far from saturation), (c) is likely small; (a) and (b)
dominate.  To disentangle them, look at which off-target metrics have the
largest residuals — those point to entanglement, not true weight interaction.

Data source
~~~~~~~~~~~
By default, ``scores.json`` is pulled from HuggingFace at::

    persona-shattering-lasr/monorepo
      └─ evals/residuals_experiment/residuals-vanton4-paired-dpo/scores.json

The local cache lives at ``scratch/residuals_experiment/_hf_cache/...``.  Pass
``--scores PATH`` to use a local scores.json instead (e.g. immediately after
``run.py``).

Outputs (all under scratch/residuals_experiment/)::

    residuals.json             — full ε table (pair × metric)
    residuals_summary.txt      — ranked summary table, sorted by mean|ε|
    plots/heatmap_{metric}.png — per-metric 11×10 ε heatmap (local)
    plots/residual_distribution.pdf            — KDE distribution (raw)
    plots/residual_distribution_normalized.pdf — KDE distribution (judge-σ normalized)

Paper figures (written to paper/figures/)::

    appendix/heatmaps_residuals/fig_residuals_heatmap.pdf                  — 1×5 panel heatmap with ctrl row
    main/fig_residuals_distribution.pdf                 — 6 KDE curves (raw)
    main/fig_residuals_distribution_normalized.pdf      — 6 KDE curves (ε / σ_k)

Usage (reproducing paper figures from HF)::

    uv run python -m scripts_dev.evals.residuals_experiment.analyze

Usage (using a local scores.json, e.g. just after run.py)::

    uv run python -m scripts_dev.evals.residuals_experiment.analyze \\
        --scores scratch/residuals_experiment/scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

SCORES_PATH = project_root / "scratch" / "residuals_experiment" / "scores.json"
OUTPUT_DIR = project_root / "scratch" / "residuals_experiment"

# HuggingFace location of the residuals data (uploaded by run.py).
HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_EVAL_NAME = "residuals-vanton4-paired-dpo"
HF_SCORES_PATH = f"evals/residuals_experiment/{HF_EVAL_NAME}/scores.json"

PAPER_FIGURES = [
    "appendix/heatmaps_residuals/fig_residuals_heatmap.pdf",
    "main/fig_residuals_distribution.pdf",
    "main/fig_residuals_distribution_normalized.pdf",
]

# Short display labels for adapter slugs → friendly key mapping.
_DISPLAY_LABELS = {
    "o_plus": "O+", "o_minus": "O−",
    "c_plus": "C+", "c_minus": "C−",
    "e_plus": "E+", "e_minus": "E−",
    "a_plus": "A+", "a_minus": "A−",
    "n_plus": "N+", "n_minus": "N−",
    "ctrl": "Ctrl",
}


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def _get_score(
    cell_scores: dict[str, dict[str, Any]],
    cell_tag: str,
    metric: str,
) -> float | None:
    return cell_scores.get(cell_tag, {}).get(metric)


def compute_residuals(
    cell_scores: dict[str, dict[str, Any]],
    adapter_slugs: list[str],
    friendly_slugs: dict[str, str],
    metrics: list[str],
    adapter_scale: float,
    cell_metadata: dict[str, Any],
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute ε_ij for all C(n,2) pairs and all metrics.

    Args:
        cell_scores: cell_tag → {metric → mean_score}
        adapter_slugs: canonical slug for each of the 10 adapters
        friendly_slugs: AdapterSpec.slug → registry key (e.g. "o_plus")
        metrics: list of OCEAN metric names
        adapter_scale: the fixed scale used for all adapters
        cell_metadata: cell_tag → {tier, adapters}

    Returns:
        {(friendly_i, friendly_j): {metric: epsilon}}
    """
    # Build a lookup from friendly key → cell_tag for singles.
    baseline_tag: str = ""
    single_tag_by_friendly: dict[str, str] = {}
    pair_tag_by_friendly: dict[tuple[str, str], str] = {}

    for tag, meta in cell_metadata.items():
        tier = meta["tier"]
        adapters = meta["adapters"]
        if tier == "baseline":
            baseline_tag = tag
        elif tier == "single_adapter":
            friendly = adapters[0]["friendly"]
            single_tag_by_friendly[friendly] = tag
        elif tier == "combo" and len(adapters) == 2:
            friendly_pair = tuple(sorted(a["friendly"] for a in adapters))
            pair_tag_by_friendly[friendly_pair] = tag  # type: ignore[index]

    baseline_scores = cell_scores.get(baseline_tag, {})

    # Friendly keys in OCEAN order (matches the original adapter list order)
    friendly_keys = [friendly_slugs.get(slug, slug) for slug in adapter_slugs]

    residuals: dict[tuple[str, str], dict[str, float]] = {}
    for i_idx, fi in enumerate(friendly_keys):
        for j_idx, fj in enumerate(friendly_keys):
            if j_idx <= i_idx:
                continue  # unordered pairs only
            pair_key = (fi, fj)
            sorted_pair = tuple(sorted([fi, fj]))
            pair_tag = pair_tag_by_friendly.get(sorted_pair)  # type: ignore[arg-type]
            i_tag = single_tag_by_friendly.get(fi)
            j_tag = single_tag_by_friendly.get(fj)

            if not all([pair_tag, i_tag, j_tag]):
                continue

            pair_residuals: dict[str, float] = {}
            for metric in metrics:
                s_ij = _get_score(cell_scores, pair_tag, metric)
                s_i = _get_score(cell_scores, i_tag, metric)
                s_j = _get_score(cell_scores, j_tag, metric)
                s_0 = baseline_scores.get(metric)
                if all(v is not None for v in [s_ij, s_i, s_j, s_0]):
                    # ε_ij = S(W+Δi+Δj) - S(W+Δi) - S(W+Δj) + S(W)
                    pair_residuals[metric] = s_ij - s_i - s_j + s_0  # type: ignore[operator]
                else:
                    pair_residuals[metric] = float("nan")
            residuals[pair_key] = pair_residuals

    return residuals


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _display_label(friendly_key: str) -> str:
    return _DISPLAY_LABELS.get(friendly_key, friendly_key)


def format_summary_table(
    residuals: dict[tuple[str, str], dict[str, float]],
    metrics: list[str],
) -> str:
    """Ranked table: each pair × mean|ε| / max|ε| / per-metric ε values."""
    rows: list[tuple[str, str, float, float, dict[str, float]]] = []
    for (fi, fj), metric_vals in residuals.items():
        finite_vals = [v for v in metric_vals.values() if np.isfinite(v)]
        if not finite_vals:
            continue
        mean_abs = float(np.mean(np.abs(finite_vals)))
        max_abs = float(np.max(np.abs(finite_vals)))
        rows.append((fi, fj, mean_abs, max_abs, metric_vals))
    rows.sort(key=lambda r: r[2], reverse=True)

    short_metrics = [m.replace("_v2", "")[:5] for m in metrics]
    header = (
        f"{'Pair':<8}  {'mean|ε|':>7}  {'max|ε|':>7}  "
        + "  ".join(f"{m:>7}" for m in short_metrics)
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for fi, fj, mean_abs, max_abs, metric_vals in rows:
        pair = f"{_display_label(fi)}×{_display_label(fj)}"
        metric_strs = "  ".join(
            f"{metric_vals.get(m, float('nan')):>+7.3f}" for m in metrics
        )
        lines.append(f"{pair:<8}  {mean_abs:>7.3f}  {max_abs:>7.3f}  {metric_strs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


# Friendly keys that should be shown as appended rows (not part of the
# symmetric OCEAN block) in the heatmap.
EXTRA_ROW_KEYS = {"ctrl"}


def _split_keys(friendly_keys: list[str]) -> tuple[list[str], list[str]]:
    """Split friendly_keys into the OCEAN block (cols/rows) and extras (rows only)."""
    ocean = [k for k in friendly_keys if k not in EXTRA_ROW_KEYS]
    extras = [k for k in friendly_keys if k in EXTRA_ROW_KEYS]
    return ocean, extras


def _build_residual_matrix(
    residuals: dict[tuple[str, str], dict[str, float]],
    ocean_keys: list[str],
    extra_keys: list[str],
    metric: str,
) -> np.ndarray:
    """Build the heatmap matrix.

    Shape (n_ocean + n_extra, n_ocean):
      - rows 0..n_ocean-1, cols j (with j > r): upper triangle of OCEAN×OCEAN.
      - rows n_ocean..: extras (e.g. ctrl); each cell is ε(extra, OCEAN_j).
      - all other cells (diagonal + lower triangle of OCEAN block): NaN.
    """
    n_ocean = len(ocean_keys)
    n_extra = len(extra_keys)
    mat = np.full((n_ocean + n_extra, n_ocean), float("nan"))

    def _eps(a: str, b: str) -> float:
        d = residuals.get((a, b)) or residuals.get((b, a)) or {}
        return d.get(metric, float("nan"))

    # Upper triangle of OCEAN×OCEAN.
    for r, fr in enumerate(ocean_keys):
        for c, fc in enumerate(ocean_keys):
            if c <= r:
                continue  # diagonal + lower triangle stay NaN
            mat[r, c] = _eps(fr, fc)

    # Extra rows: full row of extra × OCEAN_j.
    for ex_idx, ex_key in enumerate(extra_keys):
        for c, fc in enumerate(ocean_keys):
            mat[n_ocean + ex_idx, c] = _eps(ex_key, fc)

    return mat


def plot_residual_heatmaps(
    residuals: dict[tuple[str, str], dict[str, float]],
    friendly_keys: list[str],
    metrics: list[str],
    out_dir: Path,
    paper_fig_path: Path | None = None,
) -> None:
    """Write per-metric heatmaps and an optional composite paper figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ocean_keys, extra_keys = _split_keys(friendly_keys)
    row_labels = [_display_label(k) for k in ocean_keys + extra_keys]
    col_labels = [_display_label(k) for k in ocean_keys]
    n_rows = len(ocean_keys) + len(extra_keys)
    n_cols = len(ocean_keys)
    n_ocean = len(ocean_keys)

    mats = {m: _build_residual_matrix(residuals, ocean_keys, extra_keys, m) for m in metrics}
    vmax = max(1.0, max(float(np.nanmax(np.abs(mat))) for mat in mats.values()))

    # Use a colormap that renders NaN cells as light grey (clean upper-triangle look).
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")

    out_dir.mkdir(parents=True, exist_ok=True)

    def _draw_panel(ax, mat, *, label_fontsize: int, value_fontsize: int, value_fmt: str):
        """Render one heatmap panel with optional horizontal separator above extras."""
        ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=label_fontsize)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)
        for i in range(n_rows):
            for j in range(n_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, value_fmt.format(v), ha="center", va="center",
                            fontsize=value_fontsize, color=color)
        if extra_keys:
            ax.axhline(n_ocean - 0.5, color="black", lw=1.0)

    # Per-metric standalone heatmaps (for local inspection).
    for metric, mat in mats.items():
        short = metric.replace("_v2", "")
        fig, ax = plt.subplots(figsize=(7, 6))
        _draw_panel(ax, mat, label_fontsize=9, value_fontsize=6, value_fmt="{:+.2f}")
        ax.set_title(
            f"Interaction residuals ε_ij — {short} judge"
            + (" (upper triangle + ctrl row)" if extra_keys else " (upper triangle)"),
            fontsize=10,
        )
        im = ax.images[0]
        plt.colorbar(im, ax=ax, label="ε_ij(S_k)")
        fig.tight_layout()
        out_path = out_dir / f"heatmap_{short}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    # Composite 1×5 panel for the paper.
    if paper_fig_path is not None:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5.0))
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            mat = mats[metric]
            short = metric.replace("_v2", "")
            _draw_panel(ax, mat, label_fontsize=6, value_fontsize=4.5, value_fmt="{:+.1f}")
            ax.set_title(short, fontsize=9)
        im = axes[-1].images[0]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="ε_ij(S_k)")
        fig.suptitle(
            r"Interaction residuals $\epsilon_{ij}(S_k)$ across OCEAN judges"
            "\n"
            r"$\epsilon_{ij} = S(W+\Delta_i+\Delta_j) - S(W+\Delta_i) - S(W+\Delta_j) + S(W)$",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.07, right=0.90, top=0.82, wspace=0.35)
        paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote paper figure: {paper_fig_path}")


# ---------------------------------------------------------------------------
# Distribution figure (main-text)
# ---------------------------------------------------------------------------

# OCEAN metric → display colour
_METRIC_COLORS = {
    "openness": "#4477AA",
    "conscientiousness": "#EE6677",
    "extraversion": "#228833",
    "agreeableness": "#CCBB44",
    "neuroticism": "#AA3377",
}


def per_metric_single_adapter_std(
    cell_scores: dict[str, dict[str, Any]],
    cell_metadata: dict[str, Any],
    metrics: list[str],
    exclude_friendly: set[str] | None = None,
) -> dict[str, float]:
    """For each metric k, return std of S(W+Δi) − S(W) across single-adapter cells.

    Used as a per-judge "headroom" scale for normalizing residuals so that the
    5 OCEAN judges become directly comparable (cures conscientiousness's
    apparent dominance, which is a judge-sensitivity artifact).
    """
    if exclude_friendly is None:
        exclude_friendly = {"ctrl"}
    baseline_tag = ""
    single_tags: list[str] = []
    for tag, meta in cell_metadata.items():
        if meta["tier"] == "baseline":
            baseline_tag = tag
        elif meta["tier"] == "single_adapter":
            adapters = meta["adapters"]
            if any(a["friendly"] in exclude_friendly for a in adapters):
                continue
            single_tags.append(tag)

    baseline_scores = cell_scores.get(baseline_tag, {})
    sigmas: dict[str, float] = {}
    for m in metrics:
        s_0 = baseline_scores.get(m)
        if s_0 is None:
            continue
        deltas = []
        for tag in single_tags:
            s_i = cell_scores.get(tag, {}).get(m)
            if s_i is not None:
                deltas.append(s_i - s_0)
        if len(deltas) >= 2:
            sigmas[m] = float(np.std(deltas, ddof=1))
    return sigmas


def plot_residual_distribution(
    residuals: dict[tuple[str, str], dict[str, float]],
    metrics: list[str],
    out_dir: Path,
    paper_fig_path: Path | None = None,
    normalize_by: dict[str, float] | None = None,
    out_filename: str = "residual_distribution.pdf",
) -> None:
    """Single axes, 6 smooth KDE curves: 5 OCEAN scorers + control (pooled).

    Args:
        normalize_by: optional {metric_name: σ_k}.  When provided, each ε is
            divided by σ_k for its scorer — converts the x-axis to "judge-std
            units" so the 5 scorers become directly comparable.
        out_filename: local PDF filename under ``out_dir``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    short = {m: m.replace("_v2", "") for m in metrics}
    metric_order = [short[m] for m in metrics]
    short_to_metric = {short[m]: m for m in metrics}

    def _scale(metric_short: str, eps: float) -> float:
        if normalize_by is None:
            return eps
        sigma = normalize_by.get(short_to_metric[metric_short])
        if sigma is None or sigma <= 0:
            return float("nan")
        return eps / sigma

    # Collect ε values per scorer for OCEAN-pair, plus pooled control.
    ocean_by_metric: dict[str, list[float]] = {ms: [] for ms in metric_order}
    ctrl_by_metric: dict[str, list[float]] = {ms: [] for ms in metric_order}
    for (fi, fj), metric_vals in residuals.items():
        is_ctrl = "ctrl" in (fi, fj)
        for metric, eps in metric_vals.items():
            if not np.isfinite(eps):
                continue
            ms = short[metric]
            v = _scale(ms, eps)
            if not np.isfinite(v):
                continue
            (ctrl_by_metric if is_ctrl else ocean_by_metric)[ms].append(v)

    ctrl_pooled = [v for vs in ctrl_by_metric.values() for v in vs]
    all_vals = [v for vs in ocean_by_metric.values() for v in vs] + ctrl_pooled
    lo = float(min(all_vals) - 0.5)
    hi = float(max(all_vals) + 0.5)
    x_kde = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)

    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.45, zorder=1)

    # 5 OCEAN-scorer smooth KDE curves.
    for ms in metric_order:
        vals = np.array(ocean_by_metric[ms])
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        y = kde(x_kde)
        ax.plot(x_kde, y, color=_METRIC_COLORS[ms], lw=2.0,
                label=ms, zorder=3)
        ax.fill_between(x_kde, y, alpha=0.06, color=_METRIC_COLORS[ms], zorder=2)

    # Pooled control as a 6th smooth KDE curve.
    CTRL_COLOR = "#444444"
    if len(ctrl_pooled) >= 2:
        kde = gaussian_kde(np.array(ctrl_pooled), bw_method="scott")
        y = kde(x_kde)
        ax.plot(x_kde, y, color=CTRL_COLOR, lw=2.2, ls="--",
                label="control", zorder=4)

    ax.legend(fontsize=8, framealpha=0.85, loc="upper right")
    if normalize_by is not None:
        ax.set_xlabel(
            r"Normalized residual  $\epsilon_{ij}(S_k) \,/\, \sigma_k$"
            r"   ($\sigma_k$ = std of single-adapter shifts $S(W+\Delta_i)-S(W)$)",
            fontsize=9,
        )
        title = (r"LoRA interaction residuals (judge-$\sigma$ normalized): "
                 r"per OCEAN scorer + control")
    else:
        ax.set_xlabel(r"Interaction residual $\epsilon_{ij}(S_k)$", fontsize=9)
        title = r"LoRA interaction residuals $\epsilon_{ij}(S_k)$: per OCEAN scorer + control"
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    if paper_fig_path is not None:
        paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_fig_path, bbox_inches="tight")
        print(f"[plot] wrote paper figure: {paper_fig_path}")
    local_path = out_dir / out_filename
    fig.savefig(local_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {local_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot LoRA interaction residuals.  By default fetches "
            f"scores.json from HF ({HF_REPO_ID}::{HF_SCORES_PATH}); pass "
            "--scores to use a local file instead."
        ),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional local scores.json path (skips HF download).",
    )
    parser.add_argument(
        "--no-paper-fig",
        action="store_true",
        help="Skip writing the paper figure to paper/figures/.",
    )
    return parser.parse_args()


def _resolve_scores_path(local_override: Path | None) -> Path:
    """Return a path to scores.json — either the local override or downloaded from HF."""
    if local_override is not None:
        if not local_override.exists():
            print(f"[error] local scores not found: {local_override}")
            sys.exit(1)
        print(f"[scores] using local file: {local_override}")
        return local_override

    from huggingface_hub import hf_hub_download
    from src_dev.utils.hf_hub import _get_token

    cache_dir = OUTPUT_DIR / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scores] downloading from HF: {HF_REPO_ID}::{HF_SCORES_PATH}")
    path = Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=HF_SCORES_PATH,
        local_dir=str(cache_dir),
        token=_get_token(),
    ))
    print(f"[scores] cached at: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    flags = _parse_flags()
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    scores_path = _resolve_scores_path(flags.scores)

    with scores_path.open() as f:
        data = json.load(f)

    cell_scores: dict[str, dict[str, Any]] = data["cell_scores"]
    adapter_slugs: list[str] = data["adapter_slugs"]
    friendly_slugs: dict[str, str] = data["friendly_slugs"]
    metrics: list[str] = data["ocean_metrics"]
    adapter_scale: float = data.get("adapter_scale", 1.0)
    cell_metadata: dict[str, Any] = data["cell_metadata"]

    # Friendly keys in the original adapter order.
    friendly_keys = [friendly_slugs.get(slug, slug) for slug in adapter_slugs]

    # Compute residuals.
    residuals = compute_residuals(
        cell_scores, adapter_slugs, friendly_slugs, metrics, adapter_scale, cell_metadata,
    )

    n_pairs = len(residuals)
    n_finite = sum(
        1 for v in (vv for vals in residuals.values() for vv in vals.values())
        if np.isfinite(v)
    )
    print(f"[residuals] computed {n_pairs} pairs × {len(metrics)} metrics "
          f"({n_finite} finite values)")

    # Save full table.
    residuals_out = {
        f"{fi}×{fj}": dict(vals)
        for (fi, fj), vals in sorted(residuals.items())
    }
    out_path = OUTPUT_DIR / "residuals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(residuals_out, f, indent=2)
    print(f"[residuals] wrote {out_path}")

    # Print and save summary table.
    summary = format_summary_table(residuals, metrics)
    print("\n" + summary)
    summary_path = OUTPUT_DIR / "residuals_summary.txt"
    summary_path.write_text(summary)
    print(f"\n[residuals] wrote {summary_path}")

    # Plots.
    plots_dir = OUTPUT_DIR / "plots"

    paper_fig_path: Path | None = None
    if not flags.no_paper_fig:
        try:
            from src_dev.visualisations import PAPER_FIGURES_DIR
            paper_fig_path = PAPER_FIGURES_DIR / "appendix" / "heatmaps_residuals" / "fig_residuals_heatmap.pdf"
        except ImportError:
            print("[warn] Could not import PAPER_FIGURES_DIR; skipping paper figure.")

    plot_residual_heatmaps(
        residuals,
        friendly_keys=friendly_keys,
        metrics=metrics,
        out_dir=plots_dir,
        paper_fig_path=paper_fig_path,
    )

    # Main-text distributions: 6 KDE curves (5 OCEAN scorers + control), both raw
    # and judge-σ-normalized.  σ_k = std of single-adapter shifts for metric k
    # (OCEAN singles only; ctrl excluded so it doesn't pull σ down).
    dist_paper_path: Path | None = None
    dist_norm_paper_path: Path | None = None
    if not flags.no_paper_fig:
        try:
            from src_dev.visualisations import PAPER_FIGURES_DIR
            dist_paper_path = PAPER_FIGURES_DIR / "main" / "fig_residuals_distribution.pdf"
            dist_norm_paper_path = (
                PAPER_FIGURES_DIR / "main" / "fig_residuals_distribution_normalized.pdf"
            )
        except ImportError:
            pass

    # Raw ε distribution.
    plot_residual_distribution(
        residuals,
        metrics=metrics,
        out_dir=plots_dir,
        paper_fig_path=dist_paper_path,
    )

    # Judge-σ-normalized ε distribution.
    sigmas = per_metric_single_adapter_std(cell_scores, cell_metadata, metrics)
    print("[residuals] per-metric σ (single-adapter shift std, ctrl excluded):")
    for m, s in sigmas.items():
        print(f"  {m:<22} σ = {s:.3f}")
    plot_residual_distribution(
        residuals,
        metrics=metrics,
        out_dir=plots_dir,
        paper_fig_path=dist_norm_paper_path,
        normalize_by=sigmas,
        out_filename="residual_distribution_normalized.pdf",
    )

    print(f"\n[done] all outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

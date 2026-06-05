#!/usr/bin/env python3
"""Generate publication-quality figures for the judge calibration paper section.

Saves vector PDFs to paper/figures/appendix/ per paper CLAUDE.md conventions.
Reuses data loading from human_annotation_analysis.py.

Figures produced:
    fig_F_judge_cross_trait.pdf   — heatmap: 13 judges × 6 traits, ρ vs gold
    fig_F_judge_scatter.pdf       — scatter of 3 panel judges vs human mean
    fig_F_judge_agreement_bars.pdf — inter/intra-rater agreement bars for panel

Usage::

    # Generate all appendix figures
    uv run python scripts/persona_metrics/llm_judge/plot_paper_judge_calibration.py

    # Single figure
    uv run python scripts/persona_metrics/llm_judge/plot_paper_judge_calibration.py \\
        --figure cross_trait
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse everything data-related from the analysis script
from scripts.figures.human_annotation_analysis import (
    ALL_TRAITS,
    CALIBRATION_DIR,
    LLM_JUDGE_RUNS,
    SCORE_RANGE,
    discover_human_raters,
    load_golden,
    load_human_scores,
    load_llm_judge_scores,
)
from src.persona_metrics.judge_calibration import (
    spearman_r,
)
from src.persona_metrics.llm_judge_agreement import _krippendorff_alpha_ordinal
from src.visualisations import PAPER_FIGURES_DIR

# Paper figures this script generates (paths relative to paper/figures/).
PAPER_FIGURES = [
    "appendix/fig_F_judge_cross_trait.pdf",
    "appendix/fig_F_judge_mae_heatmap.pdf",
    "appendix/fig_F_judge_scatter.pdf",
    "appendix/fig_F_judge_agreement_bars.pdf",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PANEL_JUDGES = ["Qwen 3 235B", "Gemma 4 27B", "Llama 3.3 70B"]
ANNOTATED_TRAITS = ["agreeableness", "neuroticism", "coherence"]
APPENDIX_DIR = PAPER_FIGURES_DIR / "appendix"

# Display labels for traits
TRAIT_LABELS = {
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "coherence": "Coherence",
}

# Friendly display names for the ordered judge list (for heatmap rows)
JUDGE_DISPLAY_ORDER = [
    "Qwen 3 235B",
    "Gemma 4 27B",
    "Llama 3.3 70B",
    "Kimi K2",
    "Gemini Flash",
    "GPT-5 Mini",
    "DeepSeek V3",
    "Mistral Small 3.2",
    "Qwen 2.5 72B",
    "Gemini Flash Lite",
    "Haiku 3.5",
    "Llama 4 Scout",
    "GPT-4.1 Nano",
]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_all_judge_scores(trait: str) -> dict[str, dict[str, float]]:
    """Return {judge_name: {item_id: median_score}} for all judges with data."""
    results = {}
    for name in LLM_JUDGE_RUNS:
        scores = load_llm_judge_scores(name, trait)
        if scores:
            results[name] = scores
    return results


def load_all_human_scores(trait: str) -> dict[str, dict[str, float]]:
    """Return {anon_id: {item_id: score}} for human raters who rated *trait*."""
    results = {}
    for anon_id in discover_human_raters(trait):
        scores, is_dummy = load_human_scores(anon_id, trait)
        if not is_dummy:
            results[anon_id] = scores
    return results


def compute_human_mean(
    humans: dict[str, dict[str, float]], item_ids: list[str]
) -> dict[str, float]:
    """Compute per-item mean across human raters."""
    means = {}
    for iid in item_ids:
        vals = [h[iid] for h in humans.values() if iid in h]
        if vals:
            means[iid] = statistics.mean(vals)
    return means


def compute_human_median(
    humans: dict[str, dict[str, float]], item_ids: list[str]
) -> dict[str, float]:
    """Compute per-item median across human raters (preserves ordinal scale)."""
    medians = {}
    for iid in item_ids:
        vals = [h[iid] for h in humans.values() if iid in h]
        if vals:
            medians[iid] = statistics.median(vals)
    return medians


def compute_intra_rater_alpha(judge_name: str, trait: str) -> float | None:
    """Compute intra-rater Krippendorff's α across 3 runs for one judge."""
    lo, hi = SCORE_RANGE[trait]
    run_dir = CALIBRATION_DIR / LLM_JUDGE_RUNS[judge_name] / "raw"
    if not run_dir.exists():
        return None

    scores_by_id: dict[str, list[int]] = defaultdict(list)
    for ri in range(3):
        p = run_dir / f"{trait}_run_{ri}.jsonl"
        if not p.exists():
            return None
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            s = item.get("judge_score")
            if s is not None and lo <= s <= hi:
                scores_by_id[item["id"]].append(int(s))

    item_ratings = [scores for scores in scores_by_id.values() if len(scores) >= 2]
    if not item_ratings:
        return None
    return _krippendorff_alpha_ordinal(item_ratings, score_min=lo, score_max=hi)


# ---------------------------------------------------------------------------
# Figure 1: Cross-trait heatmap of ρ vs gold
# ---------------------------------------------------------------------------


def _compute_judge_trait_matrix(
    metric: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute judge × trait matrix of some metric vs gold.

    metric: "spearman", "mae", or "mae_normalised" (MAE / scale range).
    """
    judges = [j for j in JUDGE_DISPLAY_ORDER if j in LLM_JUDGE_RUNS]
    traits = ALL_TRAITS
    matrix = np.full((len(judges), len(traits)), np.nan)

    for ji, judge in enumerate(judges):
        for ti, trait in enumerate(traits):
            golden = load_golden(trait)
            judge_scores = load_llm_judge_scores(judge, trait)
            if not judge_scores:
                continue
            item_ids = [iid for iid in golden if iid in judge_scores]
            if len(item_ids) < 5:
                continue
            g = [golden[iid]["gold_score"] for iid in item_ids]
            j = [judge_scores[iid] for iid in item_ids]
            if metric == "spearman":
                matrix[ji, ti] = spearman_r(g, j)
            elif metric == "mae":
                matrix[ji, ti] = sum(abs(a - b) for a, b in zip(g, j)) / len(g)
            elif metric == "mae_normalised":
                lo, hi = SCORE_RANGE[trait]
                mae = sum(abs(a - b) for a, b in zip(g, j)) / len(g)
                matrix[ji, ti] = mae / (hi - lo)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return matrix, judges, traits


def _plot_heatmap(
    matrix: np.ndarray,
    judges: list[str],
    traits: list[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    title: str,
    output: Path,
    fmt: str = ".2f",
    reverse_contrast: bool = False,
) -> Path:
    """Generic heatmap of judges × traits with cell annotations."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([TRAIT_LABELS[t] for t in traits], rotation=30, ha="right")
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges)

    for label in ax.get_yticklabels():
        if label.get_text() in PANEL_JUDGES:
            label.set_fontweight("bold")

    # Annotate cells. For metrics where "high = good" (spearman), dark
    # cells at the high end get white text. For MAE where "low = good",
    # reverse.
    for ji in range(len(judges)):
        for ti in range(len(traits)):
            val = matrix[ji, ti]
            if np.isnan(val):
                ax.text(ti, ji, "—", ha="center", va="center", fontsize=9, color="gray")
            else:
                midpoint = (vmin + vmax) / 2
                if reverse_contrast:
                    colour = (
                        "white" if val > midpoint + (vmax - vmin) * 0.25 else "black"
                    )
                else:
                    colour = (
                        "white" if val < midpoint - (vmax - vmin) * 0.25 else "black"
                    )
                ax.text(
                    ti,
                    ji,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=colour,
                )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=11, pad=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


def plot_cross_trait_heatmap(output: Path) -> Path:
    """Heatmap: Spearman ρ vs gold for all 13 judges × 6 traits.

    Panel members highlighted with bold row labels.
    """
    matrix, judges, traits = _compute_judge_trait_matrix("spearman")
    return _plot_heatmap(
        matrix,
        judges,
        traits,
        cmap="RdYlGn",
        vmin=0.70,
        vmax=1.00,
        cbar_label="Spearman ρ vs gold",
        title="Cross-trait Spearman ρ of LLM judges vs gold labels\n"
        "(panel judges in bold)",
        output=output,
    )


def plot_mae_heatmap(output: Path) -> Path:
    """Heatmap: Normalised MAE vs gold (MAE / scale range) for 13 judges × 6 traits.

    MAE is normalised by scale range (8 for OCEAN, 10 for coherence) so
    coherence and OCEAN are on a comparable 0-1 "error per unit range"
    scale. Highlights scale-calibration failures — some judges (e.g. Kimi K2
    on coherence) have high ρ but high MAE: they rank items correctly but
    systematically mis-score them.
    """
    matrix, judges, traits = _compute_judge_trait_matrix("mae_normalised")
    return _plot_heatmap(
        matrix,
        judges,
        traits,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=0.35,
        cbar_label="MAE / scale range (lower = better)",
        title="Cross-trait normalised MAE of LLM judges vs gold labels\n"
        "(panel judges in bold; MAE divided by trait scale span)",
        output=output,
        fmt=".2f",
        reverse_contrast=True,
    )


# ---------------------------------------------------------------------------
# Figure 2: Confusion matrices — 3 panel judges × 3 annotated traits
# ---------------------------------------------------------------------------


def plot_scatter_grid(output: Path) -> Path:
    """3x3 scatter grid: panel judges (rows) vs human mean, per trait (cols).

    Each point is one item. Jitter applied to integer judge scores so
    stacked items are visible. Red dashed line = y=x (perfect agreement).
    """
    nrows, ncols = len(PANEL_JUDGES), len(ANNOTATED_TRAITS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False
    )

    judge_colours = {
        "Qwen 3 235B": "#800000",
        "Gemma 4 27B": "#808000",
        "Llama 3.3 70B": "#9A6324",
    }

    rng = np.random.default_rng(42)

    for ri, judge in enumerate(PANEL_JUDGES):
        for ci, trait in enumerate(ANNOTATED_TRAITS):
            ax = axes[ri][ci]
            lo, hi = SCORE_RANGE[trait]

            humans = load_all_human_scores(trait)
            golden = load_golden(trait)
            ids = sorted(golden.keys())
            hmean = compute_human_mean(humans, ids)
            judge_scores = load_llm_judge_scores(judge, trait)

            xs, ys_raw = [], []
            for iid in ids:
                if iid in hmean and iid in judge_scores:
                    xs.append(hmean[iid])
                    ys_raw.append(judge_scores[iid])
            ys_jittered = [y + rng.uniform(-0.15, 0.15) for y in ys_raw]

            ax.scatter(
                xs,
                ys_jittered,
                color=judge_colours[judge],
                alpha=0.55,
                s=32,
                edgecolor="white",
                linewidth=0.4,
            )

            # y=x diagonal reference
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, alpha=0.5)

            # Compute Spearman ρ on actual (non-jittered) scores
            if len(xs) >= 2:
                rho = spearman_r(xs, ys_raw)
                ax.text(
                    0.04,
                    0.96,
                    f"ρ = {rho:.2f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )

            ax.set_xlim(lo - 0.5, hi + 0.5)
            ax.set_ylim(lo - 0.5, hi + 0.5)
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_aspect("equal", adjustable="box")

            if ri == 0:
                ax.set_title(TRAIT_LABELS[trait], fontsize=11)
            if ri == nrows - 1:
                ax.set_xlabel("Human mean", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{judge}\njudge score", fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


# ---------------------------------------------------------------------------
# Figure 3: Agreement bar charts (inter- and intra-rater)
# ---------------------------------------------------------------------------


def plot_agreement_bars(output: Path) -> Path:
    """Two panels: (left) inter-rater ρ vs human mean, (right) intra-rater α.

    Panel A shows the 3 panel judges + individual humans (LOO) for each
    annotated trait. Panel B shows intra-rater Krippendorff's α for the
    3 panel judges across all 6 traits.
    """
    fig, (ax_inter, ax_intra) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: inter-rater ρ vs human mean ─────────────────────────────────
    n_traits = len(ANNOTATED_TRAITS)
    x = np.arange(n_traits)
    width = 0.12
    judge_colours = {
        "Qwen 3 235B": "#800000",
        "Gemma 4 27B": "#808000",
        "Llama 3.3 70B": "#ffd8b1",
    }
    human_colour = "#4363d8"

    # Build {rater: [rho_per_trait]} for the 3 judges and 3 humans
    rows = {}
    for judge in PANEL_JUDGES:
        vals = []
        for trait in ANNOTATED_TRAITS:
            humans = load_all_human_scores(trait)
            hmean = compute_human_mean(humans, list(load_golden(trait)))
            judge_scores = load_llm_judge_scores(judge, trait)
            ids = sorted(set(hmean.keys()) & set(judge_scores.keys()))
            if ids:
                vals.append(
                    spearman_r(
                        [hmean[iid] for iid in ids], [judge_scores[iid] for iid in ids]
                    )
                )
            else:
                vals.append(float("nan"))
        rows[judge] = vals

    # Humans: leave-one-out vs median of the other humans
    human_ids = set()
    for trait in ANNOTATED_TRAITS:
        human_ids.update(load_all_human_scores(trait).keys())
    human_ids = sorted(human_ids)
    for hid in human_ids:
        vals = []
        for trait in ANNOTATED_TRAITS:
            humans = load_all_human_scores(trait)
            if hid not in humans:
                vals.append(float("nan"))
                continue
            others = {k: v for k, v in humans.items() if k != hid}
            ids = sorted(humans[hid].keys())
            loo_vals, me_vals = [], []
            for iid in ids:
                others_scores = [o[iid] for o in others.values() if iid in o]
                if others_scores:
                    loo_vals.append(statistics.median(others_scores))
                    me_vals.append(humans[hid][iid])
            if loo_vals:
                vals.append(spearman_r(me_vals, loo_vals))
            else:
                vals.append(float("nan"))
        rows[hid] = vals

    # Draw bars
    all_raters = PANEL_JUDGES + human_ids
    n_raters = len(all_raters)
    for i, rater in enumerate(all_raters):
        offset = (i - (n_raters - 1) / 2) * width
        colour = judge_colours.get(rater, human_colour)
        bars = ax_inter.bar(
            x + offset,
            rows[rater],
            width,
            color=colour,
            edgecolor="white",
            linewidth=0.5,
            label=rater,
        )
        for bar, val in zip(bars, rows[rater]):
            if not np.isnan(val):
                ax_inter.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    # Human-human α reference lines
    for ti, trait in enumerate(ANNOTATED_TRAITS):
        humans = load_all_human_scores(trait)
        golden_ids = list(load_golden(trait))
        names = sorted(humans.keys())
        ratings_per_item = []
        lo, hi = SCORE_RANGE[trait]
        for iid in golden_ids:
            rats = [int(humans[n][iid]) for n in names if iid in humans[n]]
            if rats:
                ratings_per_item.append(rats)
        alpha = _krippendorff_alpha_ordinal(
            ratings_per_item,
            score_min=lo,
            score_max=hi,
        )
        ax_inter.hlines(
            alpha,
            ti - 0.5,
            ti + 0.5,
            colors="black",
            linestyles="--",
            linewidth=1.5,
            label="H-H α" if ti == 0 else None,
        )

    ax_inter.set_xticks(x)
    ax_inter.set_xticklabels([TRAIT_LABELS[t] for t in ANNOTATED_TRAITS])
    ax_inter.set_ylabel("Spearman ρ vs human mean / H-H α", fontsize=10)
    ax_inter.set_ylim(0, 1.05)
    ax_inter.set_title("(A) Inter-rater agreement", fontsize=11)
    ax_inter.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)
    ax_inter.grid(axis="y", alpha=0.3)
    ax_inter.spines[["top", "right"]].set_visible(False)

    # ── Panel B: intra-rater α across all 6 traits ───────────────────────────
    n_traits_all = len(ALL_TRAITS)
    x_all = np.arange(n_traits_all)
    width_b = 0.25

    intra_rows = {}
    for judge in PANEL_JUDGES:
        vals = []
        for trait in ALL_TRAITS:
            alpha = compute_intra_rater_alpha(judge, trait)
            vals.append(alpha if alpha is not None else float("nan"))
        intra_rows[judge] = vals

    for i, judge in enumerate(PANEL_JUDGES):
        offset = (i - 1) * width_b
        colour = judge_colours[judge]
        bars = ax_intra.bar(
            x_all + offset,
            intra_rows[judge],
            width_b,
            color=colour,
            edgecolor="white",
            linewidth=0.5,
            label=judge,
        )
        for bar, val in zip(bars, intra_rows[judge]):
            if not np.isnan(val):
                ax_intra.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    ax_intra.set_xticks(x_all)
    ax_intra.set_xticklabels(
        [TRAIT_LABELS[t][:5] for t in ALL_TRAITS], rotation=30, ha="right"
    )
    ax_intra.set_ylabel("Intra-rater Krippendorff's α", fontsize=10)
    ax_intra.set_ylim(0.0, 1.05)
    ax_intra.set_title("(B) Self-consistency (temp=0.7, 3 runs)", fontsize=11)
    ax_intra.legend(fontsize=7, loc="lower right")
    ax_intra.grid(axis="y", alpha=0.3)
    ax_intra.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


FIGURES: dict[str, tuple[str, callable]] = {
    "cross_trait": ("fig_F_judge_cross_trait.pdf", plot_cross_trait_heatmap),
    "mae_heatmap": ("fig_F_judge_mae_heatmap.pdf", plot_mae_heatmap),
    "scatter": ("fig_F_judge_scatter.pdf", plot_scatter_grid),
    "agreement_bars": ("fig_F_judge_agreement_bars.pdf", plot_agreement_bars),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=list(FIGURES) + ["all"],
        default="all",
        help="Which figure to generate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_DIR,
        help=f"Output directory (default: {APPENDIX_DIR}).",
    )
    args = parser.parse_args()

    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        filename, fn = FIGURES[name]
        fn(args.output_dir / filename)


if __name__ == "__main__":
    main()

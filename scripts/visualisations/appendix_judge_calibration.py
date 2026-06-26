#!/usr/bin/env python3
"""Judge-calibration appendix figures (LLM-judge selection / calibration).

Regenerable from the HF monorepo: all data is loaded from
``persona-cartography/monorepo:judge_calibration/v2`` via
:mod:`src.visualisations.judge_calibration_common` (gold author labels, human
annotations, and raw LLM-judge runs). No gitignored local data is required.

Paper figures (written to ``paper/figures/appendix/ocean_evals/``):
    - fig_F_judge_cross_trait_and_mae.pdf  ((a) Spearman ρ + (b) normalised MAE)
    - fig_F_judge_scatter.pdf              (3 panel judges vs human mean)
    - fig_F_judge_agreement_bars.pdf       (inter-/intra-rater agreement)

``--figure cross_trait`` / ``--figure mae_heatmap`` also emit the two standalone
heatmap panels, but only the combined figure above is included in the paper.

Usage::

    uv run python scripts/visualisations/appendix_judge_calibration.py
    uv run python scripts/visualisations/appendix_judge_calibration.py --figure scatter
"""

from __future__ import annotations

import argparse
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

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.judge_calibration_common import (
    ALL_TRAITS,
    LLM_JUDGE_RUNS,
    SCORE_RANGE,
    discover_human_raters,
    load_golden,
    load_human_scores,
    load_judge_runs_raw,
    load_llm_judge_scores,
)
from src_dev.persona_metrics.judge_calibration import spearman_r
from src_dev.persona_metrics.llm_judge_agreement import _krippendorff_alpha_ordinal

PAPER_FIGURES = [
    "appendix/ocean_evals/fig_F_judge_cross_trait_and_mae.pdf",
    "appendix/ocean_evals/fig_F_judge_scatter.pdf",
    "appendix/ocean_evals/fig_F_judge_agreement_bars.pdf",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PANEL_JUDGES = ["Qwen 3 235B", "Gemma 4 26B-A4B", "Llama 3.3 70B"]
ANNOTATED_TRAITS = ["agreeableness", "neuroticism", "coherence"]
APPENDIX_DIR = PAPER_FIGURES_DIR / "appendix" / "ocean_evals"

TRAIT_LABELS = {
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "coherence": "Coherence",
}

# Heatmap row order; panel members are bold-faced.
JUDGE_DISPLAY_ORDER = [
    "Qwen 3 235B",
    "Gemma 4 26B-A4B",
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

# Distinct colour per panel judge (shared across scatter + agreement-bar figs).
PANEL_JUDGE_COLOURS = {
    "Qwen 3 235B": "#800000",
    "Gemma 4 26B-A4B": "#808000",
    "Llama 3.3 70B": "#9A6324",
}

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def load_all_human_scores(trait: str) -> dict[str, dict[str, float]]:
    """Return ``{rater_id: {item_id: score}}`` for humans who rated *trait*."""
    return {hid: load_human_scores(hid, trait) for hid in discover_human_raters(trait)}


def compute_human_mean(
    humans: dict[str, dict[str, float]], item_ids: list[str]
) -> dict[str, float]:
    """Per-item mean across human raters."""
    means = {}
    for iid in item_ids:
        vals = [h[iid] for h in humans.values() if iid in h]
        if vals:
            means[iid] = statistics.mean(vals)
    return means


def compute_intra_rater_alpha(judge_name: str, trait: str) -> float | None:
    """Intra-rater Krippendorff's α across runs for one judge on *trait*."""
    lo, hi = SCORE_RANGE[trait]
    runs = load_judge_runs_raw(judge_name, trait)
    if not runs:
        return None
    scores_by_id: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for iid, score in run.items():
            scores_by_id[iid].append(score)
    item_ratings = [scores for scores in scores_by_id.values() if len(scores) >= 2]
    if not item_ratings:
        return None
    return _krippendorff_alpha_ordinal(item_ratings, score_min=lo, score_max=hi)


# ---------------------------------------------------------------------------
# Cross-trait heatmaps (ρ vs gold, normalised MAE vs gold)
# ---------------------------------------------------------------------------


def _compute_judge_trait_matrix(metric: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Judge × trait matrix vs gold. metric: "spearman" or "mae_normalised"."""
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
            elif metric == "mae_normalised":
                lo, hi = SCORE_RANGE[trait]
                mae = sum(abs(a - b) for a, b in zip(g, j)) / len(g)
                matrix[ji, ti] = mae / (hi - lo)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return matrix, judges, traits


def _draw_heatmap_panel(
    ax,
    matrix: np.ndarray,
    judges: list[str],
    traits: list[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    reverse_contrast: bool,
    show_yticklabels: bool,
):
    """Draw one judges × traits heatmap (imshow + cell annotations + colourbar).

    ``reverse_contrast`` flips the white/black text rule for metrics where low =
    good (MAE) vs high = good (ρ).
    """
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([TRAIT_LABELS[t] for t in traits], rotation=30, ha="right")
    ax.set_yticks(range(len(judges)))
    if show_yticklabels:
        ax.set_yticklabels(judges)
        for label in ax.get_yticklabels():
            if label.get_text() in PANEL_JUDGES:
                label.set_fontweight("bold")
    else:
        ax.set_yticklabels([])

    midpoint = (vmin + vmax) / 2
    band = (vmax - vmin) * 0.25
    for ji in range(len(judges)):
        for ti in range(len(traits)):
            val = matrix[ji, ti]
            if np.isnan(val):
                ax.text(ti, ji, "—", ha="center", va="center", fontsize=9, color="gray")
                continue
            if reverse_contrast:
                colour = "white" if val > midpoint + band else "black"
            else:
                colour = "white" if val < midpoint - band else "black"
            ax.text(ti, ji, f"{val:.2f}", ha="center", va="center", fontsize=9, color=colour)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)


# Per-metric heatmap styling, shared by the standalone and combined figures.
_RHO_STYLE = dict(cmap="RdYlGn", vmin=0.70, vmax=1.00,
                  cbar_label="Spearman ρ vs gold", reverse_contrast=False)
_MAE_STYLE = dict(cmap="RdYlGn_r", vmin=0.0, vmax=0.35,
                  cbar_label="MAE / scale range (lower = better)", reverse_contrast=True)


def plot_cross_trait_heatmap(output: Path) -> Path:
    """Standalone heatmap: Spearman ρ vs gold for all judges × traits."""
    matrix, judges, traits = _compute_judge_trait_matrix("spearman")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _draw_heatmap_panel(ax, matrix, judges, traits, show_yticklabels=True, **_RHO_STYLE)
    ax.set_title("Cross-trait Spearman ρ of LLM judges vs gold labels\n"
                 "(panel judges in bold)", fontsize=11, pad=10)
    return _save(fig, output)


def plot_mae_heatmap(output: Path) -> Path:
    """Standalone heatmap: normalised MAE vs gold for all judges × traits."""
    matrix, judges, traits = _compute_judge_trait_matrix("mae_normalised")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _draw_heatmap_panel(ax, matrix, judges, traits, show_yticklabels=True, **_MAE_STYLE)
    ax.set_title("Cross-trait normalised MAE of LLM judges vs gold labels\n"
                 "(panel judges in bold; MAE divided by trait scale span)",
                 fontsize=11, pad=10)
    return _save(fig, output)


def plot_cross_trait_and_mae(output: Path) -> Path:
    """Combined figure: (a) Spearman ρ and (b) normalised MAE side-by-side."""
    rho_matrix, judges, traits = _compute_judge_trait_matrix("spearman")
    mae_matrix, _, _ = _compute_judge_trait_matrix("mae_normalised")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5))
    _draw_heatmap_panel(axes[0], rho_matrix, judges, traits, show_yticklabels=True, **_RHO_STYLE)
    axes[0].set_title("(a) Spearman ρ", fontsize=11, pad=10, loc="left")
    _draw_heatmap_panel(axes[1], mae_matrix, judges, traits, show_yticklabels=False, **_MAE_STYLE)
    axes[1].set_title("(b) Normalised MAE", fontsize=11, pad=10, loc="left")
    return _save(fig, output)


# ---------------------------------------------------------------------------
# Scatter: panel judges vs human mean
# ---------------------------------------------------------------------------


def plot_scatter_grid(output: Path) -> Path:
    """3×3 scatter: panel judges (rows) vs human mean, per annotated trait (cols)."""
    nrows, ncols = len(PANEL_JUDGES), len(ANNOTATED_TRAITS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False)
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

            ax.scatter(xs, ys_jittered, color=PANEL_JUDGE_COLOURS[judge], alpha=0.55,
                       s=32, edgecolor="white", linewidth=0.4)
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, alpha=0.5)

            if len(xs) >= 2:
                rho = spearman_r(xs, ys_raw)
                ax.text(0.04, 0.96, f"ρ = {rho:.2f}", transform=ax.transAxes,
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.85))

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

    return _save(fig, output)


# ---------------------------------------------------------------------------
# Agreement bars (inter- and intra-rater)
# ---------------------------------------------------------------------------


def plot_agreement_bars(output: Path) -> Path:
    """(A) inter-rater ρ vs human mean + H-H α; (B) intra-rater α across traits."""
    fig, (ax_inter, ax_intra) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: inter-rater ρ vs human mean ────────────────────────────────
    x = np.arange(len(ANNOTATED_TRAITS))
    width = 0.12
    # Panel judges share their distinct colours; the (typically 3) humans get
    # distinct colours of their own so individual leave-one-out bars are legible.
    human_palette = ["#4363d8", "#911eb4", "#42d4f4", "#f032e6"]

    rows: dict[str, list[float]] = {}
    for judge in PANEL_JUDGES:
        vals = []
        for trait in ANNOTATED_TRAITS:
            humans = load_all_human_scores(trait)
            hmean = compute_human_mean(humans, list(load_golden(trait)))
            judge_scores = load_llm_judge_scores(judge, trait)
            ids = sorted(set(hmean) & set(judge_scores))
            vals.append(spearman_r([hmean[i] for i in ids], [judge_scores[i] for i in ids])
                        if ids else float("nan"))
        rows[judge] = vals

    # Humans: leave-one-out vs median of the other humans.
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
            loo_vals, me_vals = [], []
            for iid in sorted(humans[hid].keys()):
                others_scores = [o[iid] for o in others.values() if iid in o]
                if others_scores:
                    loo_vals.append(statistics.median(others_scores))
                    me_vals.append(humans[hid][iid])
            vals.append(spearman_r(me_vals, loo_vals) if loo_vals else float("nan"))
        rows[hid] = vals

    human_colours = {hid: human_palette[i % len(human_palette)] for i, hid in enumerate(human_ids)}
    all_raters = PANEL_JUDGES + human_ids
    n_raters = len(all_raters)
    for i, rater in enumerate(all_raters):
        offset = (i - (n_raters - 1) / 2) * width
        colour = PANEL_JUDGE_COLOURS.get(rater, human_colours.get(rater))
        bars = ax_inter.bar(x + offset, rows[rater], width, color=colour,
                            edgecolor="white", linewidth=0.5, label=rater)
        for bar, val in zip(bars, rows[rater]):
            if not np.isnan(val):
                ax_inter.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                              f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    # Human-human α reference line per trait.
    for ti, trait in enumerate(ANNOTATED_TRAITS):
        humans = load_all_human_scores(trait)
        names = sorted(humans.keys())
        lo, hi = SCORE_RANGE[trait]
        ratings_per_item = []
        for iid in list(load_golden(trait)):
            rats = [int(humans[n][iid]) for n in names if iid in humans[n]]
            if rats:
                ratings_per_item.append(rats)
        alpha = _krippendorff_alpha_ordinal(ratings_per_item, score_min=lo, score_max=hi)
        ax_inter.hlines(alpha, ti - 0.5, ti + 0.5, colors="black", linestyles="--",
                        linewidth=1.5, label="H-H α" if ti == 0 else None)

    ax_inter.set_xticks(x)
    ax_inter.set_xticklabels([TRAIT_LABELS[t] for t in ANNOTATED_TRAITS])
    ax_inter.set_ylabel("Spearman ρ vs human mean / H-H α", fontsize=10)
    ax_inter.set_ylim(0, 1.05)
    ax_inter.set_title("(A) Inter-rater agreement", fontsize=11)
    ax_inter.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)
    ax_inter.grid(axis="y", alpha=0.3)
    ax_inter.spines[["top", "right"]].set_visible(False)

    # ── Panel B: intra-rater α across all traits ────────────────────────────
    x_all = np.arange(len(ALL_TRAITS))
    width_b = 0.25
    for i, judge in enumerate(PANEL_JUDGES):
        vals = [compute_intra_rater_alpha(judge, t) for t in ALL_TRAITS]
        vals = [v if v is not None else float("nan") for v in vals]
        offset = (i - 1) * width_b
        bars = ax_intra.bar(x_all + offset, vals, width_b, color=PANEL_JUDGE_COLOURS[judge],
                            edgecolor="white", linewidth=0.5, label=judge)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax_intra.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                              f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax_intra.set_xticks(x_all)
    ax_intra.set_xticklabels([TRAIT_LABELS[t][:5] for t in ALL_TRAITS], rotation=30, ha="right")
    ax_intra.set_ylabel("Intra-rater Krippendorff's α", fontsize=10)
    ax_intra.set_ylim(0.0, 1.05)
    ax_intra.set_title("(B) Self-consistency (temp=0.7, 3 runs)", fontsize=11)
    ax_intra.legend(fontsize=7, loc="lower right")
    ax_intra.grid(axis="y", alpha=0.3)
    ax_intra.spines[["top", "right"]].set_visible(False)

    return _save(fig, output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _save(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


FIGURES: dict[str, tuple[str, callable]] = {
    "cross_trait": ("fig_F_judge_cross_trait.pdf", plot_cross_trait_heatmap),
    "mae_heatmap": ("fig_F_judge_mae_heatmap.pdf", plot_mae_heatmap),
    "cross_trait_and_mae": ("fig_F_judge_cross_trait_and_mae.pdf", plot_cross_trait_and_mae),
    "scatter": ("fig_F_judge_scatter.pdf", plot_scatter_grid),
    "agreement_bars": ("fig_F_judge_agreement_bars.pdf", plot_agreement_bars),
}

# Figures actually included in the paper (subset of FIGURES).
PAPER_FIGURE_KEYS = ["cross_trait_and_mae", "scatter", "agreement_bars"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=list(FIGURES) + ["all", "paper"], default="paper",
                        help="Which figure(s) to generate (default: paper = the 3 included).")
    parser.add_argument("--output-dir", type=Path, default=APPENDIX_DIR,
                        help=f"Output directory (default: {APPENDIX_DIR}).")
    args = parser.parse_args()

    if args.figure == "all":
        names = list(FIGURES)
    elif args.figure == "paper":
        names = PAPER_FIGURE_KEYS
    else:
        names = [args.figure]

    for name in names:
        filename, fn = FIGURES[name]
        fn(args.output_dir / filename)


if __name__ == "__main__":
    main()

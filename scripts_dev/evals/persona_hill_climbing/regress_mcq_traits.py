"""OLS of soup misalignment on per-direction OCEAN doses, with coefficient plot.

Fits average misalignment ~ intercept + sum over the 10 trait directions
(O+, O−, ..., N−) of rate × |coefficient|, per model, on trust-gated
conditions (answer rate > 0.7, best replicate per condition). The
asymmetric split captures that amplifier and suppressor doses of the same
trait need not have opposite effects — empirically almost every direction
*raises* misalignment, at direction-specific rates.

Data (registered on the HF monorepo ``persona-cartography/monorepo``):
    Reads the per-condition stats JSONs
        evals/persona_hill_climbing/analysis/mcq_lp_md_trait_v1/{gemma,qwen}_full_{tb,art}_stats.json
    (scored from evals/persona_hill_climbing/{model}/mcqfull_{tb,art}_{set}_train/responses/).
    Rendered outputs (rate heatmap, LOO-CV scatter) are registered in the same
    analysis dir. Per-topic-category rates use ``plot_mcq_cluster_heatmap.py``.

Usage::

    uv run python -m scripts_dev.evals.persona_hill_climbing.regress_mcq_traits \
        --stats gemma-3-27b-it=scratch/gemma_md_stats_trait.json \
                qwen-3-32b-it=scratch/qwen_md_stats_trait.json \
        --out scratch/plots/trait_direction_rates.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from scripts_dev.evals.persona_hill_climbing.plot_mcq_logprob_overview import (
    BASELINE, BLUE, GRID, INK, INK_2, MUTED, RED, SURFACE,
)

TOKEN = re.compile(r"([ocean])_(plus|minus)_([0-9.]+)")
TRAITS = "ocean"
DIRECTIONS = [f"{t.upper()}+" for t in TRAITS] + [f"{t.upper()}−" for t in TRAITS]


def coeffs(cond: str) -> dict[str, float]:
    d = {t: 0.0 for t in TRAITS}
    for t, s, v in TOKEN.findall(cond):
        d[t] = float(v) * (1 if s == "plus" else -1)
    return d


def best(recs: list[dict], min_ans: float) -> dict[str, float]:
    out: dict[str, tuple[float, float]] = {}
    for r in recs:
        g = r["dynamic"]
        if g["mis"] is None or g["answered"] <= min_ans:
            continue
        prev = out.get(r["condition"])
        if prev is None or g["answered"] > prev[1]:
            out[r["condition"]] = (g["mis"], g["answered"])
    return {c: v[0] for c, v in out.items()}


def fit(recs: list[dict], min_ans: float):
    data = best(recs, min_ans)
    conds = [c for c in data if c != "vanilla"]
    y = np.array([data[c] for c in conds])
    C = np.array([[coeffs(c)[t] for t in TRAITS] for c in conds])
    X = np.column_stack([np.maximum(C, 0), np.maximum(-C, 0)])
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X1.T @ X1)
    half = sps.t.ppf(0.975, dof) * np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, half, r2, len(y)


def fit_full(recs: list[dict], min_ans: float):
    """Return (conds, X, y, beta, half95, pvals, loo_pred, loo_r2)."""
    data = best(recs, min_ans)
    conds = [c for c in data if c != "vanilla"]
    y = np.array([data[c] for c in conds])
    C = np.array([[coeffs(c)[t] for t in TRAITS] for c in conds])
    X = np.column_stack([np.maximum(C, 0), np.maximum(-C, 0)])
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    pvals = 2 * sps.t.sf(np.abs(beta / se), dof)
    half = sps.t.ppf(0.975, dof) * se
    # Leave-one-out predictions via the hat matrix (PRESS residuals).
    H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    loo_resid = (y - H @ y) / (1 - np.diag(H))
    loo_pred = y - loo_resid
    loo_r2 = 1 - (loo_resid @ loo_resid) / ((y - y.mean()) @ (y - y.mean()))
    return conds, X, y, beta, half, pvals, loo_pred, loo_r2


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def heatmap_figure(fits: list, out: Path) -> None:
    """Directions x models heatmap of per-dose misalignment rates."""
    from matplotlib.colors import TwoSlopeNorm

    labels = [f[0] for f in fits]
    rates = np.array([f[4][1:] for f in fits]).T   # (10 directions, n_models)
    pv = np.array([f[6][1:] for f in fits]).T
    fig, ax = plt.subplots(figsize=(5.6, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vmax = float(np.abs(rates).max())
    im = ax.imshow(rates, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                   aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, color=INK_2)
    ax.set_yticks(range(len(DIRECTIONS)))
    ax.set_yticklabels(DIRECTIONS, fontsize=11, color=INK_2)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            sig = stars(pv[i, j])
            txt = f"{rates[i, j]:+.2f}{sig}"
            dark = abs(rates[i, j]) > 0.55 * vmax
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color=SURFACE if dark else INK,
                    fontweight="bold" if sig else "normal")
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("misalignment added per unit dose", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(
        "Misalignment rate per persona direction\n"
        "red = hurts alignment · * p<0.05 · ** p<0.01 · *** p<0.001",
        fontsize=11, color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def loo_figure(fits: list, out: Path) -> None:
    """Leave-one-out predicted vs actual misalignment per condition."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = 0.8
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1, ls="--", zorder=1)
    colors = [RED, BLUE]
    for i, f in enumerate(fits):
        label, y, loo_pred, loo_r2 = f[0], f[3], f[7], f[8]
        ax.scatter(y, loo_pred, s=42, color=colors[i], alpha=0.85, zorder=3,
                   edgecolors=SURFACE, linewidths=0.7,
                   label=f"{label}  (LOO R² = {loo_r2:.2f}, n={len(y)})")
    ax.set_xlabel("actual average misalignment", color=INK_2, fontsize=10)
    ax.set_ylabel("leave-one-out predicted", color=INK_2, fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, labelcolor=INK_2)
    ax.set_title(
        "Out-of-sample check: leave-one-out cross-validation\n"
        "each soup predicted by a 10-direction dose model fit without it",
        fontsize=11, color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", nargs="+", required=True,
                        help="label=stats_json pairs")
    parser.add_argument("--min-answered", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("scratch/plots/trait_direction_rates.png"))
    parser.add_argument("--heatmap-out", type=Path, default=Path("scratch/plots/trait_rates_heatmap.png"))
    parser.add_argument("--loo-out", type=Path, default=Path("scratch/plots/trait_rates_loo.png"))
    args = parser.parse_args()

    models = []
    fits = []
    for spec in args.stats:
        label, path = spec.split("=", 1)
        recs = json.loads(Path(path).read_text())
        models.append((label, *fit(recs, args.min_answered)))
        fits.append((label, *fit_full(recs, args.min_answered)))

    heatmap_figure(fits, args.heatmap_out)
    loo_figure(fits, args.loo_out)

    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ys = np.arange(len(DIRECTIONS))[::-1]
    colors = [RED, BLUE]
    off = [-0.17, 0.17]
    for i, (label, beta, half, r2, n) in enumerate(models):
        b, h = beta[1:], half[1:]
        ax.errorbar(b, ys + off[i], xerr=h, fmt="o", color=colors[i],
                    ms=7, elinewidth=1.4, capsize=3,
                    label=f"{label}  (R²={r2:.2f}, n={n})", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(DIRECTIONS, fontsize=11, color=INK_2)
    ax.set_xlabel("misalignment added per unit dose of the direction (± 95% CI)",
                  color=INK_2, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False, labelcolor=INK_2)
    ax.set_title(
        "Which persona directions hurt alignment?\n"
        "OLS: average misalignment ~ Σ rate × dose over the 10 trait directions "
        "(80 soups, trust-gated)",
        fontsize=11, color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

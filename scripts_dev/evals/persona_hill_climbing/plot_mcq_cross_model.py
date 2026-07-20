"""Cross-model comparison of persona-soup misalignment (md logprob scoring).

Scatter of mean P(misaligned) for the same soup conditions evaluated on two
base models (same items, same seeds, same scoring). Both readings must pass
the trust gate (answered > --min-answered) to appear. Diagonal = perfect
cross-model agreement.

Data (registered on the HF monorepo ``persona-cartography/monorepo``):
    Reads the same per-condition stats JSONs as ``plot_mcq_logprob_overview``
    (``{gemma,qwen}_full_{tb,art}_stats.json``); the rendered scatter is
    registered as ``cross_model_full_tb.png``, all under
        evals/persona_hill_climbing/analysis/mcq_lp_md_trait_v1/
    Raw responses: evals/persona_hill_climbing/{model}/mcqfull_{tb,art}_{set}_train/.

Usage::

    uv run python -m scripts_dev.evals.persona_hill_climbing.plot_mcq_cross_model \
        --stats-a scratch/gemma_md_stats_textbook.json --label-a gemma-3-27b-it \
        --stats-b scratch/qwen_md_stats_textbook.json  --label-b qwen-3-32b-it \
        --out scratch/plots/cross_model_md.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts_dev.evals.persona_hill_climbing.plot_mcq_logprob_overview import (
    BASELINE, BLUE, GRAY, GRID, INK, INK_2, MUTED, RED, SURFACE,
    a_color, compact_label, parse_coeffs,
)


def best_reading(recs: list[dict], min_answered: float) -> dict[str, dict]:
    """condition -> trust-gated record with the highest answered rate."""
    out: dict[str, dict] = {}
    for r in recs:
        g = r["dynamic"]
        if g["mis"] is None or g["answered"] <= min_answered:
            continue
        prev = out.get(r["condition"])
        if prev is None or g["answered"] > prev["dynamic"]["answered"]:
            out[r["condition"]] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-a", type=Path, required=True)
    parser.add_argument("--stats-b", type=Path, required=True)
    parser.add_argument("--label-a", default="model A")
    parser.add_argument("--label-b", default="model B")
    parser.add_argument("--min-answered", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("scratch/plots/cross_model_md.png"))
    args = parser.parse_args()

    a = best_reading(json.loads(args.stats_a.read_text()), args.min_answered)
    b = best_reading(json.loads(args.stats_b.read_text()), args.min_answered)
    common = sorted((set(a) & set(b)) - {"vanilla"})

    fig, ax = plt.subplots(figsize=(9.5, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lim = 0.78
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1, ls="--", zorder=1)

    van_a = a.get("vanilla")
    van_b = b.get("vanilla")
    if van_a and van_b:
        ax.scatter(van_a["dynamic"]["mis"], van_b["dynamic"]["mis"], marker="*",
                   s=260, color=INK, zorder=5, edgecolors=SURFACE, linewidths=0.8)
        ax.annotate("vanilla", (van_a["dynamic"]["mis"], van_b["dynamic"]["mis"]),
                    xytext=(7, -3), textcoords="offset points", fontsize=9, color=INK_2)

    pts = []
    for cond in common:
        ga, gb = a[cond]["dynamic"], b[cond]["dynamic"]
        color = a_color(parse_coeffs(cond))
        ax.errorbar(ga["mis"], gb["mis"],
                    xerr=[[ga["mis"] - ga["ci_lo"]], [ga["ci_hi"] - ga["mis"]]],
                    yerr=[[gb["mis"] - gb["ci_lo"]], [gb["ci_hi"] - gb["mis"]]],
                    fmt="none", ecolor=color, elinewidth=0.9, alpha=0.35, zorder=2)
        ax.scatter(ga["mis"], gb["mis"], s=46, color=color, zorder=4,
                   edgecolors=SURFACE, linewidths=0.8)
        pts.append((cond, ga["mis"], gb["mis"]))

    # Label extremes on either axis and the biggest disagreements.
    by_sum = sorted(pts, key=lambda p: p[1] + p[2])
    by_gap = sorted(pts, key=lambda p: abs(p[1] - p[2]))
    to_label = {p[0] for p in by_sum[-4:]} | {p[0] for p in by_sum[:2]} | {p[0] for p in by_gap[-3:]}
    for cond, xa, yb in pts:
        if cond in to_label:
            ax.annotate(compact_label(cond), (xa, yb), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5, color=INK_2)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=RED, label="contains A−"),
        Patch(facecolor=BLUE, label="contains A+"),
        Patch(facecolor=GRAY, label="no A component"),
        Line2D([], [], ls="--", color=BASELINE, label="y = x (perfect agreement)"),
        Line2D([], [], marker="*", ls="", color=INK, markersize=12, label="vanilla"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, labelcolor=INK_2)

    if len(pts) > 2:
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        rho = float(np.corrcoef(xs, ys)[0, 1])
    else:
        rho = float("nan")

    ax.set_xlabel(f"mean P(misaligned) · {args.label_a}", color=INK_2, fontsize=10)
    ax.set_ylabel(f"mean P(misaligned) · {args.label_b}", color=INK_2, fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(
        "Cross-model persona-soup misalignment · discourse-grounded MCQ (textbook, "
        "300 train items)\nmarkdown-tolerant logprob scoring · trust-gated both models "
        f"(answered > {args.min_answered}) · n={len(pts)} shared conditions · r={rho:.2f}",
        fontsize=10.5, color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out} ({len(pts)} shared conditions, r={rho:.2f})")


if __name__ == "__main__":
    main()

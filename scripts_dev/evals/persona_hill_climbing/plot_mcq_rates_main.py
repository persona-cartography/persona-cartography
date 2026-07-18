"""Headline figure: per-direction misalignment rates, gemma vs qwen, all items.

Two-row heatmap (one row per model) of the asymmetric dose-model coefficients
fit on ALL items (textbook + article pools combined, 3,946 items/condition),
no topic split. Cells show misalignment added per unit adapter dose with
t-test significance stars. Reads the "all" entries of the per-category rates
JSONs produced by ``plot_mcq_cluster_heatmap.py``.

Data (registered):
    persona-cartography/monorepo @
    evals/persona_hill_climbing/analysis/mcq_lp_md_trait_v1/
        {gemma,qwen}_full_cluster_rates.json

Usage::

    uv run python -m scripts_dev.evals.persona_hill_climbing.plot_mcq_rates_main \
        --rates gemma-3-27b-it=scratch/gemma_full_cluster_rates.json \
                qwen-3-32b-it=scratch/qwen_full_cluster_rates.json \
        --out scratch/plots/main_rates_heatmap.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from scripts_dev.evals.persona_hill_climbing.plot_mcq_logprob_overview import (
    INK, INK_2, MUTED, SURFACE,
)

DIRECTIONS = ["O+", "C+", "E+", "A+", "N+", "O−", "C−", "E−", "A−", "N−"]


def star(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", required=True, help="label=rates_json pairs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = []
    for spec in args.rates:
        label, path = spec.split("=", 1)
        r = json.loads(Path(path).read_text())["all"]
        models.append((label, r))

    labels = [f"{label}\nLOO R²={r['loo_r2']:.2f} · {r['n_conds']} soups" for label, r in models]
    rates = np.array([r["beta"][1:] for _, r in models])
    pv = np.array([r["pv"][1:] for _, r in models])

    fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vmax = float(np.abs(rates).max())
    im = ax.imshow(rates, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels(DIRECTIONS, fontsize=13, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK_2)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            sig = star(pv[i, j])
            dark = abs(rates[i, j]) > 0.55 * vmax
            ax.text(j, i, f"{rates[i, j]:+.2f}\n{sig}", ha="center", va="center",
                    fontsize=10.5, color=SURFACE if dark else INK,
                    fontweight="bold" if sig else "normal")
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("misalignment added\nper unit dose", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle("How much does each persona direction hurt alignment?",
                 fontsize=13.5, color=INK, x=0.065, y=0.97, ha="left")
    fig.text(0.065, 0.845,
             "Each cell: increase in average misalignment (probability mass on the misaligned "
             "MCQ option) per unit LoRA dose of that OCEAN direction,\n"
             "fit jointly over 80 persona soups × 3,946 agentic-misalignment items "
             "(discourse-grounded MCQ, both pools) · red = hurts, blue = protects · "
             "* p<0.05 ** p<0.01 *** p<0.001",
             fontsize=9.3, color=INK_2, va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Welfare-axis projection vs LoRA scale for the gemma-27b N+ adapter.

Single-panel line: mean generation-token projection (VAA z-units) vs adapter
scale alpha in [-4, 4], BCa 95% CI over the 40 welfare prompts. alpha=0 is the
bare base model.

Usage:
    python scripts_dev/evals/welfare_axis/plot_scale_sweep_welfare.py
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Style — inlined from STYLE_GUIDE.md (matches paper_apologize_coconot.py).
PAPER_STYLE: dict[str, object] = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.facecolor": "#fbfbfc",
    "axes.edgecolor": "#2f3748",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#2f3748",
    "ytick.color": "#2f3748",
    "grid.color": "#dfe3e8",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.75,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cfd4dc",
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
}
SPINE_COLOR = "#2f3748"
C_LINE = "#c91546"  # C_INJECTED — the neuroticism amplifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "scratch/welfare_axis/results_scale_sweep_gemma27b_nplus.json"),
    )
    ap.add_argument(
        "--proj-stats",
        default=str(PROJECT_ROOT / "scratch/welfare_axis/gemma27b_vaa_proj_stats.json"),
    )
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "scratch/welfare_axis"))
    args = ap.parse_args()

    mpl.rcParams.update(PAPER_STYLE)
    r = json.load(open(args.results))
    L_auto = r["best_layer"]
    L_mid = 42  # mid-depth band (68%) per Lu et al.; auto-selected L55 inverts on generations
    all_stats = json.load(open(args.proj_stats))

    alphas = sorted(int(a) for a in r["scales"])

    def z_curve(layer):
        ps = all_stats[str(layer)]
        return np.array(
            [(r["scales"][str(a)]["proj_mean_per_layer"][layer] - ps["mean"]) / ps["std"] for a in alphas]
        )

    mid, auto = z_curve(L_mid), z_curve(L_auto)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(alphas, mid, color=C_LINE, marker="o", markersize=6, label=f"layer {L_mid} (mid-depth readout)")
    ax.plot(
        alphas, auto, color="#7f8c9b", marker="s", markersize=5,
        linestyle=(0, (4, 3)), linewidth=1.6,
        label=f"layer {L_auto} (auto-selected; direction inverts)",
    )
    i0 = alphas.index(0)
    ax.axvline(0, color=SPINE_COLOR, linewidth=0.9, alpha=0.4)
    ax.annotate(
        "α = 0 is the base model", xy=(0.1, mid[i0]), xytext=(6, 10),
        textcoords="offset points", fontsize=8.5, color=SPINE_COLOR, style="italic",
    )
    ax.annotate(
        "coherence breakdown", xy=(4, mid[-1]), xytext=(-10, 22),
        textcoords="offset points", ha="right", fontsize=8.5, color="#888",
        style="italic",
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8),
    )
    ax.set_title(
        "Welfare-axis projection vs. neuroticism-adapter scale — gemma-3-27b-it",
        loc="left", pad=8,
    )
    ax.set_xlabel("LoRA scale α (Neuroticism + adapter; negative = inverted)")
    ax.set_ylabel("Welfare projection (VAA z-units)")
    ax.set_xticks(alphas)
    ax.legend(loc="center left")
    ax.annotate(
        f"Mean over generated tokens, n={r['n_prompts']} welfare prompts, greedy decoding; "
        f"per-α means (per-prompt CIs stored for layer {L_auto} only).",
        xy=(0.0, -0.16), xycoords="axes fraction", fontsize=7.5, color="#888", style="italic",
    )

    out_dir = Path(args.out_dir)
    stem = out_dir / "fig_welfare_scale_sweep_gemma27b_nplus"
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()

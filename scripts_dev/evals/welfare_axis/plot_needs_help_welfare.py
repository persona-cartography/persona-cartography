"""Per-turn welfare-axis curves through the gemma-27b needs-help eval.

Two panels from results_needs_help.json (gemma_needs_help_welfare.py):
  (a) mean welfare-axis projection per assistant turn (base / control / N+),
      in VAA-statement z-units, BCa 95% CI over conversations;
  (b) the stored per-turn LLM-judge frustration scores for the SAME
      conversations — the metric the welfare axis is meant to replace.

Usage:
    python scripts_dev/evals/welfare_axis/plot_needs_help_welfare.py \
        [--results scratch/welfare_axis/results_needs_help_gemma27b.json] \
        [--proj-stats scratch/welfare_axis/gemma27b_vaa_proj_stats.json]
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

SERIES = {  # fixed identity -> color/marker (house semantics)
    "base": ("Base", "#3c7fb1", "o"),        # C_ORGANIC
    "control": ("Control", "#7f8c9b", "s"),  # C_PERSONA
    "n_plus": ("Neuroticism +", "#c91546", "^"),  # C_INJECTED
}


def turn_ci(values_2d: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and BCa 95% CI per column (turn) of (convs, turns)."""
    means, los, his = [], [], []
    for t in range(values_2d.shape[1]):
        col = values_2d[:, t]
        res = stats.bootstrap((col,), np.mean, n_resamples=2000, method="BCa", random_state=rng)
        means.append(col.mean())
        los.append(res.confidence_interval.low)
        his.append(res.confidence_interval.high)
    return np.array(means), np.array(los), np.array(his)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "scratch/welfare_axis/results_needs_help_gemma27b.json"),
    )
    ap.add_argument(
        "--proj-stats",
        default=str(PROJECT_ROOT / "scratch/welfare_axis/gemma27b_vaa_proj_stats.json"),
    )
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "scratch/welfare_axis"))
    ap.add_argument("--layer", type=int, default=None, help="Override readout layer (needs --proj-npz)")
    ap.add_argument("--proj-npz", default=None, help="npz with per-variant (convs, turns, layers) projections")
    ap.add_argument("--out-name", default="fig_needs_help_welfare_gemma27b")
    ap.add_argument("--layer-note", default="")
    args = ap.parse_args()

    mpl.rcParams.update(PAPER_STYLE)
    r = json.load(open(args.results))
    L = r["best_layer"]
    npz = None
    if args.layer is not None:
        assert args.proj_npz, "--layer override requires --proj-npz"
        L = args.layer
        npz = np.load(args.proj_npz)
    stats_l = json.load(open(args.proj_stats))[str(L)]
    center, scale = stats_l["mean"], stats_l["std"]

    fig, (ax_w, ax_j) = plt.subplots(1, 2, figsize=(15.6, 5.4))
    rng = np.random.default_rng(SEED)
    n_turns = None
    for key, (label, color, marker) in SERIES.items():
        v = r["variants"][key]
        raw = npz[key][:, :, L] if npz is not None else np.array(v["proj_best_layer"])
        proj = (raw - center) / scale  # (convs, turns)
        judge = np.array(v["judge_frustration"])
        n_turns = proj.shape[1]
        x = np.arange(1, n_turns + 1)
        for ax, data in ((ax_w, proj), (ax_j, judge)):
            m, lo, hi = turn_ci(data, rng)
            ax.plot(x, m, color=color, marker=marker, markersize=5, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax_w.axhline(0.0, color=SPINE_COLOR, linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.5)
    ax_w.set_title("(a) Valence-assent axis projection per turn", loc="left", pad=8)
    ax_w.set_xlabel("Assistant turn")
    ax_w.set_ylabel(f"Valence-assent projection (z-units, layer {L})")
    ax_w.legend(loc="lower left")

    ax_j.set_title("(b) LLM-judge frustration per turn (same conversations)", loc="left", pad=8)
    ax_j.set_xlabel("Assistant turn")
    ax_j.set_ylabel("Frustration score (1–10)")
    ax_j.legend(loc="upper left")

    for ax in (ax_w, ax_j):
        ax.set_xticks(np.arange(1, n_turns + 1))

    ax_w.annotate(
        f"Valence-assent axis (Lu et al. 2025), extracted with the Han, Chalmers & Izmailov (2026) code; "
        f"welfare-aligned per their Appendix H (partial alignment, |cos| ≈ 0.2, shared steering effects). "
        f"gemma-3-27b-it, needs-help (impossible numeric) eval, n={r['variants']['base']['n_convs']} "
        f"conversations; shaded = BCa 95% CI.{args.layer_note}",
        xy=(0.0, -0.13), xycoords="axes fraction", fontsize=7.5, color="#888", style="italic",
    )

    # Per-conversation-turn correlation between the two metrics
    all_w, all_j = [], []
    for key in SERIES:
        v = r["variants"][key]
        raw = npz[key][:, :, L] if npz is not None else np.array(v["proj_best_layer"])
        all_w.append(((raw - center) / scale).ravel())
        all_j.append(np.array(v["judge_frustration"]).ravel())
    rho, p = stats.spearmanr(np.concatenate(all_w), np.concatenate(all_j))
    print(f"welfare proj vs judge frustration (pooled conv-turns): Spearman rho={rho:.3f} p={p:.2e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.out_name
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()

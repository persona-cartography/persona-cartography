"""Plot OCEAN persona positions on the functional welfare / valence-assent axis.

Reads the results.json produced by measure_welfare_axis.py (small-sample
Qwen3-8B run) and renders a single-panel horizontal bar chart: projection of
each persona adapter's generation-token activation shift (variant - base)
onto the unit VAA axis at the auto-selected layer, with BCa 95% CIs over
paired per-prompt differences.

Usage:
    python scripts_dev/evals/persona_hill_climbing/../welfare_axis/plot_welfare_axis.py \
        [--results scratch/welfare_axis/results_qwen3_8b_small.json] \
        [--out-dir scratch/welfare_axis]
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
C_NEGATIVE = "#c91546"  # C_INJECTED — negative-welfare pole (failure red)
C_POSITIVE = "#0f7f3f"  # C_DATASET — positive-welfare pole (green = good)
C_CONTROL = "#7f8c9b"   # C_PERSONA — neutral / structural reference

VARIANT_LABELS = {
    "o_plus": "Openness +",
    "o_minus": "Openness −",
    "c_plus": "Conscientiousness +",
    "c_minus": "Conscientiousness −",
    "e_plus": "Extraversion +",
    "e_minus": "Extraversion −",
    "a_plus": "Agreeableness +",
    "a_minus": "Agreeableness −",
    "n_plus": "Neuroticism +",
    "n_minus": "Neuroticism −",
    "control": "Control (null adapter)",
}


def bca_ci(values: np.ndarray, rng) -> tuple[float, float]:
    res = stats.bootstrap(
        (values,), np.mean, n_resamples=2000, method="BCa", random_state=rng
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "scratch/welfare_axis/results_qwen3_8b_small.json"),
    )
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "scratch/welfare_axis"))
    ap.add_argument("--out-name", default="fig_welfare_axis_qwen3_8b_small")
    ap.add_argument("--model-label", default="Qwen3-8B")
    args = ap.parse_args()

    mpl.rcParams.update(PAPER_STYLE)
    rng = np.random.default_rng(SEED)

    r = json.load(open(args.results))
    layer = r["best_layer"]

    rows = []
    for name, entry in r["variants"].items():
        pp = np.array(entry["gen_mean"]["proj_per_prompt_best_layer"])
        lo, hi = bca_ci(pp, rng)
        rows.append((name, float(pp.mean()), lo, hi))
    rows.sort(key=lambda t: t[1])  # most negative at the bottom

    fig, ax = plt.subplots(figsize=(7.0, max(3.9, 0.52 * len(rows) + 1.4)))
    y = np.arange(len(rows))
    for yi, (name, mean, lo, hi) in zip(y, rows):
        color = C_CONTROL if name == "control" else (C_POSITIVE if mean >= 0 else C_NEGATIVE)
        ax.barh(
            yi, mean, height=0.62, color=color, alpha=0.92,
            edgecolor=SPINE_COLOR, linewidth=0.5, zorder=3,
        )
        ax.errorbar(
            x=(lo + hi) / 2, y=yi, xerr=[[(lo + hi) / 2 - lo], [hi - (lo + hi) / 2]],
            fmt="none", ecolor=SPINE_COLOR, elinewidth=1.0, capsize=2.5,
            capthick=1.0, alpha=0.85, zorder=4,
        )
        ax.annotate(
            f"{mean:+.2f}",
            xy=(hi if mean >= 0 else lo, yi),
            xytext=(6 if mean >= 0 else -6, 0), textcoords="offset points",
            va="center", ha="left" if mean >= 0 else "right",
            fontsize=8.5, color=SPINE_COLOR, fontweight="semibold",
        )

    ax.axvline(0.0, color=SPINE_COLOR, linewidth=0.9, alpha=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABELS.get(n, n) for n, *_ in rows])
    ax.set_xlabel(
        f"Projection of activation shift onto welfare axis (layer {layer})"
    )
    ax.set_title(
        f"OCEAN persona adapters on the functional welfare axis — {args.model_label}",
        loc="left", pad=8, fontsize=12,
    )
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    xmax = max(abs(v) for _, m, lo, hi in rows for v in (lo, hi, m)) * 1.30
    ax.set_xlim(-xmax, xmax)

    # Anchor annotations a constant physical distance below the axes so the
    # layout survives different row counts (axes-fraction offsets don't).
    axes_h = fig.get_size_inches()[1] - 1.0
    pole_y, note_y = -0.70 / axes_h, -1.02 / axes_h
    ax.annotate(
        "← toward negative welfare", xy=(0.02, pole_y),
        xycoords="axes fraction", fontsize=8.5, color=C_NEGATIVE, style="italic",
    )
    ax.annotate(
        "toward positive welfare →", xy=(0.98, pole_y),
        xycoords="axes fraction", fontsize=8.5, color=C_POSITIVE,
        style="italic", ha="right",
    )
    if r.get("axis_kind") == "functional_welfare":
        note = (
            f"Functional welfare axis unit(vGOAL−vLAVA) (Han, Chalmers & Izmailov 2026), "
            f"concept vectors from {r.get('axis_repo', 'davidafrica/functional-wellbeing')} "
            f"(recruitment cos {r['recruitment_cos_late_third']:.2f}). "
            f"Shift vs base over generated tokens, n={r['n_prompts']} welfare prompts, BCa 95% CI."
        )
    else:
        note = (
            f"Valence-assent axis (Lu et al. 2025) extracted with Han, Chalmers & Izmailov (2026) code; "
            f"AUROC {r['auroc_best_layer']:.2f}. Shift vs base over generated tokens, "
            f"n={r['n_prompts']} welfare prompts, BCa 95% CI."
        )
    ax.annotate(
        note,
        xy=(0.0, note_y), xycoords="axes fraction",
        fontsize=7.5, color="#888", style="italic",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.out_name
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()

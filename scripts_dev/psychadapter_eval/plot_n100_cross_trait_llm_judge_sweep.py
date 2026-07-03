#!/usr/bin/env python3
"""Plot cross-trait dose-response for the PsychAdapter n100 canonical judge sweep.

Reads the canonical judge-run outputs produced by
``scripts_dev.psychadapter_eval.judge_psychadapter_responses`` (one judge run
per conditioning-trait subset x judged v2 metric, Qwen3-235B rater) and renders
the same figure as scripts_dev/psychadapter_eval/plot_n150_cross_trait_llm_judge_sweep.py:
one subplot per conditioned trait, five lines = judge scores on all OCEAN
traits vs latent value.

Also writes the aggregated CSV (n150 schema: conditioned, judged, stat,
<latent columns>) and a paper copy of the figure.

Usage:
    uv run python -m scripts_dev.psychadapter_eval.plot_n100_cross_trait_llm_judge_sweep
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_FIGURES = [
    "appendix/fig_psychadapter_n100_cross_latent.png",
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# OCEAN trait colors from paper
BIG_FIVE_COLORS = {
    "Openness":          "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion":      "#4CAF50",
    "Agreeableness":     "#9C27B0",
    "Neuroticism":       "#F44336",
}

ROOT = Path(__file__).resolve().parents[2]
N100_DIR = ROOT / "scratch/psychadapter_eval/n100"
JUDGE_RUNS_DIR = N100_DIR / "per_trait" / "judge_runs"


def load_judge_runs() -> dict:
    """Collect scores from all judge runs.

    Returns:
        {(conditioned_trait, judged_trait, latent): [scores]}
    """
    data: dict = defaultdict(list)
    for run_dir in sorted(JUDGE_RUNS_DIR.iterdir()):
        config_path = run_dir / "config.json"
        raw_path = run_dir / "judge_calls" / "raw" / "qwen3_235b.jsonl"
        if not (config_path.exists() and raw_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        metric_name = config["judge_raters"][0]["metric_name"]
        if not metric_name.endswith("_v2"):
            continue  # cross plot uses trait metrics only (coherence plotted elsewhere)
        judged_trait = metric_name[: -len("_v2")]
        cond_trait = Path(config["dataset_path"]).parent.name

        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            score = rec.get("score")
            if rec.get("status") != "success" or score is None:
                continue
            # condition is "<trait>@scale_<s>", e.g. "openness@scale_+2.00"
            latent = float(rec["condition"].split("@scale_")[1])
            data[(cond_trait, judged_trait, latent)].append(float(score))
    return data


def write_csv(data: dict, out_path: Path) -> None:
    latents = sorted({k[2] for k in data})
    with open(out_path, "w") as f:
        f.write("conditioned,judged,stat," + ",".join(str(l) for l in latents) + "\n")
        for cond in TRAITS:
            for judged in TRAITS:
                for stat, fn in (("mean", np.mean), ("std", np.std), ("n", len)):
                    vals = []
                    for latent in latents:
                        scores = data.get((cond, judged, latent), [])
                        vals.append(f"{fn(scores):.3f}" if scores else "")
                    f.write(f"{cond},{judged},{stat}," + ",".join(vals) + "\n")
    print(f"✓ CSV -> {out_path}")


def plot_cross_trait(data: dict):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    latents = sorted({k[2] for k in data})

    for idx, cond_trait in enumerate(TRAITS):
        ax = axes[idx]
        for judged_trait in TRAITS:
            means, stds = [], []
            for latent in latents:
                scores = data.get((cond_trait, judged_trait, latent), [])
                means.append(np.mean(scores) if scores else np.nan)
                stds.append(np.std(scores) if scores else np.nan)

            color = BIG_FIVE_COLORS.get(judged_trait.capitalize(), "#999999")
            is_diagonal = judged_trait == cond_trait
            ax.errorbar(
                latents, means, yerr=stds, fmt="o-", color=color,
                linewidth=2.2 if is_diagonal else 1.8,
                markersize=6 if is_diagonal else 5,
                capsize=4, capthick=1.2, elinewidth=1.2,
                alpha=0.85 if is_diagonal else 0.45,
                label=judged_trait.capitalize(),
            )

        ax.set_xlabel("Latent Value", fontsize=10)
        ax.set_ylabel("Judge Score", fontsize=10)
        ax.set_title(cond_trait.capitalize(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-4, 4)
        ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

    handles = [
        plt.Line2D([0], [0], color=BIG_FIVE_COLORS[t.capitalize()], linewidth=2.2,
                   marker="o", markersize=6)
        for t in TRAITS
    ]
    fig.legend(handles, [t.capitalize() for t in TRAITS],
               loc="lower center", ncol=5, fontsize=12, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.05), prop={"weight": "bold"})
    fig.suptitle(
        "PsychAdapter (gemma-2b) trait-conditioned generations — Qwen3-235B judge sweep "
        "(100 questions)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


def main() -> None:
    data = load_judge_runs()
    pairs = {(k[0], k[1]) for k in data}
    print(f"Loaded scores for {len(pairs)} (conditioned, judged) pairs "
          f"({sum(len(v) for v in data.values())} judge scores)")

    write_csv(data, N100_DIR / "n100_TRAIT_cross_latent.csv")

    fig = plot_cross_trait(data)
    out_png = N100_DIR / "n100_TRAIT_cross_latent.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"✓ Plot -> {out_png}")

    paper_png = ROOT / "paper/figures" / PAPER_FIGURES[0]
    paper_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_png, dpi=150, bbox_inches="tight")
    print(f"✓ Paper copy -> {paper_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()

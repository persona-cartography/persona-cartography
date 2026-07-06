"""PsychAdapter cross-trait dose-response figure for the appendix
(Comparison with Trait-Conditioned Adapters).

PsychAdapter (gemma-2b, Big Five KV-prefix conditioning, `huvucode/PsychAdapter`)
generations on the full 240-question assistant-axis set (5 traits x 9 latent
scales -4..+4 = 10,800 generations), scored with the canonical Qwen3-235B judge
sweep. One subplot per conditioned trait, five lines = judge scores on all
OCEAN traits vs latent value. Error bars are 95% BCa bootstrap
(1000 resamples) confidence intervals.

Hydrates judge runs from the HF monorepo:

    evals/gemma-psych-adaptors-evaluated/n240/per_trait/judge_runs/
        qwen3_235b-<fp>/config.json
        qwen3_235b-<fp>/judge_calls/raw/qwen3_235b.jsonl

(one judge run per conditioning-trait subset x judged metric; the cross plot
uses the five *_v2 trait metrics, coherence runs are skipped). Generation:
`scripts_dev/psychadapter_eval/generate_psychadapter_all_responses.py`,
judging: `scripts_dev/psychadapter_eval/judge_psychadapter_responses.py`.

Also writes the aggregated CSV (conditioned, judged, stat, <latent columns>)
next to the cached data.

Paper figures:
    - paper/figures/appendix/fig_psychadapter_n240_cross_latent.pdf

Usage:
    uv run python -m src_dev.visualisations.paper_appendix_psychadapter_cross_latent
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.evals.personality.analyze_results import (
    BIG_FIVE_COLORS,
    _interval_ci_from_bootstrap,
)
from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "appendix/fig_psychadapter_n240_cross_latent.pdf",
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

HF_REPO_ID = "persona-cartography/monorepo"
HF_PREFIX = "evals/gemma-psych-adaptors-evaluated/n240"

CACHE_DIR = project_root / "scratch" / "_psychadapter_cross_latent_cache"


def hydrate_judge_runs() -> Path:
    """Download judge-run configs and raw judge calls from the HF monorepo.

    Returns:
        Local path of the judge_runs directory.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_REPO_ID,
        repo_type="dataset",
        local_dir=CACHE_DIR,
        allow_patterns=[
            f"{HF_PREFIX}/per_trait/judge_runs/*/config.json",
            f"{HF_PREFIX}/per_trait/judge_runs/*/judge_calls/raw/*.jsonl",
        ],
    )
    return CACHE_DIR / HF_PREFIX / "per_trait" / "judge_runs"


def load_judge_runs(judge_runs_dir: Path) -> dict:
    """Collect scores from all judge runs.

    Args:
        judge_runs_dir: Directory holding one subdirectory per judge run.

    Returns:
        {(conditioned_trait, judged_trait, latent): [scores]}
    """
    data: dict = defaultdict(list)
    for run_dir in sorted(judge_runs_dir.iterdir()):
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


def _bootstrap_ci(scores: list[float]) -> tuple[float, float, float]:
    """Mean and 95% BCa bootstrap CI, matching the other judge-score figures."""
    if not scores:
        return (float("nan"),) * 3
    arr = np.asarray(scores, dtype=float)
    lo, hi = _interval_ci_from_bootstrap(arr, CI_CONFIDENCE, BOOTSTRAP_RESAMPLES, SEED)
    return float(arr.mean()), lo, hi


def write_csv(data: dict, out_path: Path) -> None:
    latents = sorted({k[2] for k in data})
    with open(out_path, "w") as f:
        f.write("conditioned,judged,stat," + ",".join(str(l) for l in latents) + "\n")
        for cond in TRAITS:
            for judged in TRAITS:
                stats: dict[str, list[str]] = {
                    "mean": [], "ci_lo": [], "ci_hi": [], "std": [], "n": []
                }
                for latent in latents:
                    scores = data.get((cond, judged, latent), [])
                    if scores:
                        mean, lo, hi = _bootstrap_ci(scores)
                        stats["mean"].append(f"{mean:.3f}")
                        stats["ci_lo"].append(f"{lo:.3f}")
                        stats["ci_hi"].append(f"{hi:.3f}")
                        stats["std"].append(f"{np.std(scores):.3f}")
                        stats["n"].append(str(len(scores)))
                    else:
                        for key in stats:
                            stats[key].append("")
                for stat, vals in stats.items():
                    f.write(f"{cond},{judged},{stat}," + ",".join(vals) + "\n")
    print(f"✓ CSV -> {out_path}")


def plot_cross_trait(data: dict):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    latents = sorted({k[2] for k in data})

    for idx, cond_trait in enumerate(TRAITS):
        ax = axes[idx]
        for judged_trait in TRAITS:
            means, err_lo, err_hi = [], [], []
            for latent in latents:
                scores = data.get((cond_trait, judged_trait, latent), [])
                mean, lo, hi = _bootstrap_ci(scores)
                means.append(mean)
                err_lo.append(mean - lo)
                err_hi.append(hi - mean)

            color = BIG_FIVE_COLORS.get(judged_trait.capitalize(), "#999999")
            is_diagonal = judged_trait == cond_trait
            ax.errorbar(
                latents, means, yerr=[err_lo, err_hi], fmt="o-", color=color,
                linewidth=2.2 if is_diagonal else 1.8,
                markersize=6 if is_diagonal else 5,
                capsize=4, capthick=1.2, elinewidth=1.2,
                alpha=0.9 if is_diagonal else 0.25,
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
        "LLM-Judge Sweep on Trait-Conditioned Adapters",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


def main() -> None:
    judge_runs_dir = hydrate_judge_runs()
    data = load_judge_runs(judge_runs_dir)
    pairs = {(k[0], k[1]) for k in data}
    print(f"Loaded scores for {len(pairs)} (conditioned, judged) pairs "
          f"({sum(len(v) for v in data.values())} judge scores)")

    write_csv(data, CACHE_DIR / "n240_TRAIT_cross_latent.csv")

    fig = plot_cross_trait(data)
    paper_pdf = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
    paper_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_pdf, bbox_inches="tight")
    print(f"✓ Paper figure -> {paper_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()

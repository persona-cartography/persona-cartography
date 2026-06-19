"""Activation-capping LLM-judge sweep figures for the appendix.

Produces one judge-score-vs-cap-scale plot per OCEAN± persona for the
paired-DPO LoRAs, evaluated under activation capping. Mirrors the trait/mmlu
sweep figures already in the appendix grid:

    paper/figures/appendix/activation_capping_mcq_llm_judge_evals/
        judge_sweep_<trait>_<sign>_actcap.pdf

Data layout (one fingerprint per home-trait dataset, shared across amp/sup):

    fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/ocean_const_paired_dpo/
        evals/llm_judge_activation_capping_sweep/{fp}/
            scale_{±X.YY}/judge_runs/qwen3_235b/{trait}_v2.jsonl

The baseline (no-adapter) lives at the shared baselines location:

    evals/baselines/llama-3.1-8b-it/llm_judge_activation_capping_sweep/{fp}/
        judge_runs/qwen3_235b/{trait}_v2.jsonl

The actcap sweep only rolled out on the home-trait dataset (one fingerprint
per persona), so each plot is a single trait trace: the home-trait judge score
vs activation-cap scale, with coherence on a secondary axis.
"""

from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from src.visualisations.palette import BIG_FIVE_COLORS
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.judge_jsonl import judge_scores
from src.visualisations.appendix_sweep_common import (
    bootstrap_ci,
    stream_to_tempfile,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
JUDGE_SUITE = "llm_judge_activation_capping_sweep"
JUDGE_RATER = "qwen3_235b"

CACHE_DIR = project_root / "scratch" / "_actcap_judge_cache"
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
SEED = 42

JUDGE_FP_BY_TRAIT = {
    "openness": "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion": "a961f641eb",
    "agreeableness": "0705e3276a",
    "neuroticism": "b2a49f1b4d",
}

SCALES = [-2.0, -1.0, 1.0, 2.0]  # baseline (0) lives under evals/baselines/

# Activation-capping figures cover the 10 OCEAN± personas only (no control).
PERSONAS: list[tuple[str, str]] = [
    (trait, direction)
    for trait in (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    )
    for direction in ("amplifier", "suppressor")
]

OUT_DIR = Path("appendix/activation_capping_mcq_llm_judge_evals")

PAPER_FIGURES = [
    f"{OUT_DIR}/judge_sweep_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_actcap.pdf"
    for trait, direction in PERSONAS
]

COHERENCE_LEAF = "better_coherence_judge.jsonl"


def _fmt_scale(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _scale_jsonl_path(
    trait: str, direction: str, fp: str, scale: float, leaf: str
) -> str:
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/"
        f"{JUDGE_SUITE}/{fp}/scale_{_fmt_scale(scale)}/judge_runs/{JUDGE_RATER}/{leaf}"
    )


def _baseline_jsonl_path(fp: str, leaf: str) -> str:
    return (
        f"evals/baselines/{MODEL_SLUG}/{JUDGE_SUITE}/{fp}/"
        f"judge_runs/{JUDGE_RATER}/{leaf}"
    )


def _scores_from_jsonl(jsonl_path: Path) -> np.ndarray:
    """Raw judge scores for bootstrap CIs (failed-call rows excluded).

    Uses the shared judge-jsonl reducer so the bootstrap sample matches the
    means drawn in the other figures.
    """
    return np.asarray([s for _, s in judge_scores(jsonl_path)], dtype=float)


_session = requests.Session()


def _fetch_scores(rel_path: str) -> np.ndarray | None:
    """Stream-download → read per-row scores → unlink the file."""
    return stream_to_tempfile(
        f"{RESOLVE_BASE}/{rel_path}",
        CACHE_DIR,
        _scores_from_jsonl,
        session=_session,
        suffix=".jsonl",
        timeout=120,
        chunk_size=1 << 16,
        quiet_on_non_200=False,
        label=rel_path,
    )


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    return bootstrap_ci(
        values, confidence=CI_CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )


# (mean, ci_lo, ci_hi) per scale, per channel ("trait" | "coherence").
ScalePoint = tuple[float, float, float]
PersonaResult = dict[str, dict[float, ScalePoint]]


def gather_all_scores() -> dict[tuple[str, str], PersonaResult]:
    """Returns ``{(trait, direction): {"trait"|"coherence": {scale: (mean, lo, hi)}}}``."""
    # 2 channels × 10 personas × 4 scales + 5 trait baselines + 5 coherence baselines.
    jobs: list[tuple[tuple[str, str], str, float, str]] = []
    for trait, direction in PERSONAS:
        fp = JUDGE_FP_BY_TRAIT[trait]
        for s in SCALES:
            jobs.append(
                (
                    (trait, direction),
                    "trait",
                    s,
                    _scale_jsonl_path(trait, direction, fp, s, f"{trait}_v2.jsonl"),
                )
            )
            jobs.append(
                (
                    (trait, direction),
                    "coherence",
                    s,
                    _scale_jsonl_path(trait, direction, fp, s, COHERENCE_LEAF),
                )
            )
    # Per-fingerprint baselines (shared across both directions).
    baseline_jobs: list[tuple[str, str, str]] = []
    for trait, fp in JUDGE_FP_BY_TRAIT.items():
        baseline_jobs.append(
            (trait, "trait", _baseline_jsonl_path(fp, f"{trait}_v2.jsonl"))
        )
        baseline_jobs.append(
            (trait, "coherence", _baseline_jsonl_path(fp, COHERENCE_LEAF))
        )

    print(
        f"Streaming {len(jobs)} per-scale + {len(baseline_jobs)} baseline jsonls "
        "(deleting each tempfile after parse) …"
    )
    scale_arrays: dict[tuple[tuple[str, str], str, float], np.ndarray | None] = {}
    baseline_arrays: dict[tuple[str, str], np.ndarray | None] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_key = {
            ex.submit(_fetch_scores, path): ("scale", key, channel, scale)
            for key, channel, scale, path in jobs
        }
        for trait, channel, path in baseline_jobs:
            future_to_key[ex.submit(_fetch_scores, path)] = ("base", trait, channel, None)
        for fut in as_completed(future_to_key):
            tag = future_to_key[fut]
            arr = fut.result()
            if tag[0] == "scale":
                _, key, channel, scale = tag
                scale_arrays[(key, channel, scale)] = arr
            else:
                _, trait, channel, _ = tag
                baseline_arrays[(trait, channel)] = arr

    out: dict[tuple[str, str], PersonaResult] = {
        p: {"trait": {}, "coherence": {}} for p in PERSONAS
    }
    for (key, channel, scale), arr in scale_arrays.items():
        if arr is None or arr.size == 0:
            continue
        out[key][channel][float(scale)] = _bootstrap_ci(arr)
    # Baseline (scale 0) shared across both directions of each trait.
    for (trait, channel), arr in baseline_arrays.items():
        if arr is None or arr.size == 0:
            continue
        point = _bootstrap_ci(arr)
        for direction in ("amplifier", "suppressor"):
            out[(trait, direction)][channel][0.0] = point
    return out


COHERENCE_COLOR = "#7F7F7F"


def render_persona(
    trait: str,
    direction: str,
    result: PersonaResult,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    trait_color = BIG_FIVE_COLORS[trait.capitalize()]
    trait_scores = result.get("trait", {})
    if trait_scores:
        xs = sorted(trait_scores.keys())
        means = np.asarray([trait_scores[s][0] for s in xs])
        los = np.asarray([trait_scores[s][1] for s in xs])
        his = np.asarray([trait_scores[s][2] for s in xs])
        yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
        ax.errorbar(
            xs,
            means,
            yerr=yerr,
            fmt="o-",
            color=trait_color,
            ecolor=trait_color,
            linewidth=2.0,
            markersize=6,
            elinewidth=1.0,
            capsize=3,
            label=f"{trait.capitalize()} judge",
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Activation Vector Limit")
    ax.set_ylabel(f"{trait.capitalize()} Judge Score")
    ax.set_ylim(-4.0, 4.0)
    # Fixed x-range so the trait/MMLU/judge panels share an axis in the
    # appendix grid regardless of which scales actually carry data.
    ax.set_xlim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)

    coherence_scores = result.get("coherence", {})
    ax2 = ax.twinx()
    if coherence_scores:
        xs = sorted(coherence_scores.keys())
        means = np.asarray([coherence_scores[s][0] for s in xs])
        los = np.asarray([coherence_scores[s][1] for s in xs])
        his = np.asarray([coherence_scores[s][2] for s in xs])
        yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
        ax2.errorbar(
            xs,
            means,
            yerr=yerr,
            fmt="s--",
            color=COHERENCE_COLOR,
            ecolor=COHERENCE_COLOR,
            linewidth=1.6,
            markersize=5,
            elinewidth=1.0,
            capsize=3,
            label="Coherence",
        )
    ax2.set_ylabel("Coherence Judge Score")
    ax2.set_ylim(0.0, 10.0)

    sign = "↑" if direction == "amplifier" else "↓"
    ax.set_title(
        f"LLM Judge Activation Capping: {trait.capitalize()} {sign}",
        fontsize=10,
    )

    # Combined legend so trait + coherence appear in one box.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(
            h1 + h2,
            l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(h1) + len(h2),
            fontsize=8,
            framealpha=0.9,
            handlelength=1.5,
            columnspacing=1.0,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        all_scores = gather_all_scores()
        print(f"Rendering {len(PERSONAS)} figures …")
        for trait, direction in PERSONAS:
            sign = "plus" if direction == "amplifier" else "minus"
            out = (
                PAPER_FIGURES_DIR
                / OUT_DIR
                / f"judge_sweep_{trait}_{sign}_actcap.pdf"
            )
            render_persona(trait, direction, all_scores[(trait, direction)], out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

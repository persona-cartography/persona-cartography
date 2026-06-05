"""Paired-DPO LoRA-scale Qwen3-235B LLM-judge sweep figures for the OCEAN appendix.

Verbatim migration of ``src_dev/visualisations/paper_appendix_paired_dpo_judge.py``;
only the imports were repointed at ``src.*`` and this provenance note added.

Mirror of ``paper_appendix_actcap_judge.py`` for the LoRA-scale (rather than
activation-capping) sweep, with two changes:

* All 5 OCEAN trait lines are plotted on the primary y-axis (one per
  per-trait dataset fingerprint), not just the persona's home trait.
* x-axis range is fixed to ``[-4, 4]`` to match the TRAIT and MMLU plots
  in the OCEAN appendix.

Coherence stays on the secondary y-axis (dashed grey, single line, home-
trait fingerprint) — same convention as the activation-capping judge plot.

Output:
    paper/figures/appendix/ocean_results/
        judge_sweep_<trait>_<sign>_paired_dpo.pdf

Per persona we hit 5 fingerprints × 4 scales × 2 leaves (trait + coherence)
= 40 small jsonls + 5 baselines, all streamed into per-task tempfiles that
are deleted after parse.
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

from src.visualisations.palette import (
    BIG_FIVE_COLORS,
)
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.judge_jsonl import judge_scores
from src.visualisations.appendix_sweep_common import (
    PERSONAS,
    bootstrap_ci,
    persona_filename_stem,
    persona_title,
    stream_to_tempfile,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
JUDGE_SUITE = "llm_judge_lora_scale_sweep"
JUDGE_RATER = "qwen3_235b"

CACHE_DIR = project_root / "scratch" / "_paired_dpo_judge_cache"
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
# PERSONAS comes from appendix_sweep_common; the JUDGE_FP_BY_TRAIT dict above is
# keyed in the same OCEAN order it builds personas from.

SCALES = [-2.0, -1.0, 1.0, 2.0]  # baseline (0) lives under combos/_baseline/


def _persona_judge_dir(trait: str, direction: str) -> str:
    """Path prefix for ``llm_judge_lora_scale_sweep`` for this persona."""
    if trait == "control":
        return (
            f"fine_tuning/{MODEL_SLUG}/other/ocean_def_control/amplifier/"
            f"ocean_const_paired_dpo_s1vs2/evals/{JUDGE_SUITE}"
        )
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/{JUDGE_SUITE}"
    )

OUT_DIR = Path("appendix/ocean_results")

PAPER_FIGURES = [
    f"{OUT_DIR}/judge_sweep_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_paired_dpo.pdf"
    for trait, direction in PERSONAS
]

COHERENCE_LEAF = "better_coherence_judge.jsonl"


def _fmt_scale(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _scale_jsonl_path(trait: str, direction: str, fp: str, scale: float, leaf: str) -> str:
    return (
        f"{_persona_judge_dir(trait, direction)}/{fp}/"
        f"scale_{_fmt_scale(scale)}/judge_runs/{JUDGE_RATER}/{leaf}"
    )


def _baseline_jsonl_path(fp: str, leaf: str) -> str:
    return (
        f"combos/{MODEL_SLUG}/_baseline/{JUDGE_SUITE}/{fp}/"
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
    """Stream-download → read per-row scores → unlink the file.

    Many fingerprint × persona combinations don't exist — that's expected for
    cross-trait baseline-only fingerprints, so non-200s stay quiet.
    """
    return stream_to_tempfile(
        f"{RESOLVE_BASE}/{rel_path}",
        CACHE_DIR,
        _scores_from_jsonl,
        session=_session,
        suffix=".jsonl",
        timeout=120,
        chunk_size=1 << 16,
        quiet_on_non_200=True,
        label=rel_path,
    )


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    return bootstrap_ci(
        values, confidence=CI_CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )


# (mean, ci_lo, ci_hi) per scale, per channel.
ScalePoint = tuple[float, float, float]
# {channel: {scale: ScalePoint}}; channels are "trait/{Trait}" or "coherence".
PersonaResult = dict[str, dict[float, ScalePoint]]


def gather_persona_scores(persona: tuple[str, str]) -> PersonaResult:
    """For one persona, fetch judge scores for all 5 OCEAN traits + coherence."""
    trait, direction = persona
    # For the control adapter, "home trait" is meaningless — coherence is
    # dataset-independent so we just use the openness fingerprint.
    home_fp = JUDGE_FP_BY_TRAIT.get(trait, JUDGE_FP_BY_TRAIT["openness"])

    # Adapter-side jobs: 5 traits × 4 scales (judge file) + 4 scales (coherence,
    # at the home-trait fingerprint only — coherence is dataset-independent).
    jobs: list[tuple[str, float, str]] = []  # (channel_key, scale, rel_path)
    for trait_lower, fp in JUDGE_FP_BY_TRAIT.items():
        for s in SCALES:
            jobs.append((
                f"trait/{trait_lower.capitalize()}", s,
                _scale_jsonl_path(trait, direction, fp, s, f"{trait_lower}_v2.jsonl"),
            ))
    for s in SCALES:
        jobs.append((
            "coherence", s,
            _scale_jsonl_path(trait, direction, home_fp, s, COHERENCE_LEAF),
        ))
    # Baselines (scale 0): one per trait + coherence at home_fp.
    baseline_jobs: list[tuple[str, str]] = []
    for trait_lower, fp in JUDGE_FP_BY_TRAIT.items():
        baseline_jobs.append((
            f"trait/{trait_lower.capitalize()}",
            _baseline_jsonl_path(fp, f"{trait_lower}_v2.jsonl"),
        ))
    baseline_jobs.append(("coherence", _baseline_jsonl_path(home_fp, COHERENCE_LEAF)))

    print(
        f"  [{trait}/{direction}] streaming {len(jobs)} per-scale + "
        f"{len(baseline_jobs)} baseline jsonls …"
    )
    arrays: dict[tuple[str, float], np.ndarray | None] = {}
    base_arrays: dict[str, np.ndarray | None] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_key = {
            ex.submit(_fetch_scores, path): ("scale", channel, scale)
            for channel, scale, path in jobs
        }
        for channel, path in baseline_jobs:
            future_to_key[ex.submit(_fetch_scores, path)] = ("base", channel, None)
        for fut in as_completed(future_to_key):
            tag = future_to_key[fut]
            arr = fut.result()
            if tag[0] == "scale":
                _, channel, scale = tag
                arrays[(channel, scale)] = arr
            else:
                _, channel, _ = tag
                base_arrays[channel] = arr

    out: PersonaResult = {}
    for (channel, scale), arr in arrays.items():
        if arr is None or arr.size == 0:
            continue
        out.setdefault(channel, {})[float(scale)] = _bootstrap_ci(arr)
    for channel, arr in base_arrays.items():
        if arr is None or arr.size == 0:
            continue
        out.setdefault(channel, {})[0.0] = _bootstrap_ci(arr)
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

    legend_handles: list = []
    legend_labels: list = []
    for trait_name in ("Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"):
        ch = f"trait/{trait_name}"
        scores = result.get(ch, {})
        if not scores:
            continue
        xs = sorted(scores.keys())
        means = np.asarray([scores[s][0] for s in xs])
        los = np.asarray([scores[s][1] for s in xs])
        his = np.asarray([scores[s][2] for s in xs])
        yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
        color = BIG_FIVE_COLORS[trait_name]
        line = ax.errorbar(
            xs, means, yerr=yerr, fmt="o-",
            color=color, ecolor=color,
            linewidth=2.0, markersize=5, elinewidth=1.0, capsize=3,
            label=trait_name,
        )
        legend_handles.append(line)
        legend_labels.append(trait_name)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("LoRA Scale", fontsize=12)
    ax.set_ylabel("Judge Score", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xlim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    coherence_scores = result.get("coherence", {})
    if coherence_scores:
        xs = sorted(coherence_scores.keys())
        means = np.asarray([coherence_scores[s][0] for s in xs])
        los = np.asarray([coherence_scores[s][1] for s in xs])
        his = np.asarray([coherence_scores[s][2] for s in xs])
        yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
        line = ax2.errorbar(
            xs, means, yerr=yerr, fmt="s--",
            color=COHERENCE_COLOR, ecolor=COHERENCE_COLOR,
            linewidth=1.6, markersize=5, elinewidth=1.0, capsize=3,
            label="Coherence",
        )
        legend_handles.append(line)
        legend_labels.append("Coherence")
    ax2.set_ylabel("Coherence Judge Score", fontsize=12)
    ax2.tick_params(axis="y", labelsize=11)
    ax2.set_ylim(0.0, 10.0)

    ax.set_title(
        persona_title(trait, direction, "LLM Judge"),
        fontsize=13,
    )

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=len(legend_handles),
            fontsize=7, framealpha=0.9,
            handlelength=1.0, handletextpad=0.4, columnspacing=0.6, borderpad=0.3,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Processing {len(PERSONAS)} personas one at a time …")
        for persona in PERSONAS:
            trait, direction = persona
            result = gather_persona_scores(persona)
            stem = persona_filename_stem(trait, direction)
            out = PAPER_FIGURES_DIR / OUT_DIR / f"judge_sweep_{stem}.pdf"
            render_persona(trait, direction, result, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

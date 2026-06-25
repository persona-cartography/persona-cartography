"""Cross-model comparison: conscientiousness suppressor v2 scale sweep.

For each of Llama-3.1-8B-Instruct, Gemma-3-4B-IT, Gemma-3-12B-IT, Gemma-3-27B-IT
(same persona LoRA methodology, same Qwen3-235B judge, same rollout fingerprint
per model), plot:

  (left panel)  Conscientiousness v2 judge score vs LoRA scale
  (right panel) Coherence ("better_coherence_judge") judge score vs LoRA scale

One colored line per model. Scale 0 (baseline, no adapter) comes from
``combos/{model}/_baseline/.../{fp}/judge_runs/qwen3_235b/*.jsonl``. Other
scales come from the per-model fine_tuning tree.

Hydrates directly from the persona-cartography/monorepo HF dataset; runs
offline after the first hydration (per-metric on-disk cache).

Paper figures (placeholder slot in the appendix — script is "paper-quality"
but the figure is not yet referenced by the LaTeX body):
    paper/figures/appendix/fig_model_size_consc_sup_v2.pdf
    paper/figures/appendix/fig_model_size_consc_sup_v2.png

Run with:
    uv run python -m src_dev.visualisations.paper_model_size_consc_sup_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import _interval_ci_from_bootstrap
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

# ---------------------------------------------------------------------------
# Config — hardcoded per CLAUDE.md / paper conventions
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-cartography/monorepo"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SCORE_MIN = -4.0
SCORE_MAX = 4.0

# (metric key, pretty label, y_lim, out_stem)
METRICS: list[tuple[str, str, tuple[float, float], str]] = [
    (
        "conscientiousness_v2",
        "Conscientiousness v2 judge",
        (SCORE_MIN, SCORE_MAX),
        "appendix/fig_model_size_consc_sup_v2_conscientiousness",
    ),
    (
        "better_coherence_judge",
        "Coherence judge",
        (0.0, 10.0),  # coherence is scored on a 0..10 scale, not OCEAN's [-4,+4]
        "appendix/fig_model_size_consc_sup_v2_coherence",
    ),
]

# Per-model (display name, HF model slug, rollout fingerprint).
# Fingerprints are deterministic from (base_model, dataset_path, max_samples,
# seed, num_rollouts, assistant_temperature, assistant_top_p,
# assistant_max_new_tokens). All four entries share the vanton4_qwen3-style
# rollout settings (240 samples × 1 rollout, T=1.0, max_new=2048) applied to
# data/ocean_open_ended/conscientiousness.jsonl. See
# scripts_dev/evals/llm_judge_sweep/configs/gemma_consc_sup/_shared.py.
MODELS: list[tuple[str, str, str, str]] = [
    # (label, model_slug, fingerprint, color)
    ("Llama-3.1-8B-Instruct", "llama-3.1-8b-it", "e6426e3031", "#1f77b4"),
    ("Qwen2.5-7B-Instruct",   "qwen-2.5-7b-it",  "02526164aa", "#9467bd"),
    ("Gemma-3-4B-IT",         "gemma-3-4b-it",   "389e5b9309", "#2ca02c"),
    ("Gemma-3-12B-IT",        "gemma-3-12b-it",  "d251d74e7d", "#ff7f0e"),
    ("Gemma-3-27B-IT",        "gemma-3-27b-it",  "5b60ecfd83", "#d62728"),
]

# Appendix placeholder — not yet referenced by the LaTeX body. Update
# MANIFEST.md when this gets a concrete slot in the paper.
PAPER_FIGURES = [
    "appendix/fig_model_size_consc_sup_v2_conscientiousness.pdf",
    "appendix/fig_model_size_consc_sup_v2_conscientiousness.png",
    "appendix/fig_model_size_consc_sup_v2_coherence.pdf",
    "appendix/fig_model_size_consc_sup_v2_coherence.png",
]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "model_size_consc_sup_v2"

BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _format_scale(scale: float) -> str:
    return f"scale_{'+' if scale >= 0 else '-'}{abs(scale):.2f}"


def _sweep_cell_judge_path(model_slug: str, fp: str, scale: float, metric: str) -> str:
    return (
        f"fine_tuning/{model_slug}/ocean/conscientiousness/suppressor/v2"
        f"/evals/{EVAL_NAME}/{fp}/{_format_scale(scale)}/judge_runs/{RATER_ID}/{metric}.jsonl"
    )


def _baseline_judge_path(model_slug: str, fp: str, metric: str) -> str:
    return (
        f"combos/{model_slug}/_baseline/{EVAL_NAME}/{fp}"
        f"/judge_runs/{RATER_ID}/{metric}.jsonl"
    )


def _judge_hf_path(model_slug: str, fp: str, scale: float, metric: str) -> str:
    if scale == 0.0:
        return _baseline_judge_path(model_slug, fp, metric)
    return _sweep_cell_judge_path(model_slug, fp, scale, metric)


# ---------------------------------------------------------------------------
# Hydration (per-file cache, mirrors paper_main_*_spider.py)
# ---------------------------------------------------------------------------

def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _hydrate_judge_file(hf_path: str) -> Path | None:
    local_path = _cache_path(hf_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    parent_hf = hf_path.rsplit("/", 1)[0]
    filename = hf_path.rsplit("/", 1)[1]
    local_parent = _cache_path(parent_hf)
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        print(f"  ✗ hydrate failed for {hf_path}: {type(exc).__name__}: {str(exc)[:160]}")
        return None
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    return None


def _load_scores(jsonl_path: Path) -> list[float]:
    out: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = row.get("score")
            if val is None or not isinstance(val, (int, float)):
                continue
            out.append(float(val))
    return out


# ---------------------------------------------------------------------------
# Build scores: {(label, metric): {scale: (mean, ci_lo, ci_hi)}}
# ---------------------------------------------------------------------------

def _mean_and_ci(scores: list[float]) -> tuple[float, float, float] | None:
    if not scores:
        return None
    mean = float(np.mean(scores))
    if len(scores) < 2:
        return (mean, mean, mean)
    try:
        lo, hi = _interval_ci_from_bootstrap(
            np.asarray(scores, dtype=float),
            confidence=CI_CONFIDENCE, n_resamples=BOOTSTRAP_RESAMPLES, seed=42,
        )
    except Exception:
        # Defensive: degenerate inputs; return point estimate as flat CI.
        return (mean, mean, mean)
    return (mean, float(lo), float(hi))


def build_scores() -> dict[tuple[str, str], dict[float, tuple[float, float, float]]]:
    out: dict[tuple[str, str], dict[float, tuple[float, float, float]]] = {}
    for label, slug, fp, _color in MODELS:
        for metric, _metric_label, _ylim, _out_stem in METRICS:
            key = (label, metric)
            out[key] = {}
            for scale in SCALE_POINTS:
                hf_path = _judge_hf_path(slug, fp, scale, metric)
                local = _hydrate_judge_file(hf_path)
                if local is None:
                    print(f"  ⚠ {label} / {metric} / scale {scale:+.1f}: missing on HF")
                    continue
                scores = _load_scores(local)
                mci = _mean_and_ci(scores)
                if mci is None:
                    print(f"  ⚠ {label} / {metric} / scale {scale:+.1f}: no valid scores")
                    continue
                out[key][scale] = mci
                print(f"  ✓ {label:25s} / {metric:25s} / scale {scale:+.1f}: mean={mci[0]:+.3f}  [{mci[1]:+.3f}, {mci[2]:+.3f}]  (n={len(scores)})")
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render_single_metric(
    *,
    metric: str,
    metric_label: str,
    y_lim: tuple[float, float],
    per_cell: dict[tuple[str, str], dict[float, tuple[float, float, float]]],
    out_paths: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for label, _slug, _fp, color in MODELS:
        data = per_cell.get((label, metric), {})
        if not data:
            continue
        xs = sorted(data)
        means = np.array([data[x][0] for x in xs])
        los = np.array([data[x][1] for x in xs])
        his = np.array([data[x][2] for x in xs])
        err_lo = means - los
        err_hi = his - means
        ax.errorbar(
            xs, means,
            yerr=np.vstack([err_lo, err_hi]),
            color=color, linewidth=2.0, marker="o", markersize=6,
            capsize=3, label=label,
        )

    ax.set_title(
        f"{metric_label} — conscientiousness suppressor v2 across model sizes",
        fontsize=12,
    )
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("judge mean score")
    ax.set_xticks(SCALE_POINTS)
    ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.3)
    if y_lim[0] <= 0.0 <= y_lim[1]:
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, title="Base model")
    fig.tight_layout()

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✓ saved {out_path}")
    plt.close(fig)


def main() -> None:
    print(f"[model-size-consc-sup] cache dir: {CACHE_DIR}")
    print("[model-size-consc-sup] hydrating judge scores from HF...")
    per_cell = build_scores()
    for metric, metric_label, y_lim, out_stem in METRICS:
        pdf_path = PAPER_FIGURES_DIR / f"{out_stem}.pdf"
        png_path = PAPER_FIGURES_DIR / f"{out_stem}.png"
        _render_single_metric(
            metric=metric, metric_label=metric_label, y_lim=y_lim,
            per_cell=per_cell, out_paths=[pdf_path, png_path],
        )


if __name__ == "__main__":
    main()

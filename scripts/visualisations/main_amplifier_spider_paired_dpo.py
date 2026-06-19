"""Main amplifier spider plot using ocean_const_paired_dpo for all 5 OCEAN traits.

This is the canonical Fig. 1 amplifier spider for the paper. All five OCEAN
amplifier LoRAs are read from
``fine_tuning/llama-3.1-8b-it/ocean/{trait}/amplifier/ocean_const_paired_dpo`` —
the paired-teacher DPO ocean_const adapters.

Rendering, legend, baseline handling, and PLOT_MODE match
``main_suppressor_spider_paired_dpo.py`` so the two subplots are visually
comparable. ``build_scores`` and ``AMPLIFIERS_META`` are imported by
``main_fig1_banner.py`` to hydrate the banner's amplifier spider panel.

Paper figures (written both as PDF and PNG):
    paper/figures/main/fig_1_amplifier_spider_ocean_const_paired_dpo.pdf
    paper/figures/main/fig_1_amplifier_spider_ocean_const_paired_dpo.png

Run with:
    uv run python scripts/visualisations/main_amplifier_spider_paired_dpo.py
"""

from __future__ import annotations

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

from src.utils.hf_hub import download_path_to_dir
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.judge_jsonl import mean_judge_score
from src.visualisations.ocean_spider import to_headroom
from src.visualisations.palette import BIG_FIVE_COLORS

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

PAPER_FIGURES = [
    "main/fig_1_amplifier_spider_ocean_const_paired_dpo.pdf",
    "main/fig_1_amplifier_spider_ocean_const_paired_dpo.png",
]

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"
SCALE = 1.0
SCALE_LABEL = "scale_+1.00"
ADAPTER_VERSION = "ocean_const_paired_dpo"

# Rollout fingerprints per judged-trait prompt set (the canonical 240×1
# per-OCEAN-trait fingerprints shared with the other Fig. 1 panels).
FP_BY_TRAIT = {
    "openness":          "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion":      "a961f641eb",
    "agreeableness":     "0705e3276a",
    "neuroticism":       "b2a49f1b4d",
}

AMPLIFIERS_META: list[tuple[str, str, str]] = [
    ("o_plus", "openness",          BIG_FIVE_COLORS["Openness"]),
    ("c_plus", "conscientiousness", BIG_FIVE_COLORS["Conscientiousness"]),
    ("e_plus", "extraversion",      BIG_FIVE_COLORS["Extraversion"]),
    ("a_plus", "agreeableness",     BIG_FIVE_COLORS["Agreeableness"]),
    ("n_plus", "neuroticism",       BIG_FIVE_COLORS["Neuroticism"]),
]

BASELINE_COLOR = "#4D4D4D"
BASELINE_LEGEND_LABEL = "baseline Llama3.1-8b-Instruct"

LEGEND_LABELS: dict[str, str] = {
    "o_plus": "O↑",
    "c_plus": "C↑",
    "e_plus": "E↑",
    "a_plus": "A↑",
    "n_plus": "N↑",
}

PLOT_MODE = "headroom"
SCORE_MIN = -4.0
SCORE_MAX = 4.0

OUT_STEM = "main/fig_1_amplifier_spider_ocean_const_paired_dpo"
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "amplifier_spider_ocean_const_paired_dpo"


def _adapter_hf_dir(home_trait: str, fingerprint: str) -> str:
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{home_trait}/amplifier/{ADAPTER_VERSION}"
        f"/evals/{EVAL_NAME}/{fingerprint}/{SCALE_LABEL}"
    )


def _baseline_hf_dir(fingerprint: str) -> str:
    # Shared, fingerprinted base-model store (same root the eval suite uses);
    # legacy location was combos/{model}/_baseline/{eval}/{fp}.
    return f"evals/baselines/{MODEL_SLUG}/{EVAL_NAME}/{fingerprint}"


def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


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


def _trait_metric(trait_lower: str) -> str:
    return f"{trait_lower}_v2"


def _trait_title(trait_lower: str) -> str:
    return trait_lower.capitalize()


def build_scores() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    per_amp: dict[str, dict[str, float]] = {key: {} for key, _, _ in AMPLIFIERS_META}
    baseline: dict[str, float] = {}

    for trait_lower, fp in FP_BY_TRAIT.items():
        hf_path = _judge_hf_path(_baseline_hf_dir(fp), _trait_metric(trait_lower))
        local = _hydrate_judge_file(hf_path)
        if local is None:
            print(f"  ⚠ baseline / {trait_lower}: missing on HF")
            continue
        mean = mean_judge_score(local)
        if mean is None:
            print(f"  ⚠ baseline / {trait_lower}: no valid scores in {local}")
            continue
        baseline[_trait_title(trait_lower)] = mean
        print(f"  ✓ baseline / {trait_lower:18s}: mean = {mean:+.3f}")

    for key, home_trait, _color in AMPLIFIERS_META:
        for judged_trait, fp in FP_BY_TRAIT.items():
            cell_dir = _adapter_hf_dir(home_trait, fp)
            hf_path = _judge_hf_path(cell_dir, _trait_metric(judged_trait))
            local = _hydrate_judge_file(hf_path)
            if local is None:
                kind = "own" if judged_trait == home_trait else "cross"
                print(f"  ⚠ {key} / {judged_trait:18s} ({kind}): missing on HF — likely still running")
                continue
            mean = mean_judge_score(local)
            if mean is None:
                print(f"  ⚠ {key} / {judged_trait}: no valid scores in {local}")
                continue
            per_amp[key][_trait_title(judged_trait)] = mean
            print(f"  ✓ {key} / {judged_trait:18s}: mean = {mean:+.3f}")

    return per_amp, baseline


def _render_spider(
    *,
    per_amplifier: dict[str, dict[str, float]],
    baseline: dict[str, float],
    out_paths: list[Path],
) -> None:
    traits = OCEAN_TRAITS

    if PLOT_MODE == "headroom":
        if not baseline:
            raise RuntimeError(
                "headroom mode requires a non-empty baseline — all adapter rows transform "
                "to fractions of (score_bound - baseline)."
            )
        per_amplifier = to_headroom(
            per_amplifier, baseline, score_min=SCORE_MIN, score_max=SCORE_MAX,
        )
        baseline = {t: 0.0 for t in baseline}
        y_lim = (-1.0, 1.0)
        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        y_tick_labels = ["-100%", "-50%", "0", "+50%", "+100%"]
    elif PLOT_MODE == "raw":
        y_lim = (SCORE_MIN, SCORE_MAX)
        y_ticks = [SCORE_MIN, SCORE_MIN / 2, 0.0, SCORE_MAX / 2, SCORE_MAX]
        y_tick_labels = [f"{t:+.0f}" for t in y_ticks]
    else:
        raise ValueError(f"unknown PLOT_MODE={PLOT_MODE!r}")

    angles = np.linspace(0.0, 2.0 * np.pi, len(traits), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    def _polygon(scores: dict[str, float], *, label: str, color: str, linewidth: float) -> None:
        means = [scores.get(t, float("nan")) for t in traits]
        means_closed = means + means[:1]
        ax.plot(angles_closed, means_closed, "-", color=color, linewidth=linewidth, label=label)
        for angle, val in zip(angles, means):
            if not np.isnan(val):
                ax.plot([angle], [val], "o", color=color, markersize=6)
        if not any(np.isnan(v) for v in means):
            ax.fill(angles_closed, means_closed, color=color, alpha=0.10)

    if baseline and PLOT_MODE == "raw":
        _polygon(baseline, label=BASELINE_LEGEND_LABEL, color=BASELINE_COLOR, linewidth=2.5)

    for key, _home_trait, color in AMPLIFIERS_META:
        row = per_amplifier.get(key, {})
        if not row:
            continue
        _polygon(row, label=LEGEND_LABELS.get(key, key), color=color, linewidth=2.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(traits, fontsize=11)
    for tick_label, trait in zip(ax.get_xticklabels(), traits):
        tick_label.set_color(BIG_FIVE_COLORS[trait])
    ax.set_ylim(*y_lim)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=9)
    if PLOT_MODE == "headroom":
        ring_theta = np.linspace(0.0, 2.0 * np.pi, 180)
        ax.plot(
            ring_theta, np.zeros_like(ring_theta),
            "-", color=BASELINE_COLOR, linewidth=2.5, alpha=1.0,
            label=BASELINE_LEGEND_LABEL,
        )
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=10, framealpha=0.9)

    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✓ saved {out_path}")
    plt.close(fig)


def main() -> None:
    pdf_path = PAPER_FIGURES_DIR / f"{OUT_STEM}.pdf"
    png_path = PAPER_FIGURES_DIR / f"{OUT_STEM}.png"

    print(f"[amp-spider] adapter version: {ADAPTER_VERSION} (all 5 OCEAN traits)")
    print(f"[amp-spider] cache dir: {CACHE_DIR}")
    print(f"[amp-spider] outputs:   {pdf_path}, {png_path}")
    print("[amp-spider] hydrating judge scores from HF...")

    per_amp, baseline = build_scores()
    _render_spider(per_amplifier=per_amp, baseline=baseline, out_paths=[pdf_path, png_path])


if __name__ == "__main__":
    main()

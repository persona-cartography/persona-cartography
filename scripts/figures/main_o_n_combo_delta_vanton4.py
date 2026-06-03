"""Per-trait delta bar chart for the O↑ × N↑ vanton4_paired_dpo combo.

Hydrates the (+1, +1) combo cell for each of the 5 OCEAN traits, plus the
base-model baseline and the single-adapter O↑ / N↑ vanton4_paired_dpo cells.
Plots the Qwen3-235B judge's mean trait score delta vs baseline for each
OCEAN trait — similar layout to ``scripts_dev/evals/ocean_delta_plot.py``.

Data sources:
 - Combo (+1, +1): ``combos/llama-3.1-8b-it/ocean-neuroticism-amplifier-vanton4_paired_dpo__ocean-openness-amplifier-vanton4_paired_dpo/llm_judge_lora_scale_sweep/{fp}/cell_{spec}/``
 - Baseline: ``combos/llama-3.1-8b-it/_baseline/llm_judge_lora_scale_sweep/{fp}/``
 - Single-adapter O↑: ``fine_tuning/.../openness/amplifier/vanton4_paired_dpo/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``
 - Single-adapter N↑: ``fine_tuning/.../neuroticism/amplifier/vanton4_paired_dpo/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``

All data sources use the same canonical 240×1 fingerprint per OCEAN trait.

Paper figures:
    - paper/figures/main/fig_1_o_n_amplifier_combo_delta_vanton4_paired_dpo.pdf

Provenance:
    Migrated verbatim from
    ``src_dev/visualisations/paper_main_o_n_combo_delta_vanton4.py``;
    only import lines and this note differ.

Run with:
    uv run python -m scripts.figures.main_o_n_combo_delta_vanton4
"""

from __future__ import annotations

import json
import statistics
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.visualisations.palette import BIG_FIVE_COLORS
from src.utils.hf_hub import download_path_to_dir
from src.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "main/fig_1_o_n_amplifier_combo_delta_vanton4_paired_dpo.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"
SCORE_MIN = -4.0
SCORE_MAX = 4.0

# Current combo: O↑ (openness amplifier) × N↑ (neuroticism amplifier), both
# from the paired-teacher DPO vanton4 adapters.
O_SLUG = "ocean-openness-amplifier-vanton4_paired_dpo"
N_SLUG = "ocean-neuroticism-amplifier-vanton4_paired_dpo"
O_DIR = "openness/amplifier/vanton4_paired_dpo"
N_DIR = "neuroticism/amplifier/vanton4_paired_dpo"
O_TRAIT_LOWER = "openness"
N_TRAIT_LOWER = "neuroticism"

# Alphabetical combo slug (matches CanonicalCell.combo_slug)
COMBO_SLUG = "__".join(sorted([O_SLUG, N_SLUG]))

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Per-trait: rollout fingerprint for the 240×1 single-adapter/baseline sweep
# on that trait's dataset.
FP_BY_TRAIT: dict[str, str] = {
    "openness":          "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion":      "a961f641eb",
    "agreeableness":     "0705e3276a",
    "neuroticism":       "b2a49f1b4d",
}

OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "o_n_amplifier_combo_delta_vanton4_paired_dpo"
# Local mirror of the HF monorepo — populated by the sweep runners in
# skip-upload mode. Used as a fallback when HF hydrate fails (or while a
# sweep has finished locally but hasn't been pushed to HF yet).
LOCAL_MONOREPO = project_root / "scratch" / "monorepo"


# ---------------------------------------------------------------------------
# Path helpers (match CanonicalCell.hf_dir)
# ---------------------------------------------------------------------------

def _fmt_scale(x: float) -> str:
    """Scale formatting matching ``format_scale()`` in cell_identity.py."""
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _combo_cell_hf_dir(fingerprint: str, o_scale: float, n_scale: float) -> str:
    """HF dir for the (O-adapter=o_scale, N-adapter=n_scale) combo cell."""
    spec = f"cell_{N_SLUG}{_fmt_scale(n_scale)}_{O_SLUG}{_fmt_scale(o_scale)}"
    return f"combos/{MODEL_SLUG}/{COMBO_SLUG}/{EVAL_NAME}/{fingerprint}/{spec}"


def _single_o_cell_hf_dir(fingerprint: str, o_scale: float) -> str:
    """O↑ vanton4_paired_dpo alone at given scale."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{O_DIR}"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt_scale(o_scale)}"
    )


def _single_n_cell_hf_dir(fingerprint: str, n_scale: float) -> str:
    """N↑ vanton4_paired_dpo alone at given scale."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{N_DIR}"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt_scale(n_scale)}"
    )


def _baseline_hf_dir(fingerprint: str) -> str:
    return f"combos/{MODEL_SLUG}/_baseline/{EVAL_NAME}/{fingerprint}"


def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


# ---------------------------------------------------------------------------
# Hydration + cache
# ---------------------------------------------------------------------------

def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _hydrate_judge_file(hf_path: str) -> Path | None:
    """Resolve ``hf_path`` to a local file, preferring cache → HF → local monorepo.

    Lookup order:
      1. HTTP cache at ``CACHE_DIR/{hf_path}`` (from a previous run).
      2. Fresh download from HF.
      3. ``scratch/monorepo/{hf_path}`` — used when HF is down / rate-limited
         and the file is sitting locally from a skip-upload sweep.
    """
    local = _cache_path(hf_path)
    if local.exists() and local.stat().st_size > 0:
        return local
    parent_hf = hf_path.rsplit("/", 1)[0]
    filename = hf_path.rsplit("/", 1)[1]
    local_parent = _cache_path(parent_hf)
    hf_exc: Exception | None = None
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        hf_exc = exc
    if local.exists() and local.stat().st_size > 0:
        return local
    # HF didn't have it — try the local monorepo mirror before giving up.
    local_mirror = LOCAL_MONOREPO / hf_path
    if local_mirror.exists() and local_mirror.stat().st_size > 0:
        print(f"  ← local mirror: {hf_path}")
        return local_mirror
    if hf_exc is not None:
        print(f"  ✗ hydrate failed: {type(hf_exc).__name__}: {str(hf_exc)[:140]}")
    return None


def _mean_score_median_across_repeats(jsonl_path: Path) -> float | None:
    """Median over repeats per response_id, then mean across responses."""
    grouped: dict[str, list[int]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") not in {"success", "parse_error"}:
                continue
            s = row.get("score")
            if not isinstance(s, (int, float)):
                continue
            rid = str(row.get("response_id", ""))
            grouped[rid].append(int(s))
    if not grouped:
        return None
    medians = [statistics.median(v) for v in grouped.values() if v]
    return statistics.fmean(medians) if medians else None


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _trait_metric(trait_lower: str) -> str:
    return f"{trait_lower}_v2"


def _fetch(cell_hf_dir: str, trait_lower: str) -> float | None:
    """Hydrate + parse one (cell, trait) mean score."""
    hf_path = _judge_hf_path(cell_hf_dir, _trait_metric(trait_lower))
    local = _hydrate_judge_file(hf_path)
    if local is None:
        return None
    return _mean_score_median_across_repeats(local)


def gather() -> dict[str, dict[str, float | None]]:
    """Returns per-trait means for baseline, single adapters, and combo.

    ``o_adapter`` / ``n_adapter`` are internal keys for the O↑ and N↑
    single-adapter cells. Display labels live in ``TARGET_DISPLAY``.
    """
    out: dict[str, dict[str, float | None]] = {}
    for trait_lower, fp in FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        print(f"\n[trait] {trait_title}  (fp={fp})")

        combo_mean    = _fetch(_combo_cell_hf_dir(fp, 1.0, 1.0), trait_lower)
        baseline_mean = _fetch(_baseline_hf_dir(fp), trait_lower)
        o_mean        = _fetch(_single_o_cell_hf_dir(fp, 1.0), trait_lower)
        n_mean        = _fetch(_single_n_cell_hf_dir(fp, 1.0), trait_lower)

        print(f"  baseline           : {baseline_mean}")
        print(f"  O↑ v4 paired DPO   : {o_mean}")
        print(f"  N↑ v4 paired DPO   : {n_mean}")
        print(f"  combo (+1, +1)     : {combo_mean}")

        out[trait_title] = {
            "baseline":  baseline_mean,
            "o_adapter": o_mean,
            "n_adapter": n_mean,
            "combo":     combo_mean,
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TARGET_ORDER = ["o_adapter", "n_adapter", "combo"]
# Short arrow labels to match the spider plots.
TARGET_DISPLAY = {
    "o_adapter": "O↑",
    "n_adapter": "N↑",
    "combo":     "O↑ × N↑",
}
TARGET_COLORS = {
    "o_adapter": BIG_FIVE_COLORS["Openness"],
    "n_adapter": BIG_FIVE_COLORS["Neuroticism"],
    # Teal — deliberately outside the OCEAN palette so the combo bar is
    # unambiguously "not a trait colour". Avoid purple (collides with
    # Agreeableness), red/blue/green/orange (collide with the other four).
    "combo":     "#00838F",
}
TARGET_HATCHES = {
    "o_adapter": "//",
    "n_adapter": "\\\\",
    "combo":     "xx",
}

BASELINE_LEGEND_LABEL = "baseline Llama3.1-8b-Instruct"


def _output_path(*, headroom: bool = False, ceiling: bool = False) -> Path:
    if headroom:
        return OUT_PATH.with_name(f"{OUT_PATH.stem}_headroom{OUT_PATH.suffix}")
    if ceiling:
        return OUT_PATH.with_name(f"{OUT_PATH.stem}_ceiling{OUT_PATH.suffix}")
    return OUT_PATH


def _bar_value(value: float, baseline: float, *, headroom: bool) -> float:
    delta = value - baseline
    if not headroom:
        return delta
    room = (SCORE_MAX - baseline) if delta >= 0 else (baseline - SCORE_MIN)
    return delta / room if room > 0 else 0.0


def render(
    scores: dict[str, dict[str, float | None]],
    out_path: Path,
    *,
    headroom: bool = False,
    ceiling: bool = False,
) -> None:
    x = np.arange(len(OCEAN_TRAITS))
    width = 0.8 / len(TARGET_ORDER)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for i, key in enumerate(TARGET_ORDER):
        deltas = []
        for trait_title in OCEAN_TRAITS:
            row = scores.get(trait_title, {})
            base = row.get("baseline")
            val = row.get(key)
            if base is None or val is None:
                deltas.append(np.nan)
            else:
                deltas.append(_bar_value(val, base, headroom=headroom))
        ax.bar(
            x + (i - (len(TARGET_ORDER) - 1) / 2) * width,
            deltas,
            width,
            label=TARGET_DISPLAY[key],
            color=TARGET_COLORS[key],
            hatch=TARGET_HATCHES[key],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axhline(0.0, color="k", linewidth=0.8, label=BASELINE_LEGEND_LABEL)

    if ceiling and not headroom:
        # Per-trait Δ ceilings: how far above / below baseline the judge
        # score *could* go on a [SCORE_MIN, SCORE_MAX] scale. Drawn as a
        # short horizontal segment spanning that trait's bar group.
        half_span = len(TARGET_ORDER) * width / 2.0
        for i, trait_title in enumerate(OCEAN_TRAITS):
            base = scores.get(trait_title, {}).get("baseline")
            if base is None:
                continue
            max_up = SCORE_MAX - base
            max_down = SCORE_MIN - base
            label = "theoretical Δ ceiling" if i == 0 else None
            ax.hlines(
                [max_up, max_down],
                xmin=i - half_span,
                xmax=i + half_span,
                colors="k",
                linestyles="--",
                linewidth=1.1,
                label=label,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=15, ha="right", fontsize=10)
    # Colour each trait tick label with its canonical OCEAN hue, matching
    # the spider plots in the same figure.
    for tick_label, trait in zip(ax.get_xticklabels(), OCEAN_TRAITS):
        tick_label.set_color(BIG_FIVE_COLORS[trait])
    if headroom:
        ax.set_ylabel("Signed headroom vs base model")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_ylabel("Δ judge score vs base model")
    # Let matplotlib auto-scale — deltas can range up to ±8 on the [−4, +4]
    # judge scale (e.g. baseline near +4 driven down to −4), so a hardcoded
    # (−4, +4) would clip. Tinted backgrounds follow the auto-scaled axis.
    ymin, ymax = ax.get_ylim()
    ax.axhspan(min(ymin, 0.0), 0.0, alpha=0.04, color="red")
    ax.axhspan(0.0, max(ymax, 0.0), alpha=0.04, color="blue")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ saved {out_path}")


def main() -> None:
    parser = ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headroom",
        action="store_true",
        help="Plot signed headroom fraction instead of raw judge-score delta.",
    )
    mode.add_argument(
        "--ceiling",
        action="store_true",
        help="Plot raw Δ with per-trait theoretical Δ ceilings overlaid as horizontal lines.",
    )
    args = parser.parse_args()
    out_path = _output_path(headroom=args.headroom, ceiling=args.ceiling)
    mode_name = "headroom" if args.headroom else ("ceiling" if args.ceiling else "delta")
    print(f"[combo-delta] cache dir: {CACHE_DIR}")
    print(f"[combo-delta] mode:      {mode_name}")
    print(f"[combo-delta] out path:  {out_path}")
    scores = gather()
    # Persist raw scores alongside the figure for iteration / debugging.
    scores_path = CACHE_DIR / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"✓ saved scores {scores_path}")
    render(scores, out_path, headroom=args.headroom, ceiling=args.ceiling)


if __name__ == "__main__":
    main()

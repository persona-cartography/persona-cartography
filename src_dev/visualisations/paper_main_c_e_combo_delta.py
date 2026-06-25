"""Per-trait delta bar chart for the c_minus_v2 × e_plus_v3 1:1 combo at (+1, +1).

Hydrates the (+1, +1) combo cell for each of the 5 OCEAN traits, plus the
base-model baseline and the single-adapter c_minus_v2 / e_plus_v3 cells.
Plots the Qwen3-235B judge's mean trait score delta vs baseline for each
OCEAN trait — similar layout to ``scripts_dev/evals/ocean_delta_plot.py``.

Data sources (all at the canonical 240×1 fingerprints — MAX_SAMPLES=240,
NUM_ROLLOUTS_PER_PROMPT=1, one fingerprint per OCEAN dataset):
 - Combo (+1, +1): ``combos/llama-3.1-8b-it/ocean-conscientiousness-suppressor-v2__ocean-extraversion-amplifier-v3/llm_judge_lora_scale_sweep/{fp}/cell_{spec}/``
 - Baseline: ``combos/llama-3.1-8b-it/_baseline/llm_judge_lora_scale_sweep/{fp}/``
 - Single-adapter c_minus_v2: ``fine_tuning/.../conscientiousness/suppressor/v2/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``
 - Single-adapter e_plus_v3: ``fine_tuning/.../extraversion/amplifier/v3/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``

All four data sources share the same fingerprint per OCEAN dataset, so the
dual-fingerprint fallback that the original vanton4 combo required is gone.

Paper figures:
    - paper/figures/main/fig_1_c_e_combo_delta.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_main_c_e_combo_delta
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "main/fig_1_c_e_combo_delta.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"

# Previous combo (vanton4 suppressors) — kept for reference if we want to
# re-render the old plot.
# C_SLUG_PREV = "ocean-conscientiousness-suppressor-vanton4"
# E_SLUG_PREV = "ocean-extraversion-suppressor-vanton4"
# C_DIR_PREV = "conscientiousness/suppressor/vanton4"
# E_DIR_PREV = "extraversion/suppressor/vanton4"
# FP_BY_TRAIT_PREV: dict[str, tuple[str, str]] = {
#     "openness":          ("1817b5cf78", "67eed27d02"),
#     "conscientiousness": ("97743334f6", "e6426e3031"),
#     "extraversion":      ("47a37c39b7", "a961f641eb"),
#     "agreeableness":     ("b2e6755ff3", "0705e3276a"),
#     "neuroticism":       ("8b01e9fa2c", "b2a49f1b4d"),
# }

# Current combo: c_minus_v2 (conscientiousness suppressor v2) × e_plus_v3
# (extraversion amplifier v3). Asymmetric — one suppressor, one amplifier.
C_SLUG = "ocean-conscientiousness-suppressor-v2"
E_SLUG = "ocean-extraversion-amplifier-v3"
C_DIR = "conscientiousness/suppressor/v2"
E_DIR = "extraversion/amplifier/v3"
C_TRAIT_LOWER = "conscientiousness"
E_TRAIT_LOWER = "extraversion"

# Alphabetical combo slug (matches CanonicalCell.combo_slug)
COMBO_SLUG = "__".join(sorted([C_SLUG, E_SLUG]))

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Per-trait: rollout fingerprint for the 240×1 sweep on that trait's dataset.
# All four data sources (combo, baseline, single-adapter c, single-adapter e)
# live under the SAME fingerprint for a given trait now — the new soups were
# configured with the same MAX_SAMPLES/NUM_ROLLOUTS as the single-adapter
# Option B sweeps, so no dual-fingerprint fallback needed.
FP_BY_TRAIT: dict[str, str] = {
    "openness":          "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion":      "a961f641eb",
    "agreeableness":     "0705e3276a",
    "neuroticism":       "b2a49f1b4d",
}

OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "c_e_combo_delta"
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


def _combo_cell_hf_dir(fingerprint: str, c_scale: float, e_scale: float) -> str:
    """HF dir for the (C-adapter=c_scale, E-adapter=e_scale) combo cell."""
    spec = f"cell_{C_SLUG}{_fmt_scale(c_scale)}_{E_SLUG}{_fmt_scale(e_scale)}"
    return f"combos/{MODEL_SLUG}/{COMBO_SLUG}/{EVAL_NAME}/{fingerprint}/{spec}"


def _single_c_cell_hf_dir(fingerprint: str, c_scale: float) -> str:
    """c_minus_v2 alone at given scale."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{C_DIR}"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt_scale(c_scale)}"
    )


def _single_e_cell_hf_dir(fingerprint: str, e_scale: float) -> str:
    """e_plus_v3 alone at given scale."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{E_DIR}"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt_scale(e_scale)}"
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
    """Returns per-trait dict: {trait_title: {baseline, c_adapter, e_adapter, combo}}.

    ``c_adapter`` / ``e_adapter`` are internal keys for the c_minus_v2 and
    e_plus_v3 single-adapter cells. Display labels live in ``TARGET_DISPLAY``.
    """
    out: dict[str, dict[str, float | None]] = {}
    for trait_lower, fp in FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        print(f"\n[trait] {trait_title}  (fp={fp})")

        combo_mean    = _fetch(_combo_cell_hf_dir(fp, 1.0, 1.0), trait_lower)
        baseline_mean = _fetch(_baseline_hf_dir(fp), trait_lower)
        c_mean        = _fetch(_single_c_cell_hf_dir(fp, 1.0), trait_lower)
        e_mean        = _fetch(_single_e_cell_hf_dir(fp, 1.0), trait_lower)

        print(f"  baseline           : {baseline_mean}")
        print(f"  c_minus_v2 (+1)    : {c_mean}")
        print(f"  e_plus_v3  (+1)    : {e_mean}")
        print(f"  combo (+1, +1)     : {combo_mean}")

        out[trait_title] = {
            "baseline":  baseline_mean,
            "c_adapter": c_mean,
            "e_adapter": e_mean,
            "combo":     combo_mean,
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TARGET_ORDER = ["c_adapter", "e_adapter", "combo"]
# Short arrow labels to match the spider plots (C↓ = c_minus_v2, E↑ = e_plus_v3).
TARGET_DISPLAY = {
    "c_adapter": "C↓",
    "e_adapter": "E↑",
    "combo":     "C↓ × E↑",
}
TARGET_COLORS = {
    "c_adapter": BIG_FIVE_COLORS["Conscientiousness"],
    "e_adapter": BIG_FIVE_COLORS["Extraversion"],
    # Teal — deliberately outside the OCEAN palette so the combo bar is
    # unambiguously "not a trait colour". Avoid purple (collides with
    # Agreeableness), red/blue/green/orange (collide with the other four).
    "combo":     "#00838F",
}
TARGET_HATCHES = {
    "c_adapter": "//",
    "e_adapter": "\\\\",
    "combo":     "xx",
}

BASELINE_LEGEND_LABEL = "baseline Llama3.1-8b-Instruct"


def render(scores: dict[str, dict[str, float | None]], out_path: Path) -> None:
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
                deltas.append(val - base)
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
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=15, ha="right", fontsize=10)
    # Colour each trait tick label with its canonical OCEAN hue, matching
    # the spider plots in the same figure.
    for tick_label, trait in zip(ax.get_xticklabels(), OCEAN_TRAITS):
        tick_label.set_color(BIG_FIVE_COLORS[trait])
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
    print(f"[combo-delta] cache dir: {CACHE_DIR}")
    print(f"[combo-delta] out path:  {OUT_PATH}")
    scores = gather()
    # Persist raw scores alongside the figure for iteration / debugging.
    scores_path = CACHE_DIR / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"✓ saved scores {scores_path}")
    render(scores, OUT_PATH)


if __name__ == "__main__":
    main()

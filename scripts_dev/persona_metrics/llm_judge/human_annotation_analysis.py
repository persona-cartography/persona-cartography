#!/usr/bin/env python3
"""Analyze inter-rater agreement across human annotators and LLM judges.

Loads human annotation JSON files from scratch/annotation_results/<rater>/<trait>.json,
LLM judge raw scores from scratch/golden_calibration/<run_dir>/raw/<trait>_run_*.jsonl,
and gold (author) scores from data/judge_calibration/<trait>.jsonl.

Computes all pairwise agreement metrics and produces:
  - Agreement summary table (all raters: humans + LLM judges + gold)
  - Scatter plots: each rater vs gold
  - Confusion heatmaps: each rater vs gold
  - Bland-Altman plots: pairwise rater agreement

Usage::

    # Analyze agreeableness (all available raters)
    uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py \\
        --trait agreeableness

    # Exclude specific human raters
    uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py \\
        --trait agreeableness --exclude-raters H1

    # Analyze all traits with data
    uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src_dev.persona_metrics.judge_calibration import (
    quadratic_weighted_agreement,
    summarize_pair,
)
from src_dev.persona_metrics.llm_judge_agreement import _krippendorff_alpha_ordinal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLDEN_DATA_DIR = project_root / "data" / "judge_calibration"
ANNOTATION_DIR = project_root / "scratch" / "annotation_results"
CALIBRATION_DIR = project_root / "scratch" / "golden_calibration"
OUTPUT_DIR = project_root / "scratch" / "human_annotation_analysis"

SCORE_RANGE: dict[str, tuple[int, int]] = {
    "agreeableness": (-4, 4),
    "conscientiousness": (-4, 4),
    "extraversion": (-4, 4),
    "neuroticism": (-4, 4),
    "openness": (-4, 4),
    "coherence": (0, 10),
}

ALL_TRAITS = list(SCORE_RANGE.keys())

# Which LLM judge run dirs to use (most complete runs for each model).
# Recommended panel per README: Gemini Flash, Haiku, Kimi K2.
# GPT-5 Mini is retired (Azure 403s) — included only as fallback if others
# lack data for a trait.
LLM_JUDGE_RUNS: dict[str, str] = {
    # Original judges
    "Gemini Flash": "google_gemini-2.0-flash-001__r3__20260326T203008",
    "Kimi K2": "moonshotai_kimi-k2__r3__20260326T221255",
    "Haiku 3.5": "anthropic_claude-3.5-haiku__r3__20260407T172156",
    "DeepSeek V3": "deepseek_deepseek-chat-v3__r3__20260407T171943",
    "Llama 4 Scout": "meta-llama_llama-4-scout__r3__20260407T172122",
    "GPT-5 Mini": "openai_gpt-5-mini__r3__20260326T220614",
    # New candidates (2026-04-21, coherence + all OCEAN)
    "Gemini Flash Lite": "google_gemini-2.0-flash-lite-001__r3__20260421T101844",
    "Gemma 4 26B-A4B": "google_gemma-4-26b-a4b-it__r3__20260421T101900",
    "Llama 3.3 70B": "meta-llama_llama-3.3-70b-instruct__r3__20260421T141420",
    "Mistral Small 3.2": "mistralai_mistral-small-3.2-24b-instruct__r3__20260421T141453",
    "GPT-4.1 Nano": "openai_gpt-4.1-nano__r3__20260421T140412",
    # GPT-5 Nano excluded: too many empty/out-of-range responses
    "Qwen 2.5 72B": "qwen_qwen-2.5-72b-instruct__r3__20260421T140658",
    "Qwen 3 235B": "qwen_qwen3-235b-a22b-2507__r3__20260421T141134",
}

# Anonymisation mapping: real rater dir names → anonymous IDs for output/plots.
# Real names appear only in gitignored scratch/ dirs; all tracked output uses these IDs.
HUMAN_ANON_MAP: dict[str, str] = {
    "anton": "H1",
    "irakli": "H2",
    "mariia": "H3",
    "sid": "H4",
    # Already-anonymised dirs restored from HF judge_calibration/v2/human_scores
    # (verified identical to H1-H3 in the published analysis.json)
    "human_judge_1": "H1",
    "human_judge_2": "H2",
    "human_judge_3": "H3",
}

# Colours for plotting
RATER_COLOURS: dict[str, str] = {
    # Humans (anonymous IDs)
    "H1": "#ffe119",
    "H2": "#e6194b",
    "H3": "#f58231",
    "H4": "#bfef45",
    # LLM judges
    "Gemini Flash": "#4363d8",
    "Kimi K2": "#3cb44b",
    "Haiku 3.5": "#911eb4",
    "DeepSeek V3": "#42d4f4",
    "Llama 4 Scout": "#f032e6",
    "GPT-5 Mini": "#a9a9a9",
    # New candidates
    "Gemini Flash Lite": "#aaffc3",
    "Gemma 4 26B-A4B": "#808000",
    "Llama 3.3 70B": "#ffd8b1",
    "Mistral Small 3.2": "#000075",
    "GPT-4.1 Nano": "#469990",
    "Qwen 2.5 72B": "#9A6324",
    "Qwen 3 235B": "#800000",
    # Gold
    "gold": "#000000",
}

RATER_MARKERS: dict[str, str] = {
    "H1": "^",
    "H2": "o",
    "H3": "s",
    "H4": "D",
    "Gemini Flash": "P",
    "Kimi K2": "X",
    "Haiku 3.5": "h",
    "DeepSeek V3": "v",
    "Llama 4 Scout": "<",
    "GPT-5 Mini": "*",
    "Gemini Flash Lite": "1",
    "Gemma 4 26B-A4B": "2",
    "Llama 3.3 70B": "3",
    "Mistral Small 3.2": "4",
    "GPT-4.1 Nano": "+",
    "Qwen 2.5 72B": "8",
    "Qwen 3 235B": "p",
    "gold": "d",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_golden(trait: str) -> dict[str, dict]:
    """Load golden items keyed by id."""
    path = GOLDEN_DATA_DIR / f"{trait}.jsonl"
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            items[item["id"]] = item
    return items


_ANON_REVERSE = {v: k for k, v in HUMAN_ANON_MAP.items()}


def discover_human_raters(trait: str) -> list[str]:
    """Find all human raters who have annotation files for a given trait.

    Returns anonymised IDs (H1, H2, ...) sorted alphabetically.
    """
    raters = []
    if not ANNOTATION_DIR.exists():
        return raters
    for rater_dir in sorted(ANNOTATION_DIR.iterdir()):
        if not rater_dir.is_dir():
            continue
        annotation_file = rater_dir / f"{trait}.json"
        if annotation_file.exists():
            real_name = rater_dir.name
            anon_id = HUMAN_ANON_MAP.get(real_name, real_name)
            raters.append(anon_id)
    return sorted(raters)


def load_human_scores(rater: str, trait: str) -> tuple[dict[str, int], bool]:
    """Load a human rater's scores as ({item_id: score}, is_dummy).

    Args:
        rater: Anonymised rater ID (e.g. "H1") or real dir name.
    """
    real_name = _ANON_REVERSE.get(rater, rater)
    path = ANNOTATION_DIR / real_name / f"{trait}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    is_dummy = data.get("dummy", False)
    return {item["id"]: item["score"] for item in data["scores"]}, is_dummy


def load_llm_judge_scores(
    judge_name: str, trait: str, n_runs: int = 3
) -> dict[str, float]:
    """Load LLM judge scores (median across runs) as {item_id: median_score}.

    Args:
        judge_name: Key in LLM_JUDGE_RUNS.
        trait: Trait name.
        n_runs: Number of runs to load (default 3).

    Returns:
        {item_id: median_score} across available runs.
    """
    run_dir = CALIBRATION_DIR / LLM_JUDGE_RUNS[judge_name] / "raw"
    if not run_dir.exists():
        return {}

    scores_by_id: dict[str, list[int]] = defaultdict(list)
    for run_idx in range(n_runs):
        path = run_dir / f"{trait}_run_{run_idx}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            score = item.get("judge_score")
            if score is not None:
                lo, hi = SCORE_RANGE[trait]
                if lo <= score <= hi:
                    scores_by_id[item["id"]].append(score)

    return {
        iid: statistics.median(scores)
        for iid, scores in scores_by_id.items()
        if scores
    }


def discover_llm_judges(trait: str) -> list[str]:
    """Find which LLM judges have data for a trait."""
    judges = []
    for name, run_key in LLM_JUDGE_RUNS.items():
        run_dir = CALIBRATION_DIR / run_key / "raw"
        if (run_dir / f"{trait}_run_0.jsonl").exists():
            judges.append(name)
    return judges


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_all_raters(
    trait: str,
    golden: dict[str, dict],
    all_scores: dict[str, dict[str, float | int]],
    human_raters: list[str],
    llm_judges: list[str],
) -> dict[str, Any]:
    """Compute full pairwise agreement across all raters (humans + LLMs) and gold.

    Args:
        trait: Trait name.
        golden: Golden items keyed by id.
        all_scores: {rater_name: {item_id: score}} for all raters.
        human_raters: Names of human raters.
        llm_judges: Names of LLM judges.

    Returns:
        Analysis dict.
    """
    score_min, score_max = SCORE_RANGE[trait]
    item_ids = list(golden.keys())
    gold_scores_list = [golden[iid]["gold_score"] for iid in item_ids]

    rater_names = human_raters + llm_judges

    # Build aligned score lists
    aligned: dict[str, list[float | None]] = {}
    for rater in rater_names:
        scores = all_scores.get(rater, {})
        aligned[rater] = [scores.get(iid) for iid in item_ids]
    aligned["gold"] = [float(g) for g in gold_scores_list]

    all_names = rater_names + ["gold"]

    # --- Pairwise agreement for every pair ---
    pairwise = []
    for i, name_a in enumerate(all_names):
        for name_b in all_names[i + 1 :]:
            stats = summarize_pair(aligned[name_a], aligned[name_b])
            valid_a, valid_b = [], []
            for a, b in zip(aligned[name_a], aligned[name_b]):
                if a is not None and b is not None:
                    valid_a.append(int(round(a)))
                    valid_b.append(int(round(b)))
            stats["qwk"] = quadratic_weighted_agreement(
                valid_a, valid_b, score_min=score_min, score_max=score_max
            )
            pair_type = _pair_type(name_a, name_b, human_raters, llm_judges)
            pairwise.append({
                "rater_a": name_a,
                "rater_b": name_b,
                "type": pair_type,
                **stats,
            })

    # --- Each rater vs gold ---
    rater_vs_gold = []
    for rater in rater_names:
        entry = next(
            p for p in pairwise
            if (p["rater_a"] == rater and p["rater_b"] == "gold")
            or (p["rater_a"] == "gold" and p["rater_b"] == rater)
        )
        rater_vs_gold.append({
            "rater": rater,
            "is_human": rater in human_raters,
            **{k: entry[k] for k in ["n", "pearson", "spearman", "mae", "within_one", "exact", "qwk"]},
        })

    # --- Each rater vs human consensus (median and mean) ---
    # For each item, compute median and mean of human rater scores
    human_median_list: list[float | None] = []
    human_mean_list: list[float | None] = []
    for iid in item_ids:
        h_scores = [
            all_scores[r][iid]
            for r in human_raters
            if iid in all_scores.get(r, {}) and all_scores[r][iid] is not None
        ]
        human_median_list.append(statistics.median(h_scores) if h_scores else None)
        human_mean_list.append(
            round(statistics.mean(h_scores), 2) if h_scores else None
        )

    rater_vs_human_consensus = []
    llm_vs_human_mean = []
    llm_vs_human_median = []

    def _compare_to_ref(
        rater: str, ref: list[float | None], label: str,
    ) -> dict:
        pred = aligned.get(rater, [None] * len(item_ids))
        stats = summarize_pair(ref, pred)
        valid_r, valid_p = [], []
        for r, p in zip(ref, pred):
            if r is not None and p is not None:
                valid_r.append(int(round(r)))
                valid_p.append(int(round(p)))
        stats["qwk"] = quadratic_weighted_agreement(
            valid_r, valid_p, score_min=score_min, score_max=score_max,
        )
        return {
            "rater": rater,
            "is_human": rater in human_raters,
            "reference": label,
            **{k: stats[k] for k in ["n", "pearson", "spearman", "mae", "within_one", "exact", "qwk"]},
        }

    if len(human_raters) >= 2:
        # Full consensus table (humans via leave-one-out, LLMs/gold vs median)
        for rater in rater_names + ["gold"]:
            if rater in human_raters:
                loo_median: list[float | None] = []
                for iid in item_ids:
                    others = [
                        all_scores[r][iid]
                        for r in human_raters
                        if r != rater
                        and iid in all_scores.get(r, {})
                        and all_scores[r][iid] is not None
                    ]
                    loo_median.append(
                        statistics.median(others) if others else None
                    )
                rater_vs_human_consensus.append(
                    _compare_to_ref(rater, loo_median, "leave-one-out median")
                )
            else:
                rater_vs_human_consensus.append(
                    _compare_to_ref(rater, human_median_list, "human median")
                )

        # LLM judges vs human mean and median
        for judge in llm_judges:
            llm_vs_human_mean.append(
                _compare_to_ref(judge, human_mean_list, "human mean")
            )
            llm_vs_human_median.append(
                _compare_to_ref(judge, human_median_list, "human median")
            )
        # Also add gold for context
        llm_vs_human_mean.append(
            _compare_to_ref("gold", human_mean_list, "human mean")
        )
        llm_vs_human_median.append(
            _compare_to_ref("gold", human_median_list, "human median")
        )

    # --- Krippendorff alpha for subsets ---
    def _alpha(names: list[str]) -> float:
        item_ratings = []
        for iid in item_ids:
            ratings = []
            for name in names:
                scores = all_scores.get(name, {})
                s = scores.get(iid) if name != "gold" else golden[iid]["gold_score"]
                if s is not None:
                    ratings.append(int(round(s)))
            item_ratings.append(ratings)
        return _krippendorff_alpha_ordinal(
            item_ratings, score_min=score_min, score_max=score_max
        )

    alpha_humans = _alpha(human_raters) if len(human_raters) >= 2 else float("nan")
    alpha_llms = _alpha(llm_judges) if len(llm_judges) >= 2 else float("nan")
    alpha_humans_llms = (
        _alpha(human_raters + llm_judges)
        if human_raters and llm_judges
        else float("nan")
    )
    alpha_humans_gold = _alpha(human_raters + ["gold"]) if human_raters else float("nan")
    alpha_llms_gold = _alpha(llm_judges + ["gold"]) if llm_judges else float("nan")
    alpha_all = _alpha(rater_names + ["gold"])

    # --- Group summaries ---
    def _mean_stats(pairs: list[dict]) -> dict[str, float]:
        summary = {}
        for key in ["pearson", "spearman", "mae", "within_one", "exact", "qwk"]:
            values = [float(p[key]) for p in pairs if not math.isnan(float(p[key]))]
            if values:
                summary[f"mean_{key}"] = statistics.mean(values)
        return summary

    human_human = [p for p in pairwise if p["type"] == "human-human"]
    llm_llm = [p for p in pairwise if p["type"] == "llm-llm"]
    human_llm = [p for p in pairwise if p["type"] == "human-llm"]
    human_gold = [p for p in pairwise if p["type"] == "human-gold"]
    llm_gold = [p for p in pairwise if p["type"] == "llm-gold"]

    # --- Per-item detail ---
    per_item = []
    for idx, iid in enumerate(item_ids):
        rater_vals = {}
        for rater in rater_names:
            s = all_scores.get(rater, {}).get(iid)
            rater_vals[rater] = round(s, 1) if s is not None else None
        human_vals = [rater_vals[r] for r in human_raters if rater_vals[r] is not None]
        llm_vals = [rater_vals[r] for r in llm_judges if rater_vals[r] is not None]
        per_item.append({
            "id": iid,
            "gold_score": gold_scores_list[idx],
            "scores": rater_vals,
            "human_median": statistics.median(human_vals) if human_vals else None,
            "llm_median": statistics.median(llm_vals) if llm_vals else None,
            "human_std": round(statistics.stdev(human_vals), 2) if len(human_vals) >= 2 else 0.0,
            "llm_std": round(statistics.stdev(llm_vals), 2) if len(llm_vals) >= 2 else 0.0,
        })

    return {
        "trait": trait,
        "n_items": len(item_ids),
        "human_raters": human_raters,
        "llm_judges": llm_judges,
        "score_range": [score_min, score_max],
        "rater_vs_gold": rater_vs_gold,
        "rater_vs_human_consensus": rater_vs_human_consensus,
        "llm_vs_human_mean": llm_vs_human_mean,
        "llm_vs_human_median": llm_vs_human_median,
        "pairwise": pairwise,
        "group_summaries": {
            "human_human": _mean_stats(human_human),
            "llm_llm": _mean_stats(llm_llm),
            "human_llm": _mean_stats(human_llm),
            "human_gold": _mean_stats(human_gold),
            "llm_gold": _mean_stats(llm_gold),
        },
        "krippendorff_alpha": {
            "humans_only": alpha_humans,
            "llms_only": alpha_llms,
            "humans_and_llms": alpha_humans_llms,
            "humans_plus_gold": alpha_humans_gold,
            "llms_plus_gold": alpha_llms_gold,
            "all_raters_plus_gold": alpha_all,
        },
        "per_item": per_item,
    }


def _pair_type(
    a: str, b: str, humans: list[str], llms: list[str]
) -> str:
    a_h, b_h = a in humans, b in humans
    a_l, b_l = a in llms, b in llms
    a_g, b_g = a == "gold", b == "gold"
    if a_h and b_h:
        return "human-human"
    if a_l and b_l:
        return "llm-llm"
    if (a_h and b_l) or (a_l and b_h):
        return "human-llm"
    if (a_h and b_g) or (a_g and b_h):
        return "human-gold"
    if (a_l and b_g) or (a_g and b_l):
        return "llm-gold"
    return "other"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_analysis(analysis: dict) -> None:
    """Print a readable summary."""
    trait = analysis["trait"]
    humans = analysis["human_raters"]
    llms = analysis["llm_judges"]

    dummy = analysis.get("dummy_raters", [])
    print(f"\n{'=' * 70}")
    print(f"  {trait.upper()}  —  {analysis['n_items']} items")
    print(f"  Humans: {', '.join(humans)}  |  LLM judges: {', '.join(llms)}")
    if dummy:
        print(f"  *** DUMMY DATA: {', '.join(dummy)} ***")
    print(f"{'=' * 70}")

    # Rater vs gold table
    print(f"\n  Each rater vs Gold:")
    print(f"  {'Rater':<18} {'Type':<7} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'±1':>6} {'Exact':>6} {'QWK':>6}")
    print(f"  {'─' * 68}")
    for r in sorted(analysis["rater_vs_gold"], key=lambda x: -x["spearman"]):
        rtype = "Human" if r["is_human"] else "LLM"
        print(
            f"  {r['rater']:<18} {rtype:<7} {r['pearson']:>8.3f} {r['spearman']:>9.3f} "
            f"{r['mae']:>6.2f} {r['within_one']:>5.0%} {r['exact']:>5.0%} {r['qwk']:>6.3f}"
        )

    # Group summaries
    print(f"\n  Group agreement summaries (mean across pairs):")
    gs = analysis["group_summaries"]
    for group_name, label in [
        ("human_human", "Human-Human"),
        ("llm_llm", "LLM-LLM"),
        ("human_llm", "Human-LLM"),
        ("human_gold", "Human-Gold"),
        ("llm_gold", "LLM-Gold"),
    ]:
        s = gs.get(group_name, {})
        if not s:
            continue
        print(
            f"  {label:<18} Spearman={s.get('mean_spearman', float('nan')):.3f}  "
            f"MAE={s.get('mean_mae', float('nan')):.2f}  "
            f"QWK={s.get('mean_qwk', float('nan')):.3f}  "
            f"±1={s.get('mean_within_one', float('nan')):.0%}"
        )

    # Rater vs human consensus (full table)
    rvhc = analysis.get("rater_vs_human_consensus", [])
    if rvhc:
        print(f"\n  Each rater vs Human consensus:")
        print(f"  {'Rater':<18} {'Type':<7} {'Ref':<22} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'±1':>6} {'QWK':>6}")
        print(f"  {'─' * 84}")
        for r in sorted(rvhc, key=lambda x: -x["spearman"]):
            rtype = "Human" if r["is_human"] else ("Gold" if r["rater"] == "gold" else "LLM")
            print(
                f"  {r['rater']:<18} {rtype:<7} {r['reference']:<22} "
                f"{r['pearson']:>8.3f} {r['spearman']:>9.3f} "
                f"{r['mae']:>6.2f} {r['within_one']:>5.0%} {r['exact']:>5.0%} {r['qwk']:>6.3f}"
            )

    # LLM judges vs human mean/median
    for key, label in [
        ("llm_vs_human_mean", "LLM judges vs Human mean"),
        ("llm_vs_human_median", "LLM judges vs Human median"),
    ]:
        entries = analysis.get(key, [])
        if entries:
            print(f"\n  {label}:")
            print(f"  {'Rater':<18} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'±1':>6} {'QWK':>6}")
            print(f"  {'─' * 55}")
            for r in sorted(entries, key=lambda x: -x["spearman"]):
                rtype = "Gold" if r["rater"] == "gold" else "LLM"
                print(
                    f"  {r['rater']:<18} {r['pearson']:>8.3f} {r['spearman']:>9.3f} "
                    f"{r['mae']:>6.2f} {r['within_one']:>5.0%} {r['exact']:>5.0%} {r['qwk']:>6.3f}"
                )

    # Krippendorff
    ka = analysis["krippendorff_alpha"]
    print(f"\n  Krippendorff's alpha:")
    print(f"    Humans only:        {ka['humans_only']:.3f}")
    print(f"    LLMs only:          {ka['llms_only']:.3f}")
    print(f"    Humans + LLMs:      {ka['humans_and_llms']:.3f}")
    print(f"    Humans + gold:      {ka['humans_plus_gold']:.3f}")
    print(f"    LLMs + gold:        {ka['llms_plus_gold']:.3f}")
    print(f"    All + gold:         {ka['all_raters_plus_gold']:.3f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _dummy_suffix(analysis: dict) -> str:
    """Return a warning suffix for plot titles if dummy data is present."""
    dummy = analysis.get("dummy_raters", [])
    return f"\n[DUMMY: {', '.join(dummy)}]" if dummy else ""


def _get_colour(name: str) -> str:
    return RATER_COLOURS.get(name, "#888888")


def _get_marker(name: str) -> str:
    return RATER_MARKERS.get(name, "o")


def plot_rater_vs_gold_scatter(analysis: dict, output_dir: Path) -> None:
    """Scatter plot: each rater's scores vs gold, one subplot per rater."""
    trait = analysis["trait"]
    raters = analysis["human_raters"] + analysis["llm_judges"]
    n = len(raters)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    score_min, score_max = analysis["score_range"]

    for idx, rater in enumerate(raters):
        ax = axes[idx // ncols][idx % ncols]
        golds, preds = [], []
        for item in analysis["per_item"]:
            g = item["gold_score"]
            r = item["scores"].get(rater)
            if r is not None:
                golds.append(g)
                preds.append(r)

        # Jitter for visibility
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(golds))
        ax.scatter(
            np.array(golds) + jitter,
            np.array(preds) + jitter,
            c=_get_colour(rater),
            marker=_get_marker(rater),
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        # Perfect agreement line
        ax.plot(
            [score_min, score_max], [score_min, score_max],
            "k--", alpha=0.3, linewidth=1,
        )

        # Stats annotation
        entry = next(r2 for r2 in analysis["rater_vs_gold"] if r2["rater"] == rater)
        is_human = rater in analysis["human_raters"]
        rtype = "Human" if is_human else "LLM"
        ax.text(
            0.05, 0.95,
            f"ρ={entry['spearman']:.3f}\nMAE={entry['mae']:.2f}\nQWK={entry['qwk']:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Gold score", fontsize=10)
        ax.set_ylabel("Rater score", fontsize=10)
        ax.set_title(f"{rater} ({rtype})", fontsize=11, fontweight="bold",
                      color=_get_colour(rater))
        ax.set_xlim(score_min - 0.5, score_max + 0.5)
        ax.set_ylim(score_min - 0.5, score_max + 0.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{trait.title()} — Each rater vs Gold scores{_dummy_suffix(analysis)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_scatter_vs_gold.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter → {path}")


def plot_agreement_bars(analysis: dict, output_dir: Path) -> None:
    """Bar chart comparing agreement metrics across all raters vs gold."""
    trait = analysis["trait"]
    rater_vs_gold = sorted(analysis["rater_vs_gold"], key=lambda x: -x["spearman"])

    names = [r["rater"] for r in rater_vs_gold]
    is_human = [r["is_human"] for r in rater_vs_gold]
    colours = [_get_colour(n) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(names))

    for ax, metric, label, fmt in [
        (axes[0], "spearman", "Spearman ρ", ".3f"),
        (axes[1], "qwk", "Quadratic Weighted Kappa", ".3f"),
        (axes[2], "mae", "MAE (lower = better)", ".2f"),
    ]:
        values = [r[metric] for r in rater_vs_gold]
        bars = ax.bar(x, values, color=colours, alpha=0.85, edgecolor="white", linewidth=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )

        # Add human/LLM markers
        for i, human in enumerate(is_human):
            marker = "H" if human else "L"
            ax.text(
                i, -0.08 if metric != "mae" else max(values) * 1.08,
                marker, ha="center", fontsize=8, fontweight="bold",
                color="#333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{trait.title()} — Rater agreement with Gold (H=Human, L=LLM){_dummy_suffix(analysis)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_agreement_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bars → {path}")


def _plot_llm_vs_human_ref(
    entries: list[dict],
    trait: str,
    ref_label: str,
    output_dir: Path,
    filename_suffix: str,
) -> None:
    """Bar chart: LLM judges (+ gold) vs a human reference (mean or median)."""
    if not entries:
        return

    sorted_entries = sorted(entries, key=lambda x: -x["spearman"])
    names = [r["rater"] for r in sorted_entries]
    colours = [_get_colour(n) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(names) * 2.5), 5))
    x = np.arange(len(names))

    for ax, metric, label, fmt in [
        (axes[0], "spearman", "Spearman ρ", ".3f"),
        (axes[1], "qwk", "Quadratic Weighted Kappa", ".3f"),
        (axes[2], "mae", "MAE (lower = better)", ".2f"),
    ]:
        values = [r[metric] for r in sorted_entries]
        bars = ax.bar(x, values, color=colours, alpha=0.85, edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{trait.title()} — LLM judges vs {ref_label} of 3 human raters",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_{filename_suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {ref_label} bars → {path}")


def plot_vs_human_consensus(analysis: dict, output_dir: Path) -> None:
    """Two separate bar charts: LLMs vs human mean, and LLMs vs human median."""
    trait = analysis["trait"]
    _plot_llm_vs_human_ref(
        analysis.get("llm_vs_human_mean", []),
        trait, "mean", output_dir, "llm_vs_human_mean",
    )
    _plot_llm_vs_human_ref(
        analysis.get("llm_vs_human_median", []),
        trait, "median", output_dir, "llm_vs_human_median",
    )


def plot_confusion_heatmaps(analysis: dict, output_dir: Path) -> None:
    """Confusion heatmap: each rater vs gold."""
    trait = analysis["trait"]
    score_min, score_max = analysis["score_range"]
    score_range = list(range(score_min, score_max + 1))
    n_scores = len(score_range)
    idx_of = {s: i for i, s in enumerate(score_range)}

    raters = analysis["human_raters"] + analysis["llm_judges"]
    n = len(raters)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for ri, rater in enumerate(raters):
        ax = axes[ri // ncols][ri % ncols]
        mat = np.zeros((n_scores, n_scores), dtype=int)
        for item in analysis["per_item"]:
            g = item["gold_score"]
            r = item["scores"].get(rater)
            if r is not None:
                gi, rj = idx_of.get(g), idx_of.get(int(round(r)))
                if gi is not None and rj is not None:
                    mat[gi, rj] += 1

        im = ax.imshow(mat, aspect="equal", cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
        for i in range(n_scores):
            for j in range(n_scores):
                v = mat[i, j]
                if v > 0:
                    colour = "white" if v > mat.max() * 0.55 else "#333"
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color=colour)

        # Diagonal
        ax.plot([-0.5, n_scores - 0.5], [-0.5, n_scores - 0.5],
                "r--", alpha=0.4, linewidth=1)

        ax.set_xticks(range(n_scores))
        ax.set_yticks(range(n_scores))
        ax.set_xticklabels(score_range, fontsize=7)
        ax.set_yticklabels(score_range, fontsize=7)
        ax.set_xlabel("Rater score", fontsize=9)
        ax.set_ylabel("Gold score", fontsize=9)
        is_human = rater in analysis["human_raters"]
        rtype = "Human" if is_human else "LLM"
        ax.set_title(f"{rater} ({rtype})", fontsize=10, fontweight="bold",
                      color=_get_colour(rater))

    for ri in range(n, nrows * ncols):
        axes[ri // ncols][ri % ncols].set_visible(False)

    fig.suptitle(
        f"{trait.title()} — Gold vs Rater confusion (rows=gold, cols=rater){_dummy_suffix(analysis)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_confusion_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion → {path}")


def plot_bland_altman(analysis: dict, output_dir: Path) -> None:
    """Bland-Altman plots for selected pairwise comparisons.

    Shows: each human vs each LLM, and human-human pairs.
    X-axis = mean of pair, Y-axis = difference. Horizontal lines at mean diff ± 1.96 SD.
    """
    trait = analysis["trait"]
    humans = analysis["human_raters"]
    llms = analysis["llm_judges"]

    # Select interesting pairs
    pairs_to_plot = []
    # Human-LLM
    for h in humans:
        for l in llms:
            pairs_to_plot.append((h, l, "human-llm"))
    # Human-Human
    for i, h1 in enumerate(humans):
        for h2 in humans[i + 1:]:
            pairs_to_plot.append((h1, h2, "human-human"))

    if not pairs_to_plot:
        return

    n = len(pairs_to_plot)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (name_a, name_b, ptype) in enumerate(pairs_to_plot):
        ax = axes[idx // ncols][idx % ncols]
        means, diffs = [], []
        for item in analysis["per_item"]:
            a = item["scores"].get(name_a)
            b = item["scores"].get(name_b)
            if a is not None and b is not None:
                means.append((a + b) / 2)
                diffs.append(a - b)

        means_arr = np.array(means)
        diffs_arr = np.array(diffs)
        mean_diff = np.mean(diffs_arr)
        std_diff = np.std(diffs_arr, ddof=1)

        ax.scatter(means_arr, diffs_arr, alpha=0.6, s=40, c="#4363d8", edgecolors="white", linewidth=0.5)
        ax.axhline(mean_diff, color="red", linewidth=1.2, label=f"Mean = {mean_diff:.2f}")
        ax.axhline(mean_diff + 1.96 * std_diff, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(mean_diff - 1.96 * std_diff, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        # Shade limits of agreement
        ax.fill_between(
            ax.get_xlim(), mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff,
            alpha=0.07, color="red",
        )

        ax.set_xlabel(f"Mean ({name_a}, {name_b})", fontsize=9)
        ax.set_ylabel(f"{name_a} − {name_b}", fontsize=9)
        ax.set_title(f"{name_a} vs {name_b}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{trait.title()} — Bland-Altman plots (difference vs mean){_dummy_suffix(analysis)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_bland_altman.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Bland-Altman → {path}")


def plot_pairwise_agreement_matrix(analysis: dict, output_dir: Path) -> None:
    """Heatmap matrix of pairwise Spearman correlations across all raters + gold."""
    trait = analysis["trait"]
    all_names = analysis["human_raters"] + analysis["llm_judges"] + ["gold"]
    n = len(all_names)

    # Build correlation matrix
    corr = np.eye(n)
    for p in analysis["pairwise"]:
        i = all_names.index(p["rater_a"])
        j = all_names.index(p["rater_b"])
        val = p["spearman"] if not math.isnan(p["spearman"]) else 0
        corr[i, j] = val
        corr[j, i] = val

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            colour = "white" if abs(corr[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=colour, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(all_names, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

    # Highlight human vs LLM boundary
    n_humans = len(analysis["human_raters"])
    if n_humans > 0 and len(analysis["llm_judges"]) > 0:
        ax.axhline(n_humans - 0.5, color="black", linewidth=1.5, alpha=0.5)
        ax.axvline(n_humans - 0.5, color="black", linewidth=1.5, alpha=0.5)

    ax.set_title(
        f"{trait.title()} — Pairwise Spearman ρ (all raters + gold){_dummy_suffix(analysis)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / f"{trait}_pairwise_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved matrix → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human + LLM judge agreement")
    parser.add_argument(
        "--trait", choices=ALL_TRAITS, default=None,
        help="Trait to analyze. If omitted, analyzes all traits with data.",
    )
    parser.add_argument(
        "--exclude-raters", nargs="+", default=[],
        help="Human raters to exclude.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for plots and JSON.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    traits = [args.trait] if args.trait else ALL_TRAITS
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for trait in traits:
        human_raters = [
            r for r in discover_human_raters(trait)
            if r not in args.exclude_raters
        ]
        llm_judges = [
            j for j in discover_llm_judges(trait)
            if j not in args.exclude_raters
        ]

        if not human_raters and not llm_judges:
            continue

        golden = load_golden(trait)

        # Load all scores
        all_scores: dict[str, dict[str, float | int]] = {}
        dummy_raters: list[str] = []
        for rater in human_raters:
            scores, is_dummy = load_human_scores(rater, trait)
            all_scores[rater] = scores
            if is_dummy:
                dummy_raters.append(rater)
        for judge in llm_judges:
            all_scores[judge] = load_llm_judge_scores(judge, trait)

        analysis = analyze_all_raters(trait, golden, all_scores, human_raters, llm_judges)
        analysis["dummy_raters"] = dummy_raters
        print_analysis(analysis)
        all_results[trait] = analysis

        if not args.no_plots:
            trait_dir = output_dir / trait
            trait_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Generating plots for {trait}...")
            plot_rater_vs_gold_scatter(analysis, trait_dir)
            plot_agreement_bars(analysis, trait_dir)
            plot_vs_human_consensus(analysis, trait_dir)
            plot_confusion_heatmaps(analysis, trait_dir)
            plot_bland_altman(analysis, trait_dir)
            plot_pairwise_agreement_matrix(analysis, trait_dir)

    if not all_results:
        print("No traits with rater data found.")
        return

    # Save JSON
    json_path = output_dir / "analysis.json"

    def clean(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    json_path.write_text(json.dumps(clean(all_results), indent=2), encoding="utf-8")
    print(f"\nFull analysis saved to {json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibrate professional (Prolific) human labels against non-professional
human labels and LLM judge runs.

Extends the existing judge-calibration analysis
(scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py) with a
third rater group: professional Prolific annotators. Attention-check rows are
excluded from all analysis (reported only as QC).

Inputs:
  - Prolific labels (long CSV): respondent_id, is_attention_check, score,
    question, response. Items are matched to golden calibration items by
    normalised (question, response) text.
  - Non-professional human scores: scratch/annotation_results/<rater>/<trait>.json
    (restored from HF judge_calibration/v2/human_scores).
  - LLM judge runs: scratch/golden_calibration/<run>/raw/<trait>_run_*.jsonl
    (restored from HF judge_calibration/v2/judge_runs; median of 3 runs).
  - Golden (author) scores: data/judge_calibration/<trait>.jsonl.

Usage::

    uv run python scripts_dev/evals/llm_judge_sweep/prolific/prolific_judge_calibration.py \\
        --csv scripts_dev/evals/llm_judge_sweep/prolific/prolific_coherence_responses_long.csv

Outputs to scratch/prolific_calibration/: printed report, analysis JSON, plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from scripts_dev.persona_metrics.llm_judge.human_annotation_analysis import (
    CALIBRATION_DIR,
    LLM_JUDGE_RUNS,
    RATER_COLOURS,
    RATER_MARKERS,
    SCORE_RANGE,
    analyze_all_raters,
    discover_human_raters,
    discover_llm_judges,
    load_golden,
    load_human_scores,
    load_llm_judge_scores,
    plot_agreement_bars,
    plot_confusion_heatmaps,
    plot_pairwise_agreement_matrix,
    plot_rater_vs_gold_scatter,
)
from src_dev.persona_metrics.judge_calibration import (
    quadratic_weighted_agreement,
    summarize_pair,
)
from src_dev.persona_metrics.llm_judge_agreement import _krippendorff_alpha_ordinal

OUTPUT_DIR = project_root / "scratch" / "prolific_calibration"

# Colours/markers for Prolific raters (registered into the shared maps so the
# reused plotting functions pick them up).
_PRO_CMAP = plt.get_cmap("winter")


def _register_prolific_style(raters: list[str]) -> None:
    for i, r in enumerate(raters):
        RATER_COLOURS.setdefault(
            r, matplotlib.colors.to_hex(_PRO_CMAP(i / max(len(raters) - 1, 1)))
        )
        RATER_MARKERS.setdefault(r, "o")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split())


def load_prolific_scores(
    csv_path: Path, golden: dict[str, dict], score_shift: int = 0
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Load Prolific labels keyed by anonymised rater, excluding attention checks.

    Items are matched to golden items by normalised (question, response) text,
    falling back to response text alone.

    Args:
        csv_path: Long-format Prolific CSV.
        golden: Golden items keyed by id.
        score_shift: Added to every non-attention score. The OCEAN forms
            collect 0..8 while golden gold_score uses -4..4, so pass -4 for
            those traits (attention-check QC stays on the raw form scale).

    Returns:
        ({rater: {item_id: score}}, qc) where qc holds attention-check
        deviations and any unmatched rows.
    """
    by_qr = {(_norm(it["question"]), _norm(it["response"])): iid for iid, it in golden.items()}
    by_r = {_norm(it["response"]): iid for iid, it in golden.items()}

    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    respondents = sorted({r["respondent_id"] for r in rows})
    anon = {rid: f"P{idx + 1}" for idx, rid in enumerate(respondents)}

    scores: dict[str, dict[str, int]] = defaultdict(dict)
    attention: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []

    for row in rows:
        rater = anon[row["respondent_id"]]
        score = int(row["score"])
        if row["is_attention_check"].strip().lower() == "yes":
            expected = row.get("expected_score", "")
            attention[rater].append({
                "expected": int(expected) if expected.strip() else None,
                "score": score,
            })
            continue
        iid = by_qr.get((_norm(row["question"]), _norm(row["response"]))) or by_r.get(
            _norm(row["response"])
        )
        if iid is None:
            unmatched.append({"rater": rater, "question": row["question"][:80]})
            continue
        scores[rater][iid] = score + score_shift

    qc = {
        "rater_map": anon,
        "n_unmatched_rows": len(unmatched),
        "unmatched": unmatched,
        "attention_checks": {
            rater: {
                "n": len(checks),
                "max_abs_dev": max(
                    (abs(c["score"] - c["expected"]) for c in checks if c["expected"] is not None),
                    default=None,
                ),
            }
            for rater, checks in sorted(attention.items())
        },
    }
    return dict(scores), qc


def load_judge_runs_raw(judge_name: str, trait: str, n_runs: int = 3) -> dict[str, list[int]]:
    """Load per-run judge scores as {item_id: [run scores]} for intra-judge agreement."""
    run_dir = CALIBRATION_DIR / LLM_JUDGE_RUNS[judge_name] / "raw"
    lo, hi = SCORE_RANGE[trait]
    scores_by_id: dict[str, list[int]] = defaultdict(list)
    for run_idx in range(n_runs):
        path = run_dir / f"{trait}_run_{run_idx}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            score = item.get("judge_score")
            if score is not None and lo <= score <= hi:
                scores_by_id[item["id"]].append(score)
    return dict(scores_by_id)


# ---------------------------------------------------------------------------
# Group-aware analysis
# ---------------------------------------------------------------------------


def _pair_stats(
    ref: list[float | None], pred: list[float | None], score_min: int, score_max: int
) -> dict[str, float]:
    stats = summarize_pair(ref, pred)
    valid_r, valid_p = [], []
    for a, b in zip(ref, pred):
        if a is not None and b is not None:
            valid_r.append(int(round(a)))
            valid_p.append(int(round(b)))
    stats["qwk"] = quadratic_weighted_agreement(
        valid_r, valid_p, score_min=score_min, score_max=score_max
    )
    return stats


def _median_of(
    raters: list[str],
    all_scores: dict[str, dict[str, float | int]],
    item_ids: list[str],
    exclude: str | None = None,
) -> list[float | None]:
    out: list[float | None] = []
    for iid in item_ids:
        vals = [
            all_scores[r][iid]
            for r in raters
            if r != exclude and iid in all_scores.get(r, {})
        ]
        out.append(statistics.median(vals) if vals else None)
    return out


def group_analysis(
    trait: str,
    golden: dict[str, dict],
    all_scores: dict[str, dict[str, float | int]],
    nonpro: list[str],
    pro: list[str],
    llm_judges: list[str],
    base: dict[str, Any],
) -> dict[str, Any]:
    """Three-group calibration: pro humans vs non-pro humans vs LLM judges.

    Args:
        trait: Trait name.
        golden: Golden items keyed by id.
        all_scores: {rater: {item_id: score}} for every rater.
        nonpro: Non-professional human rater names.
        pro: Professional (Prolific) rater names.
        llm_judges: LLM judge names.
        base: Output of analyze_all_raters over all raters combined.

    Returns:
        Dict with group pair summaries, Krippendorff alphas per group, and
        consensus-reference comparisons.
    """
    score_min, score_max = SCORE_RANGE[trait]
    item_ids = list(golden.keys())
    gold_list: list[float | None] = [float(golden[iid]["gold_score"]) for iid in item_ids]

    def group_of(name: str) -> str:
        if name in pro:
            return "pro"
        if name in nonpro:
            return "nonpro"
        if name in llm_judges:
            return "llm"
        return "gold"

    # --- Pairwise summaries by group pair (from base pairwise) ---
    by_group_pair: dict[str, list[dict]] = defaultdict(list)
    for p in base["pairwise"]:
        key = "-".join(sorted([group_of(p["rater_a"]), group_of(p["rater_b"])]))
        by_group_pair[key].append(p)

    def _mean_stats(pairs: list[dict]) -> dict[str, float]:
        out = {"n_pairs": len(pairs)}
        for key in ["pearson", "spearman", "mae", "within_one", "exact", "qwk"]:
            vals = [float(p[key]) for p in pairs if not math.isnan(float(p[key]))]
            if vals:
                out[f"mean_{key}"] = statistics.mean(vals)
        return out

    group_pair_summaries = {k: _mean_stats(v) for k, v in sorted(by_group_pair.items())}

    # --- Krippendorff alphas per rater set ---
    def _alpha(names: list[str], include_gold: bool = False) -> float:
        item_ratings = []
        for idx, iid in enumerate(item_ids):
            ratings = [
                int(round(all_scores[r][iid]))
                for r in names
                if iid in all_scores.get(r, {})
            ]
            if include_gold:
                ratings.append(int(gold_list[idx]))
            item_ratings.append(ratings)
        return _krippendorff_alpha_ordinal(item_ratings, score_min=score_min, score_max=score_max)

    alphas = {
        "pro_only": _alpha(pro),
        "nonpro_only": _alpha(nonpro),
        "llm_only": _alpha(llm_judges),
        "pro_plus_nonpro": _alpha(pro + nonpro),
        "pro_plus_gold": _alpha(pro, include_gold=True),
        "nonpro_plus_gold": _alpha(nonpro, include_gold=True),
        "pro_plus_llm": _alpha(pro + llm_judges),
        "all_plus_gold": _alpha(pro + nonpro + llm_judges, include_gold=True),
    }

    # --- Consensus references ---
    pro_median = _median_of(pro, all_scores, item_ids)
    nonpro_median = _median_of(nonpro, all_scores, item_ids)

    def _aligned(rater: str) -> list[float | None]:
        return [all_scores.get(rater, {}).get(iid) for iid in item_ids]

    def _vs_ref(rater: str, ref: list[float | None], ref_label: str) -> dict:
        pred = gold_list if rater == "gold" else _aligned(rater)
        if rater in pro and ref is pro_median:
            ref = _median_of(pro, all_scores, item_ids, exclude=rater)
            ref_label = "pro leave-one-out median"
        stats = _pair_stats(ref, pred, score_min, score_max)
        return {"rater": rater, "group": group_of(rater), "reference": ref_label, **stats}

    everyone = pro + nonpro + llm_judges + ["gold"]
    vs_pro_median = [_vs_ref(r, pro_median, "pro median") for r in everyone]
    vs_nonpro_median = [
        _vs_ref(r, nonpro_median, "nonpro median") for r in pro + llm_judges + ["gold"]
    ]

    consensus_cross = {
        "pro_median_vs_gold": _pair_stats(gold_list, pro_median, score_min, score_max),
        "pro_median_vs_nonpro_median": _pair_stats(nonpro_median, pro_median, score_min, score_max),
        "nonpro_median_vs_gold": _pair_stats(gold_list, nonpro_median, score_min, score_max),
    }

    # --- Intra-judge agreement across repeated runs ---
    intra_judge = {}
    for judge in llm_judges:
        runs = load_judge_runs_raw(judge, trait)
        item_ratings = [runs.get(iid, []) for iid in item_ids]
        n_runs = max((len(v) for v in item_ratings), default=0)
        pair_rhos = []
        for a in range(n_runs):
            for b in range(a + 1, n_runs):
                ra = [v[a] if len(v) > max(a, b) else None for v in item_ratings]
                rb = [v[b] if len(v) > max(a, b) else None for v in item_ratings]
                s = summarize_pair(ra, rb)
                if not math.isnan(s["spearman"]):
                    pair_rhos.append(s["spearman"])
        intra_judge[judge] = {
            "alpha_across_runs": _krippendorff_alpha_ordinal(
                item_ratings, score_min=score_min, score_max=score_max
            ),
            "mean_pairwise_spearman": statistics.mean(pair_rhos) if pair_rhos else float("nan"),
            "n_runs": n_runs,
        }

    return {
        "group_pair_summaries": group_pair_summaries,
        "krippendorff_alpha": alphas,
        "vs_pro_median": vs_pro_median,
        "vs_nonpro_median": vs_nonpro_median,
        "consensus_cross": consensus_cross,
        "intra_judge": intra_judge,
        "pro_median": {iid: m for iid, m in zip(item_ids, pro_median)},
        "nonpro_median": {iid: m for iid, m in zip(item_ids, nonpro_median)},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_row(entry: dict) -> str:
    return (
        f"{entry['pearson']:>8.3f} {entry['spearman']:>9.3f} {entry['mae']:>6.2f} "
        f"{entry['within_one']:>5.0%} {entry['exact']:>5.0%} {entry['qwk']:>6.3f}"
    )


def print_report(trait: str, ga: dict[str, Any], qc: dict[str, Any], pro: list[str]) -> None:
    print(f"\n{'=' * 78}")
    print(f"  PROLIFIC CALIBRATION — {trait.upper()}")
    print(f"{'=' * 78}")

    print("\n  QC — attention checks (excluded from analysis):")
    for rater, info in qc["attention_checks"].items():
        print(f"    {rater}: n={info['n']}, max |score-expected| = {info['max_abs_dev']}")
    if qc["n_unmatched_rows"]:
        print(f"    WARNING: {qc['n_unmatched_rows']} unmatched rows")

    print("\n  Krippendorff's alpha (ordinal):")
    for key, val in ga["krippendorff_alpha"].items():
        print(f"    {key:<22} {val:.3f}")

    print("\n  Consensus cross-checks:")
    for key, s in ga["consensus_cross"].items():
        print(
            f"    {key:<30} rho={s['spearman']:.3f}  QWK={s['qwk']:.3f}  "
            f"MAE={s['mae']:.2f}  ±1={s['within_one']:.0%}"
        )

    print("\n  Group pair summaries (mean over rater pairs):")
    for key, s in ga["group_pair_summaries"].items():
        if "mean_spearman" not in s:
            continue
        print(
            f"    {key:<15} ({s['n_pairs']:>3} pairs)  rho={s['mean_spearman']:.3f}  "
            f"QWK={s['mean_qwk']:.3f}  MAE={s['mean_mae']:.2f}  ±1={s['mean_within_one']:.0%}"
        )

    header = f"  {'Rater':<20} {'Group':<8} {'Pearson':>8} {'Spearman':>9} {'MAE':>6} {'±1':>6} {'Exact':>6} {'QWK':>6}"
    print("\n  Every rater vs PRO consensus (median of Prolific raters; pro raters via leave-one-out):")
    print(header)
    print(f"  {'─' * 76}")
    for e in sorted(ga["vs_pro_median"], key=lambda x: -x["spearman"]):
        print(f"  {e['rater']:<20} {e['group']:<8} {_fmt_row(e)}")

    print("\n  Pro raters + judges vs NON-PRO consensus (median of H1-H3):")
    print(header)
    print(f"  {'─' * 76}")
    for e in sorted(ga["vs_nonpro_median"], key=lambda x: -x["spearman"]):
        print(f"  {e['rater']:<20} {e['group']:<8} {_fmt_row(e)}")

    print("\n  Intra-judge consistency across 3 repeated runs:")
    print(f"  {'Judge':<20} {'alpha':>7} {'mean pairwise rho':>19}")
    for judge, s in sorted(ga["intra_judge"].items(), key=lambda kv: -kv[1]["alpha_across_runs"]):
        print(f"  {judge:<20} {s['alpha_across_runs']:>7.3f} {s['mean_pairwise_spearman']:>19.3f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_vs_pro_consensus_bars(ga: dict[str, Any], trait: str, output_dir: Path) -> None:
    """Bar chart: every rater vs Prolific consensus, grouped colour-coded."""
    entries = sorted(ga["vs_pro_median"], key=lambda x: -x["spearman"])
    names = [e["rater"] for e in entries]
    group_colour = {"pro": "#1f77b4", "nonpro": "#ff7f0e", "llm": "#2ca02c", "gold": "#000000"}
    colours = [group_colour[e["group"]] for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(names) * 0.85), 5))
    x = np.arange(len(names))
    for ax, metric, label, fmt in [
        (axes[0], "spearman", "Spearman ρ", ".2f"),
        (axes[1], "qwk", "Quadratic Weighted Kappa", ".2f"),
        (axes[2], "mae", "MAE (lower = better)", ".2f"),
    ]:
        values = [e[metric] for e in entries]
        bars = ax.bar(x, values, color=colours, alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_colour.values()]
    labels = ["Prolific (pro, LOO)", "Non-pro human", "LLM judge", "Gold (author)"]
    fig.legend(handles, labels, loc="upper right", fontsize=9, ncol=4)
    fig.suptitle(f"{trait.title()} — agreement with Prolific consensus (median of 9 pro raters)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / f"{trait}_vs_pro_consensus_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pro-consensus bars → {path}")


def plot_consensus_scatters(
    ga: dict[str, Any],
    golden: dict[str, dict],
    all_scores: dict[str, dict[str, float | int]],
    llm_judges: list[str],
    trait: str,
    output_dir: Path,
) -> None:
    """Scatter panels: pro median vs gold, vs non-pro median, vs top judges."""
    item_ids = list(golden.keys())
    pro_med = [ga["pro_median"][iid] for iid in item_ids]
    score_min, score_max = SCORE_RANGE[trait]

    top_judges = sorted(
        (e for e in ga["vs_pro_median"] if e["group"] == "llm"),
        key=lambda x: -x["spearman"],
    )[:3]
    panels = [
        ("Gold (author)", [float(golden[iid]["gold_score"]) for iid in item_ids]),
        ("Non-pro median (H1-H3)", [ga["nonpro_median"][iid] for iid in item_ids]),
    ] + [
        (e["rater"], [all_scores[e["rater"]].get(iid) for iid in item_ids])
        for e in top_judges
    ]

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4), squeeze=False)
    rng = np.random.default_rng(SEED)
    for ax, (label, other) in zip(axes[0], panels):
        xs, ys = [], []
        for a, b in zip(other, pro_med):
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
        stats = _pair_stats(list(map(float, xs)), list(map(float, ys)), score_min, score_max)
        jx = rng.uniform(-0.12, 0.12, len(xs))
        jy = rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(np.array(xs) + jx, np.array(ys) + jy, alpha=0.7, s=45,
                   c="#1f77b4", edgecolors="white", linewidth=0.5)
        ax.plot([score_min, score_max], [score_min, score_max], "k--", alpha=0.3, linewidth=1)
        ax.text(0.05, 0.95,
                f"ρ={stats['spearman']:.3f}\nQWK={stats['qwk']:.3f}\nMAE={stats['mae']:.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Prolific median", fontsize=10)
        ax.set_xlim(score_min - 0.5, score_max + 0.5)
        ax.set_ylim(score_min - 0.5, score_max + 0.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.suptitle(f"{trait.title()} — Prolific consensus vs references", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_dir / f"{trait}_pro_consensus_scatters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved consensus scatters → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prolific vs non-pro humans vs LLM judges")
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Prolific long CSV; default: prolific_<trait>_responses_long.csv next to this script.",
    )
    parser.add_argument("--trait", default="coherence", choices=list(SCORE_RANGE.keys()))
    parser.add_argument(
        "--score-shift", type=int, default=None,
        help="Added to every non-attention Prolific score; default: SCORE_RANGE[trait][0] "
        "(the forms collect 0-based scales, golden OCEAN scores are -4..4).",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    trait = args.trait
    csv_path = args.csv or Path(__file__).parent / f"prolific_{trait}_responses_long.csv"
    score_shift = args.score_shift if args.score_shift is not None else SCORE_RANGE[trait][0]
    print(f"CSV: {csv_path}  (score shift: {score_shift:+d})")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    golden = load_golden(trait)
    pro_scores, qc = load_prolific_scores(csv_path, golden, score_shift=score_shift)
    pro = sorted(pro_scores.keys(), key=lambda r: int(r[1:]))
    nonpro = discover_human_raters(trait)
    llm_judges = discover_llm_judges(trait)
    _register_prolific_style(pro)

    all_scores: dict[str, dict[str, float | int]] = {}
    for rater in nonpro:
        all_scores[rater], _ = load_human_scores(rater, trait)
    for rater in pro:
        all_scores[rater] = pro_scores[rater]
    for judge in llm_judges:
        all_scores[judge] = load_llm_judge_scores(judge, trait)

    # Base pairwise machinery over all raters (pro + nonpro count as "humans")
    base = analyze_all_raters(trait, golden, all_scores, pro + nonpro, llm_judges)
    base["dummy_raters"] = []
    ga = group_analysis(trait, golden, all_scores, nonpro, pro, llm_judges, base)

    print_report(trait, ga, qc, pro)

    if not args.no_plots:
        trait_dir = output_dir / trait
        trait_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Generating plots for {trait}...")
        plot_vs_pro_consensus_bars(ga, trait, trait_dir)
        plot_consensus_scatters(ga, golden, all_scores, llm_judges, trait, trait_dir)
        plot_agreement_bars(base, trait_dir)
        plot_rater_vs_gold_scatter(base, trait_dir)
        plot_confusion_heatmaps(base, trait_dir)
        plot_pairwise_agreement_matrix(base, trait_dir)

    def clean(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    json_path = output_dir / f"{trait}_prolific_analysis.json"
    json_path.write_text(
        json.dumps(clean({"qc": qc, "group_analysis": ga, "base": base}), indent=2),
        encoding="utf-8",
    )
    print(f"\nFull analysis saved to {json_path}")


if __name__ == "__main__":
    main()

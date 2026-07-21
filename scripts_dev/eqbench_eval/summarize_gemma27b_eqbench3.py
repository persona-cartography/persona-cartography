"""Summarize EQBench3 gemma-3-27b neuroticism sweep results.

Reads runs.json and elo_results.json (output from eqbench3.py), extracts
per-variant rubric scores and descriptive criteria averages, writes summary.json,
and produces comparison charts (matplotlib).

Usage:
    python summarize_gemma27b_eqbench3.py [--output-dir DIR] [--upload]

Flags:
    --output-dir: Directory containing runs.json and elo_results.json
                  (default scratch/evals/eqbench3/gemma27b_n_sweep).
    --upload: Upload results to HuggingFace (default False).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_runs_json(runs_file: Path) -> dict[str, Any]:
    """Load runs.json file.

    Args:
        runs_file: Path to runs.json.

    Returns:
        Parsed JSON data.
    """
    if not runs_file.exists():
        logger.error(f"runs.json not found at {runs_file}")
        raise FileNotFoundError(runs_file)

    with open(runs_file, "r") as f:
        return json.load(f)


def _load_elo_results_json(elo_file: Path) -> dict[str, Any]:
    """Load elo_results.json file.

    Args:
        elo_file: Path to elo_results.json.

    Returns:
        Parsed JSON data.
    """
    if not elo_file.exists():
        logger.error(f"elo_results.json not found at {elo_file}")
        raise FileNotFoundError(elo_file)

    with open(elo_file, "r") as f:
        return json.load(f)


def _get_rubric_score_for_model(runs_data: dict[str, Any], model_name: str) -> Optional[float]:
    """Extract the final rubric score (0-100) for a model from runs data.

    eqbench3 persists its own authoritative headline score (0-20 scale) at
    run_data["results"]["average_rubric_score"], computed during the run with
    the vendored dir as CWD (so analysis-task criteria are properly included).
    We read that value and scale by 5.0 — exactly what upstream eqbench3.py does
    for its 0-100 display — rather than recomputing it, which would be CWD-
    fragile (the analysis criteria file is resolved relative to CWD) and could
    silently drop analysis tasks.

    Args:
        runs_data: Parsed runs.json.
        model_name: Logical model name (e.g., 'gemma3_27b_base').

    Returns:
        Final rubric score (0-100) or None if not found.
    """
    for run_key, run_data in runs_data.items():
        if run_data.get("model_name") != model_name:
            continue
        score_0_20 = run_data.get("results", {}).get("average_rubric_score")
        if isinstance(score_0_20, (int, float)):
            return round(score_0_20 * 5.0, 2)
        logger.warning(
            f"No numeric average_rubric_score for {model_name} "
            f"(value: {score_0_20!r})"
        )
        return None

    logger.warning(f"No run found for model: {model_name}")
    return None


def _load_descriptive_criteria(vendor_dir: Path) -> list[str]:
    """Return the 12 descriptive (non-headline) standard rubric criteria.

    The standard rubric scores 18 criteria (data/rubric_scoring_criteria.txt).
    Six are the "qualitative, higher-is-better" criteria that feed the headline
    0-100 score (per data/rubric_scoring_prompt.txt and
    STANDARD_ALLOWED_RUBRIC_CRITERIA in core/benchmark.py). The remaining 12 are
    "quantitative" style/personality descriptors (warmth, sycophantic,
    moralising, ...) recorded but excluded from the headline score. Those 12 are
    the descriptive signal we surface for the N+/N- comparison.

    Args:
        vendor_dir: Path to the vendored eqbench3 directory.

    Returns:
        Ordered list of the 12 descriptive criterion names.
    """
    headline_qualitative = {
        "demonstrated_empathy",
        "pragmatic_ei",
        "depth_of_insight",
        "social_dexterity",
        "emotional_reasoning",
        "message_tailoring",
    }
    criteria_file = vendor_dir / "data" / "rubric_scoring_criteria.txt"
    standard_criteria = [
        line.strip()
        for line in criteria_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [c for c in standard_criteria if c not in headline_qualitative]


def _get_descriptive_criteria_for_model(runs_data: dict[str, Any], model_name: str) -> dict[str, float]:
    """Extract average scores for the 12 descriptive criteria for a model.

    Args:
        runs_data: Parsed runs.json.
        model_name: Logical model name.

    Returns:
        Dict mapping criterion name to average score (0-20 scale), or None per
        criterion if no scores were recorded.
    """
    vendor_dir = Path(__file__).parent / "vendor" / "eqbench3"
    descriptive_criteria = _load_descriptive_criteria(vendor_dir)

    criterion_scores: dict[str, list[float]] = {crit: [] for crit in descriptive_criteria}

    for run_key, run_data in runs_data.items():
        if run_data.get("model_name") != model_name:
            continue

        scenario_tasks = run_data.get("scenario_tasks", {})
        for iter_idx, scenario_dict in scenario_tasks.items():
            if not isinstance(scenario_dict, dict):
                continue

            for scenario_id, task_info in scenario_dict.items():
                if not isinstance(task_info, dict):
                    continue
                if task_info.get("status") != "rubric_scored":
                    continue

                rubric_scores = task_info.get("rubric_scores", {})
                if not isinstance(rubric_scores, dict):
                    continue

                for crit in descriptive_criteria:
                    score = rubric_scores.get(crit)
                    if isinstance(score, (int, float)):
                        criterion_scores[crit].append(score)

    return {
        crit: (sum(scores) / len(scores) if scores else None)
        for crit, scores in criterion_scores.items()
    }


def _get_elo_metrics_for_model(
    elo_data: dict[str, Any],
    model_name: str,
) -> dict[str, Optional[float]]:
    """Extract ELO metrics for a model.

    In elo_results.json the ratings are stored with the logical model name as a
    top-level key mapping to {"elo", "elo_norm", "sigma", ...} (see
    core/elo.py final-ratings save). There is no "models" wrapper and the
    normalized field is "elo_norm" (not "elo_normalized").

    Args:
        elo_data: Parsed elo_results.json.
        model_name: Logical model name.

    Returns:
        Dict with 'elo', 'elo_norm', 'sigma' (or None if not found).
    """
    model_info = elo_data.get(model_name)
    if isinstance(model_info, dict) and "elo" in model_info:
        return {
            "elo": model_info.get("elo"),
            "elo_norm": model_info.get("elo_norm"),
            "sigma": model_info.get("sigma"),
        }

    logger.warning(f"No ELO metrics found for model: {model_name}")
    return {"elo": None, "elo_norm": None, "sigma": None}


def summarize_sweep(output_dir: Path) -> dict[str, Any]:
    """Summarize the entire sweep into a summary dict.

    Args:
        output_dir: Directory containing runs.json and elo_results.json.

    Returns:
        Summary dict with per-variant scores and metrics.
    """
    runs_file = output_dir / "runs.json"
    elo_file = output_dir / "elo_results.json"

    runs_data = _load_runs_json(runs_file)
    elo_data = _load_elo_results_json(elo_file)

    variants = [
        "gemma3_27b_base",
        "gemma3_27b_n_plus",
        "gemma3_27b_n_minus",
    ]

    summary = {"variants": {}}

    for variant in variants:
        logger.info(f"Extracting summary for {variant}")

        rubric_score = _get_rubric_score_for_model(runs_data, variant)
        descriptive_criteria = _get_descriptive_criteria_for_model(runs_data, variant)
        elo_metrics = _get_elo_metrics_for_model(elo_data, variant)

        summary["variants"][variant] = {
            "rubric_score": rubric_score,
            "descriptive_criteria": descriptive_criteria,
            "elo_metrics": elo_metrics,
        }

    return summary


def write_summary_json(summary: dict[str, Any], output_dir: Path) -> None:
    """Write summary.json to output directory.

    Args:
        summary: Summary dict.
        output_dir: Output directory.
    """
    summary_file = output_dir / "summary.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote summary to {summary_file}")


def plot_rubric_comparison(summary: dict[str, Any], output_dir: Path) -> None:
    """Create horizontal bar chart comparing rubric scores.

    Args:
        summary: Summary dict.
        output_dir: Output directory.
    """
    variants = list(summary["variants"].keys())
    scores = [summary["variants"][v]["rubric_score"] for v in variants]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.barh(variants, scores, color=colors)

    ax.set_xlabel("Rubric Score (0-100)")
    ax.set_title("EQBench3 Rubric Scores: Gemma-3-27b Neuroticism Sweep")
    ax.set_xlim(0, 100)

    for i, (v, score) in enumerate(zip(variants, scores)):
        if score is not None:
            ax.text(score + 1, i, f"{score:.1f}", va="center")

    plt.tight_layout()
    chart_file = output_dir / "rubric_comparison.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    logger.info(f"Wrote rubric comparison chart to {chart_file}")
    plt.close()


def plot_descriptive_criteria_heatmap(summary: dict[str, Any], output_dir: Path) -> None:
    """Create heatmap of descriptive criteria × variants.

    Args:
        summary: Summary dict.
        output_dir: Output directory.
    """
    variants = list(summary["variants"].keys())

    all_criteria = set()
    for var_data in summary["variants"].values():
        criteria = var_data.get("descriptive_criteria", {})
        all_criteria.update(criteria.keys())

    all_criteria = sorted(all_criteria)

    if not all_criteria:
        logger.warning("No descriptive criteria found; skipping heatmap")
        return

    scores_matrix = []
    for variant in variants:
        criteria_dict = summary["variants"][variant].get("descriptive_criteria", {})
        row = [criteria_dict.get(crit) for crit in all_criteria]
        scores_matrix.append(row)

    scores_matrix = np.array(scores_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(scores_matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=20)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticks(range(len(all_criteria)))
    ax.set_yticklabels(all_criteria)

    ax.set_title("Descriptive Criteria Scores Heatmap (0-20 scale)")

    for i in range(len(variants)):
        for j in range(len(all_criteria)):
            val = scores_matrix[i, j]
            if not np.isnan(val):
                text = ax.text(i, j, f"{val:.1f}", ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score (0-20)")

    plt.tight_layout()
    heatmap_file = output_dir / "descriptive_criteria_heatmap.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches="tight")
    logger.info(f"Wrote descriptive criteria heatmap to {heatmap_file}")
    plt.close()


def upload_results(output_dir: Path) -> None:
    """Upload results to HuggingFace.

    Args:
        output_dir: Directory containing results to upload.
    """
    from src_dev.utils.hf_hub import upload_folder_to_dataset_repo

    try:
        logger.info(f"Uploading results from {output_dir} to HuggingFace")
        upload_folder_to_dataset_repo(
            local_dir=output_dir,
            repo_id="persona-cartography/monorepo",
            path_in_repo="evals/eqbench3/gemma27b_n_sweep",
            commit_message="eqbench3 gemma-3-27b-it base/N+/N- PoC sweep",
        )
        logger.info("Upload complete")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


def main() -> int:
    """Parse arguments and run summary generation."""
    parser = argparse.ArgumentParser(
        description="Summarize EQBench3 gemma-3-27b neuroticism sweep results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/evals/eqbench3/gemma27b_n_sweep"),
        help="Output directory containing runs.json and elo_results.json.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to HuggingFace.",
    )

    args = parser.parse_args()

    try:
        logger.info("=== EQBench3 Gemma-3-27b Neuroticism Sweep Summary ===")
        logger.info(f"Reading from: {args.output_dir}")

        summary = summarize_sweep(args.output_dir)

        write_summary_json(summary, args.output_dir)
        plot_rubric_comparison(summary, args.output_dir)
        plot_descriptive_criteria_heatmap(summary, args.output_dir)

        logger.info("Summary generation complete")

        if args.upload:
            upload_results(args.output_dir)

        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

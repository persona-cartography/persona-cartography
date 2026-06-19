#!/usr/bin/env python3
"""Plot LLM judge scores vs LoRA scale factor for one or more judge runs.

Reads raw judge call JSONL files produced by ``ocean_judge_calibration.py``
and plots mean judge score ± 1 std vs LoRA scale, one line per rater.
Multiple judge run directories can be overlaid on the same figure (one subplot
per run / adapter).

Usage::

    # Single adapter
    python -m src_dev.visualisations.plot_judge_sweep \\
        --judge-dir scratch/judge_runs/gpt_4o_mini-gemini_flash_20-30211aa3a32c \\
        --title "Neuroticism old adapter"

    # Multiple adapters side-by-side
    python -m src_dev.visualisations.plot_judge_sweep \\
        --judge-dir scratch/judge_runs/...sft scratch/judge_runs/...old \\
        --labels sft old \\
        --output scratch/plots/neuroticism_sft_vs_old.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style ─────────────────────────────────────────────────────────────────────

_RATER_COLOURS = {
    # Recommended panel
    "qwen3_235b": "#e6194b",
    "gemma4_27b": "#3cb44b",  # Gemma 4 26B-A4B (google/gemma-4-26b-a4b-it)
    "llama33_70b": "#4363d8",
    # Legacy
    "gpt_4o_mini": "#a9a9a9",
    "gemini_flash_20": "#f58231",
    "haiku_35": "#911eb4",
}
_FALLBACK_COLOURS = ["#f58231", "#911eb4", "#42d4f4", "#f032e6"]
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _rater_colour(rater_id: str, idx: int) -> str:
    if rater_id in _RATER_COLOURS:
        return _RATER_COLOURS[rater_id]
    return _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)]


# ── Data loading ───────────────────────────────────────────────────────────────

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")


def _parse_scale(condition: str) -> float | None:
    m = _SCALE_RE.search(condition)
    return float(m.group(1)) if m else None


def load_judge_run(judge_dir: Path) -> dict[str, dict[float, list[float]]]:
    """Load all raw rater JSONL files from a judge run directory.

    Scores are averaged per response_id first, so std across the returned
    lists reflects response-level variance (not repeat noise).

    Returns:
        ``{rater_id: {scale: [per_response_means]}}``
    """
    raw_dir = judge_dir / "judge_calls" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"No judge_calls/raw dir found under {judge_dir}")

    # {rater_id: {scale: {response_id: [scores]}}}
    raw: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") != "success":
                    continue
                score = record.get("score")
                if score is None:
                    continue
                scale = _parse_scale(record.get("condition", ""))
                if scale is None:
                    continue
                rater_id = record.get("rater_id", path.stem)
                response_id = record.get("response_id", "unknown")
                raw[rater_id][scale][response_id].append(float(score))

    return {
        rater_id: {
            scale: [sum(scores) / len(scores) for scores in resp_scores.values()]
            for scale, resp_scores in scale_map.items()
        }
        for rater_id, scale_map in raw.items()
    }


def _std(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1))


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_judge_sweep(
    judge_dirs: list[Path],
    labels: list[str] | None = None,
    output: Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot judge score vs scale for one or more judge run directories.

    Args:
        judge_dirs: List of judge run directories (one subplot each).
        labels: Optional subplot labels (defaults to directory names).
        output: Output PNG path.
        title: Overall figure title.

    Returns:
        Path to saved figure.
    """
    n = len(judge_dirs)
    labels = labels or [d.name for d in judge_dirs]

    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = 2
        nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for dir_idx, (judge_dir, label) in enumerate(zip(judge_dirs, labels)):
        ax = axes[dir_idx // ncols][dir_idx % ncols]
        data = load_judge_run(judge_dir)

        # Per-rater lines (thin, semi-transparent when panel median is shown)
        n_raters = len(data)
        show_median = n_raters >= 2
        rater_alpha = 0.4 if show_median else 1.0
        rater_lw = 1.2 if show_median else 2.0

        # Collect per-rater means for median computation
        rater_means: dict[str, dict[float, float]] = {}

        for ridx, (rater_id, scale_scores) in enumerate(sorted(data.items())):
            scales = sorted(scale_scores)
            means = [sum(scale_scores[s]) / len(scale_scores[s]) for s in scales]
            cis = [_std(scale_scores[s]) for s in scales]
            rater_means[rater_id] = dict(zip(scales, means))

            colour = _rater_colour(rater_id, ridx)
            linestyle = _LINESTYLES[ridx % len(_LINESTYLES)]
            marker = _MARKERS[ridx % len(_MARKERS)]

            ax.plot(
                scales,
                means,
                marker=marker,
                linestyle=linestyle,
                color=colour,
                label=rater_id,
                linewidth=rater_lw,
                markersize=4 if show_median else 5,
                alpha=rater_alpha,
            )
            if any(ci > 0 for ci in cis) and not show_median:
                ax.errorbar(
                    scales,
                    means,
                    yerr=cis,
                    fmt="none",
                    color=colour,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.6,
                )

        # Panel median line (thick, black)
        if show_median:
            all_scales = sorted({s for rm in rater_means.values() for s in rm})
            median_vals = []
            for s in all_scales:
                vals = [rm[s] for rm in rater_means.values() if s in rm]
                median_vals.append(statistics.median(vals) if vals else float("nan"))
            ax.plot(
                all_scales,
                median_vals,
                marker="D",
                linestyle="-",
                color="black",
                label="panel median",
                linewidth=2.5,
                markersize=5,
                zorder=10,
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("LoRA scale factor", fontsize=10)
        ax.set_ylabel("Judge score (mean ± 1 std)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if n is odd
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()

    if output is None:
        output = judge_dirs[0].parent / "judge_sweep_plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output}")
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LLM judge scores vs LoRA scale factor."
    )
    parser.add_argument(
        "--judge-dir",
        nargs="+",
        type=Path,
        required=True,
        metavar="DIR",
        help="One or more judge run directories (one subplot per dir).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Subplot labels (one per --judge-dir). Defaults to dir names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <first-judge-dir>/../judge_sweep_plot.png)",
    )
    parser.add_argument("--title", default=None, help="Overall figure title.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_judge_sweep(
        judge_dirs=args.judge_dir,
        labels=args.labels,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()

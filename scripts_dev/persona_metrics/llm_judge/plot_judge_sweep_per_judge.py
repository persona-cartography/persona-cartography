#!/usr/bin/env python3
"""Plot per-judge (and panel-median) trait + coherence scores across LoRA scales.

Reads a sweep directory produced by ``scripts_dev/evals/llm_judge_sweep/runner.py``
(either a local path or a remote path on the HF monorepo dataset). The sweep
directory layout expected::

    <sweep_dir>/
      scale_{X}/
        judge_runs/
          {rater_id}/
            {metric_name}.jsonl     # one line per (response_id, repeat_index)

Produces a dual-axis plot showing, for each judge individually, trait score
(left) and coherence score (right) as a function of LoRA scale. Also overlays
the panel median in bold black.

Usage::

    # Remote HF path (recommended — downloads only what's needed)
    uv run python scripts_dev/persona_metrics/llm_judge/plot_judge_sweep_per_judge.py \\
        --hf-repo persona-shattering-lasr/monorepo \\
        --sweep-path fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4/evals/llm_judge_lora_scale_sweep/371d29192a \\
        --trait-metric openness_v2 \\
        --coherence-metric better_coherence_judge \\
        --output /tmp/openness_sweep_per_judge.png

    # Local path
    uv run python scripts_dev/persona_metrics/llm_judge/plot_judge_sweep_per_judge.py \\
        --local-path scratch/evals/my_sweep_dir \\
        --trait-metric openness_v2 \\
        --coherence-metric better_coherence_judge
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colour map for the recommended panel
_JUDGE_COLOURS = {
    "qwen3_235b": "#800000",
    "gemma4_27b": "#808000",  # Gemma 4 26B-A4B (google/gemma-4-26b-a4b-it)
    "llama33_70b": "#9A6324",
}
_MEDIAN_COLOUR = "#111111"
_FALLBACK_COLOURS = ["#4363d8", "#3cb44b", "#f58231", "#911eb4", "#42d4f4"]


_SCALE_RE = re.compile(r"scale_([+-]?\d+(?:\.\d+)?)")


def _parse_scale_from_path(name: str) -> float | None:
    m = _SCALE_RE.search(name)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------


def _download_from_hf(
    repo_id: str,
    sweep_path: str,
    trait_metric: str,
    coherence_metric: str,
    dest: Path,
) -> Path:
    """Download only the per-judge JSONL files we need for the plot."""
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = list(api.list_repo_tree(
        repo_id, path_in_repo=sweep_path, repo_type="dataset", recursive=True,
    ))

    wanted_suffixes = {
        f"/{trait_metric}.jsonl",
        f"/{coherence_metric}.jsonl",
    }
    to_download = [
        f.path for f in files
        if hasattr(f, "size") and f.size > 0
        and any(f.path.endswith(sfx) for sfx in wanted_suffixes)
        and "/judge_runs/" in f.path
    ]

    if not to_download:
        raise ValueError(
            f"Found 0 judge files matching {wanted_suffixes} under {sweep_path}; "
            "check --trait-metric and --coherence-metric."
        )

    print(f"Downloading {len(to_download)} judge files from HF to {dest} ...")
    for remote in to_download:
        hf_hub_download(
            repo_id, remote, repo_type="dataset",
            local_dir=str(dest), token=token,
        )

    # Return the local path that mirrors `sweep_path`
    return dest / sweep_path


# ---------------------------------------------------------------------------
# Data loading + aggregation
# ---------------------------------------------------------------------------


def _load_scores_by_scale(
    sweep_dir: Path,
    metric_name: str,
) -> dict[float, dict[str, list[float]]]:
    """Return ``{scale: {rater_id: [per_response_median_scores]}}``.

    Scores are collapsed per (rater, response_id) via median over repeats so
    std across the returned lists reflects response-level variance (not
    repeat noise).
    """
    # {scale: {rater: {response_id: [scores]}}}
    grouped: dict[float, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for scale_dir in sorted(sweep_dir.glob("scale_*")):
        if not scale_dir.is_dir():
            continue
        scale = _parse_scale_from_path(scale_dir.name)
        if scale is None:
            continue
        judge_runs = scale_dir / "judge_runs"
        if not judge_runs.exists():
            continue
        for rater_dir in sorted(judge_runs.iterdir()):
            if not rater_dir.is_dir():
                continue
            rater_id = rater_dir.name
            path = rater_dir / f"{metric_name}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") not in ("success", "parse_error"):
                    continue
                score = rec.get("score")
                if not isinstance(score, (int, float)):
                    continue
                resp_id = str(rec.get("response_id", ""))
                grouped[scale][rater_id][resp_id].append(float(score))

    # Collapse per-response via median across repeats
    out: dict[float, dict[str, list[float]]] = {}
    for scale, raters in grouped.items():
        out[scale] = {}
        for rater, resp_map in raters.items():
            out[scale][rater] = [
                statistics.median(vals) for vals in resp_map.values() if vals
            ]
    return out


def _mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) using naive ±1 SEM band."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    if len(values) == 1:
        m = values[0]
        return (m, m, m)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    sem = std / (len(values) ** 0.5)
    return (mean, mean - sem, mean + sem)


def _panel_median_across_judges(
    per_judge: dict[str, list[float]],
) -> list[float]:
    """For each response, take the median across judges; return list of medians.

    The input is {rater: [per_response_scores]}. Response-level alignment
    across judges is assumed (same number per rater, in the same order as
    produced by _load_scores_by_scale).
    """
    if not per_judge:
        return []
    lengths = {len(v) for v in per_judge.values()}
    if len(lengths) != 1:
        # Misaligned — fall back to within-judge median then median across judges
        return [
            statistics.median([statistics.median(v) for v in per_judge.values()])
        ]
    n = lengths.pop()
    out = []
    for i in range(n):
        out.append(statistics.median(v[i] for v in per_judge.values()))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _judge_colour(rater_id: str, idx: int) -> str:
    return _JUDGE_COLOURS.get(rater_id, _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)])


def plot_per_judge(
    sweep_dir: Path,
    trait_metric: str,
    coherence_metric: str,
    output: Path,
    title: str | None = None,
    show_median: bool = True,
) -> Path:
    """Dual-axis per-judge scale sweep plot.

    Panel A (left axis): trait score per judge as a function of LoRA scale.
    Panel B (right axis): coherence score per judge as a function of LoRA scale.
    Panel median overlaid as a bold black line if show_median.
    """
    trait_data = _load_scores_by_scale(sweep_dir, trait_metric)
    coh_data = _load_scores_by_scale(sweep_dir, coherence_metric)

    if not trait_data and not coh_data:
        raise ValueError(f"No judge data found under {sweep_dir}")

    # All raters seen across both metrics
    all_raters: set[str] = set()
    for d in (trait_data, coh_data):
        for _, per_judge in d.items():
            all_raters.update(per_judge.keys())
    raters_sorted = sorted(all_raters)

    fig, left_axis = plt.subplots(figsize=(11.0, 5.2))
    right_axis = left_axis.twinx()

    def _plot_metric(
        axis,
        data: dict[float, dict[str, list[float]]],
        is_coherence: bool,
        label_suffix: str,
    ) -> None:
        if not data:
            return
        scales = sorted(data.keys())

        # Per-judge line per metric
        for idx, rater in enumerate(raters_sorted):
            colour = _judge_colour(rater, idx)
            xs, ys, errs_lo, errs_hi = [], [], [], []
            for s in scales:
                vals = data.get(s, {}).get(rater, [])
                if not vals:
                    continue
                m, lo, hi = _mean_and_ci(vals)
                xs.append(s)
                ys.append(m)
                errs_lo.append(max(0.0, m - lo))
                errs_hi.append(max(0.0, hi - m))
            if not xs:
                continue
            axis.plot(
                xs, ys,
                marker="o" if not is_coherence else "s",
                linestyle="-" if not is_coherence else "--",
                color=colour,
                alpha=0.7,
                linewidth=1.4,
                markersize=5,
                label=f"{rater} ({label_suffix})",
            )
            axis.errorbar(
                xs, ys,
                yerr=[errs_lo, errs_hi],
                fmt="none", color=colour, alpha=0.35,
                capsize=2, capthick=0.8, elinewidth=0.8,
            )

        # Panel median
        if show_median and len(raters_sorted) >= 2:
            xs_m, ys_m = [], []
            for s in scales:
                per_judge = data.get(s, {})
                medians = _panel_median_across_judges(per_judge)
                if medians:
                    m, _, _ = _mean_and_ci(medians)
                    xs_m.append(s)
                    ys_m.append(m)
            if xs_m:
                axis.plot(
                    xs_m, ys_m,
                    marker="D",
                    linestyle="-" if not is_coherence else "--",
                    color=_MEDIAN_COLOUR,
                    linewidth=2.4,
                    markersize=6,
                    label=f"Panel median ({label_suffix})",
                    zorder=10,
                )

    _plot_metric(left_axis, trait_data, is_coherence=False,
                 label_suffix=trait_metric)
    _plot_metric(right_axis, coh_data, is_coherence=True,
                 label_suffix=coherence_metric)

    left_axis.set_xlabel("LoRA scale factor", fontsize=11)
    left_axis.set_ylabel(f"{trait_metric} (per-response mean)", fontsize=11)
    right_axis.set_ylabel(f"{coherence_metric} (per-response mean)", fontsize=11)

    # Y-axis conventions: OCEAN = -4..+4, coherence = 0..10
    left_axis.set_ylim(-4.2, 4.2)
    right_axis.set_ylim(-0.2, 10.2)

    left_axis.axvline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    left_axis.grid(alpha=0.25)

    # Combined legend
    h1, l1 = left_axis.get_legend_handles_labels()
    h2, l2 = right_axis.get_legend_handles_labels()
    left_axis.legend(
        h1 + h2, l1 + l2,
        loc="upper left", bbox_to_anchor=(1.11, 1.0),
        fontsize=8, frameon=False,
    )

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace dataset repo id (e.g. persona-shattering-lasr/monorepo).",
    )
    src.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Local sweep directory (alternative to --hf-repo + --sweep-path).",
    )
    parser.add_argument(
        "--sweep-path",
        default=None,
        help="Path within the HF repo to the sweep dir (required with --hf-repo).",
    )
    parser.add_argument(
        "--trait-metric",
        required=True,
        help="Trait metric name, e.g. openness_v2 or agreeableness_v2.",
    )
    parser.add_argument(
        "--coherence-metric",
        default="better_coherence_judge",
        help="Coherence metric name (default: better_coherence_judge).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <sweep_dir>/plots/llm_judge_scale_sweep_per_judge.png).",
    )
    parser.add_argument("--title", default=None, help="Overall figure title.")
    parser.add_argument(
        "--no-median",
        action="store_true",
        help="Suppress the panel median line.",
    )
    parser.add_argument(
        "--download-dest",
        type=Path,
        default=Path("scratch/hf_downloads/llm_judge_sweeps"),
        help="Local cache dir for HF downloads (default: scratch/hf_downloads/llm_judge_sweeps).",
    )
    args = parser.parse_args()

    if args.hf_repo:
        if not args.sweep_path:
            parser.error("--sweep-path is required when using --hf-repo.")
        sweep_dir = _download_from_hf(
            args.hf_repo, args.sweep_path,
            args.trait_metric, args.coherence_metric,
            args.download_dest,
        )
    else:
        sweep_dir = args.local_path

    output = args.output or sweep_dir / "plots" / "llm_judge_scale_sweep_per_judge.png"

    plot_per_judge(
        sweep_dir,
        args.trait_metric,
        args.coherence_metric,
        output,
        title=args.title or sweep_dir.name,
        show_median=not args.no_median,
    )


if __name__ == "__main__":
    main()

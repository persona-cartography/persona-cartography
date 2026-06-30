#!/usr/bin/env python3
"""Plot 2D heatmaps from prefix-concatenated evaluations.

X-axis: Position (std value) for trait A
Y-axis: Position (std value) for trait B
Color: OCEAN judge score or other metric

Customize which traits and target metric at the top of main().

Usage:
    uv run python scripts_dev/psychadapter_eval/plot_2d_prefix_heatmap.py \\
        --input scratch/psychadapter_eval/2d_prefix_concat_scored.jsonl \\
        --trait-a openness --trait-b neuroticism \\
        --target openness --output scratch/psychadapter_eval/heatmap_2d_o_n_target_o.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_scored_data(path: Path) -> list[dict]:
    """Load scored generations from JSONL."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def build_2d_heatmap(
    rows: list[dict],
    trait_a: str,
    trait_b: str,
    target_metric: str,
) -> tuple[dict, list[float], list[float], dict]:
    """Build 2D heatmap from prefix-concat data.

    Args:
        rows: Scored generation dicts with trait_a_pos, trait_b_pos, ocean_judge
        trait_a: Trait for X-axis (e.g., "openness")
        trait_b: Trait for Y-axis (e.g., "neuroticism")
        target_metric: Which trait's score to color by

    Returns:
        (heatmap_dict, pos_a_values, pos_b_values, metadata)
    """
    metric_key = f"{target_metric}_v2.score"

    heatmap = {}
    position_pairs = set()

    for row in rows:
        # Filter to matching trait pair
        if row.get("trait_a") != trait_a or row.get("trait_b") != trait_b:
            continue

        pos_a = row.get("trait_a_pos")
        pos_b = row.get("trait_b_pos")
        ocean_judge = row.get("ocean_judge") or {}
        score = ocean_judge.get(metric_key)

        if score is None or score == -99:
            continue

        score = float(score)
        if not (-4 <= score <= 4):
            continue

        key = (float(pos_a), float(pos_b))
        position_pairs.add(key)

        if key not in heatmap:
            heatmap[key] = []
        heatmap[key].append(score)

    # Aggregate
    aggregated = {}
    pos_a_vals = sorted(set(k[0] for k in position_pairs))
    pos_b_vals = sorted(set(k[1] for k in position_pairs))

    for pa in pos_a_vals:
        for pb in pos_b_vals:
            key = (pa, pb)
            if key in heatmap:
                scores = heatmap[key]
                aggregated[key] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "n": len(scores),
                }

    metadata = {
        "num_cells": len(aggregated),
        "total_samples": sum(len(v) for v in heatmap.values()),
        "avg_samples_per_cell": np.mean([len(v) for v in heatmap.values()]) if heatmap else 0,
    }

    return aggregated, pos_a_vals, pos_b_vals, metadata


def plot_2d_heatmap(
    heatmap_dict: dict,
    pos_a_vals: list[float],
    pos_b_vals: list[float],
    trait_a: str,
    trait_b: str,
    target_metric: str,
    out_path: Path,
) -> None:
    """Plot 2D heatmap."""
    # Build matrix
    matrix_mean = np.full((len(pos_b_vals), len(pos_a_vals)), np.nan)
    matrix_n = np.full((len(pos_b_vals), len(pos_a_vals)), 0, dtype=int)

    for i, pb in enumerate(pos_b_vals):
        for j, pa in enumerate(pos_a_vals):
            key = (pa, pb)
            if key in heatmap_dict:
                matrix_mean[i, j] = heatmap_dict[key]["mean"]
                matrix_n[i, j] = heatmap_dict[key]["n"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix_mean, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto", origin="lower")

    # Axes
    ax.set_xticks(range(len(pos_a_vals)))
    ax.set_xticklabels([f"{v:+.1f}" for v in pos_a_vals], rotation=45)
    ax.set_yticks(range(len(pos_b_vals)))
    ax.set_yticklabels([f"{v:+.1f}" for v in pos_b_vals])

    ax.set_xlabel(f"{trait_a.capitalize()} position (std)", fontsize=12, weight="bold")
    ax.set_ylabel(f"{trait_b.capitalize()} position (std)", fontsize=12, weight="bold")
    ax.set_title(
        f"2D Prefix Concatenation: {trait_a.capitalize()} × {trait_b.capitalize()}\n"
        f"Color = {target_metric.capitalize()} judge score | Composition in softmax attention",
        fontsize=14,
        weight="bold",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, label="OCEAN judge score (−4..+4)")

    # Cell values with sample count
    for i in range(len(pos_b_vals)):
        for j in range(len(pos_a_vals)):
            if not np.isnan(matrix_mean[i, j]):
                n = matrix_n[i, j]
                text = ax.text(
                    j, i, f"{matrix_mean[i, j]:.1f}\n(n={n})",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix_mean[i, j]) > 2 else "black",
                    weight="bold",
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Heatmap saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D prefix-concat heatmap"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to scored JSONL",
    )
    parser.add_argument(
        "--trait-a",
        default="openness",
        help="Trait for X-axis (default: openness)",
    )
    parser.add_argument(
        "--trait-b",
        default="neuroticism",
        help="Trait for Y-axis (default: neuroticism)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target metric for color (default: same as trait-a)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: scratch/psychadapter_eval/heatmap_2d_<traits>_target_<target>.png)",
    )
    args = parser.parse_args()

    trait_a = args.trait_a.lower()
    trait_b = args.trait_b.lower()
    target_metric = (args.target or trait_a).lower()

    if args.output is None:
        args.output = (
            Path(__file__).resolve().parents[2]
            / f"scratch/psychadapter_eval/heatmap_2d_{trait_a}_{trait_b}_target_{target_metric}.png"
        )

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        return

    print(f"Loading scored data from {args.input}...")
    rows = load_scored_data(args.input)
    print(f"  Loaded {len(rows)} rows")

    print(f"\nBuilding heatmap:")
    print(f"  X-axis: {trait_a} position (std)")
    print(f"  Y-axis: {trait_b} position (std)")
    print(f"  Color: {target_metric} OCEAN judge score")

    heatmap_dict, pos_a_vals, pos_b_vals, metadata = build_2d_heatmap(
        rows, trait_a, trait_b, target_metric
    )

    if not heatmap_dict:
        print(f"❌ No data found for trait pair {trait_a} × {trait_b}")
        print(f"Available trait pairs:")
        pairs = set()
        for row in rows:
            ta = row.get("trait_a")
            tb = row.get("trait_b")
            if ta and tb:
                pairs.add((ta, tb))
        for pair in sorted(pairs):
            print(f"    {pair[0]} × {pair[1]}")
        return

    print(f"  Position A (X) values: {pos_a_vals}")
    print(f"  Position B (Y) values: {pos_b_vals}")
    print(f"  Grid cells with data: {metadata['num_cells']}")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Avg samples per cell: {metadata['avg_samples_per_cell']:.1f}")

    plot_2d_heatmap(heatmap_dict, pos_a_vals, pos_b_vals, trait_a, trait_b, target_metric, args.output)

    print(f"\n✅ Heatmap saved → {args.output}")
    print(f"\nTry other target metrics:")
    for other_trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        if other_trait != target_metric:
            print(f"  --target {other_trait}")


if __name__ == "__main__":
    main()

"""2D Blend Heatmap: X=trait_a blend scale, Y=trait_b blend scale, color=judge score.

Plots a heatmap where:
  - X-axis: blend scale for trait A (0.0 to 1.0)
  - Y-axis: blend scale for trait B (0.0 to 1.0)
  - Color: OCEAN judge score for a target metric

Customize the target metric and traits at the top of main().

Usage:
    uv run python scripts_dev/psychadapter_eval/plot_2d_blend_heatmap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_scored_data(path: Path) -> list[dict]:
    """Load scored generations from JSONL."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def build_2d_heatmap_from_blend(
    rows: list[dict],
    trait_a: str,
    trait_b: str,
    target_metric: str,
) -> tuple[dict, list[float], list[float]]:
    """Build 2D heatmap from blend data.

    Args:
        rows: List of scored generation dicts with scale_a, scale_b, persona_metrics
        trait_a: Trait name for X-axis (e.g., "openness")
        trait_b: Trait name for Y-axis (e.g., "neuroticism")
        target_metric: Which trait's score to color by (e.g., "openness" or "neuroticism")

    Returns:
        (heatmap_dict, scale_a_values, scale_b_values)
        where heatmap_dict[(scale_a, scale_b)] = list of scores
    """
    metric_key = f"{target_metric}_v2.score"

    # Collect samples grouped by (scale_a, scale_b)
    heatmap = {}

    for row in rows:
        # Filter: only rows that have both trait_a and trait_b conditioning
        if row.get("trait_a") != trait_a or row.get("trait_b") != trait_b:
            continue

        scale_a = row.get("scale_a")
        scale_b = row.get("scale_b")
        pm = row.get("persona_metrics") or {}
        score = pm.get(metric_key)

        if score is None or score == -99:
            continue

        score = float(score)
        if not (-4 <= score <= 4):
            continue

        key = (float(scale_a), float(scale_b))
        if key not in heatmap:
            heatmap[key] = []
        heatmap[key].append(score)

    # Aggregate: mean of scores per cell
    aggregated = {}
    scale_a_vals = sorted(set(k[0] for k in heatmap.keys()))
    scale_b_vals = sorted(set(k[1] for k in heatmap.keys()))

    for sa in scale_a_vals:
        for sb in scale_b_vals:
            key = (sa, sb)
            if key in heatmap:
                scores = heatmap[key]
                aggregated[key] = np.mean(scores) if scores else np.nan

    return aggregated, scale_a_vals, scale_b_vals


def plot_2d_heatmap(
    heatmap_dict: dict,
    scale_a_vals: list[float],
    scale_b_vals: list[float],
    trait_a: str,
    trait_b: str,
    target_metric: str,
    out_path: Path,
) -> None:
    """Plot 2D heatmap from blend data."""
    # Build matrix
    matrix = np.full((len(scale_b_vals), len(scale_a_vals)), np.nan)
    for i, sb in enumerate(scale_b_vals):
        for j, sa in enumerate(scale_a_vals):
            key = (sa, sb)
            if key in heatmap_dict:
                matrix[i, j] = heatmap_dict[key]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto", origin="lower")

    # Labels for axes (0.0, 0.25, 0.5, 0.75, 1.0, etc.)
    ax.set_xticks(range(len(scale_a_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in scale_a_vals])
    ax.set_yticks(range(len(scale_b_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in scale_b_vals])

    ax.set_xlabel(f"LoRA blend scale: {trait_a.capitalize()}", fontsize=12)
    ax.set_ylabel(f"LoRA blend scale: {trait_b.capitalize()}", fontsize=12)
    ax.set_title(
        f"2D Blend Heatmap: {trait_a.capitalize()} × {trait_b.capitalize()} "
        f"→ {target_metric.capitalize()} judge score",
        fontsize=14,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, label="OCEAN judge score (−4..+4)")

    # Add cell values
    for i in range(len(scale_b_vals)):
        for j in range(len(scale_a_vals)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(
                    j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black", weight="bold"
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ Heatmap saved → {out_path}")


def main():
    """Main: customize here."""
    # === CUSTOMIZE THESE ===
    TRAIT_A = "openness"        # X-axis blend scale
    TRAIT_B = "neuroticism"     # Y-axis blend scale
    TARGET_METRIC = "openness"  # What to color-code (can be TRAIT_A, TRAIT_B, or another trait)
    # =======================

    root = Path(__file__).resolve().parents[2]
    scored_path = root / "scratch/psychadapter_eval/2d_blend_openness_neuroticism_scored.jsonl"
    out_dir = scored_path.parent
    out_path = out_dir / f"heatmap_2d_{TRAIT_A}_{TRAIT_B}_target_{TARGET_METRIC}.png"

    if not scored_path.exists():
        print(f"❌ Data not found: {scored_path}")
        print(f"\nGenerate it first with:")
        print(f"  uv run python scripts_dev/psychadapter_eval/gen_2d_trait_blend.py \\")
        print(f"      --trait-a {TRAIT_A.lower()} --trait-b {TRAIT_B.lower()}")
        return

    print(f"Loading scored data from {scored_path}...")
    rows = load_scored_data(scored_path)
    print(f"  Loaded {len(rows)} rows")

    print(f"\nBuilding heatmap:")
    print(f"  X-axis (trait A): {TRAIT_A} blend scale")
    print(f"  Y-axis (trait B): {TRAIT_B} blend scale")
    print(f"  Color: {TARGET_METRIC} OCEAN judge score")
    heatmap_dict, scale_a_vals, scale_b_vals = build_2d_heatmap_from_blend(
        rows, TRAIT_A, TRAIT_B, TARGET_METRIC
    )

    if not heatmap_dict:
        print(f"❌ No data found for trait pair {TRAIT_A} × {TRAIT_B}")
        print(f"Available trait pairs in the data:")
        pairs = set()
        for row in rows:
            ta = row.get("trait_a")
            tb = row.get("trait_b")
            if ta and tb:
                pairs.add((ta, tb))
        for pair in sorted(pairs):
            print(f"    {pair[0]} × {pair[1]}")
        return

    print(f"  Scale A (X) values: {scale_a_vals}")
    print(f"  Scale B (Y) values: {scale_b_vals}")
    print(f"  Grid cells with data: {len(heatmap_dict)}")

    plot_2d_heatmap(heatmap_dict, scale_a_vals, scale_b_vals, TRAIT_A, TRAIT_B, TARGET_METRIC, out_path)


if __name__ == "__main__":
    main()

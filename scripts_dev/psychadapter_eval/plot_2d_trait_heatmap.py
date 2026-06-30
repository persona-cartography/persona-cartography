"""2D Trait Heatmap: Cross-conditioning effects between two OCEAN traits.

This script creates a 2D heatmap showing how conditioning on trait A (x-axis)
at different intensities affects the judgment of trait B (y-axis).

Example: How does Openness conditioning intensity affect Neuroticism judgment?

Usage:
    uv run --python /path/to/python3.12 python scripts_dev/psychadapter_eval/plot_2d_trait_heatmap.py

Customize TRAIT_A and TRAIT_B at the top of main() to plot different pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_scored_data(path: Path) -> list[dict]:
    """Load scored generations from JSONL."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def build_2d_heatmap(
    rows: list[dict],
    trait_a: str,
    trait_b: str,
    metric_b: str | None = None,
) -> tuple[dict, list[float], list[float]]:
    """Build 2D heatmap: condition on trait_a, measure trait_b.

    Returns:
        (heatmap_dict, latent_a_values, latent_b_values)
        where heatmap_dict[latent_a][latent_b] = list of scores
    """
    if metric_b is None:
        metric_b = f"{trait_b}_v2"

    # Collect samples conditioned on trait_a, grouped by (latent_a, latent_b)
    heatmap = {}

    for row in rows:
        if row["trait"] == "baseline":
            continue

        # We want: conditioned on trait_a, what's the latent_b value?
        # But we only have one conditioning per sample...
        # Instead: condition on trait_a, and for each sample track its implicit trait_b signal
        # Actually, let's do: for all samples conditioned on trait_a,
        # group by the latent_a value, then measure how trait_b responds

        if row["trait"] != trait_a:
            continue

        latent_a = row["latent_value"]
        pm = row.get("persona_metrics") or {}
        score_b = pm.get(f"{metric_b}.score")

        if score_b is None or score_b == -99:
            continue

        score_b = float(score_b)
        if not (-4 <= score_b <= 4):
            continue

        # For 2D: we also want to bin by trait_b latent values if available
        # Since we don't have that directly, we'll use the judge's rating as proxy
        # and bin it into latent_b buckets based on the score

        # Discretize score into latent bins: score ranges [-4, 4] -> latent [-5, 5]
        latent_b_bin = round((score_b / 4.0) * 5.0)  # Map [-4,4] to [-5,5] bins
        latent_b_bin = max(-5.0, min(5.0, latent_b_bin))

        key = (latent_a, float(latent_b_bin))
        if key not in heatmap:
            heatmap[key] = []
        heatmap[key].append(score_b)

    # Aggregate: for each cell, take mean of up to 10 samples
    aggregated = {}
    latent_a_vals = sorted(set(k[0] for k in heatmap.keys()))
    latent_b_vals = sorted(set(k[1] for k in heatmap.keys()))

    for la in latent_a_vals:
        for lb in latent_b_vals:
            key = (la, lb)
            if key in heatmap:
                scores = heatmap[key][:10]  # Keep max 10 samples per cell
                aggregated[key] = np.mean(scores) if scores else np.nan

    return aggregated, latent_a_vals, latent_b_vals


def plot_2d_heatmap(
    heatmap_dict: dict,
    latent_a_vals: list[float],
    latent_b_vals: list[float],
    trait_a: str,
    trait_b: str,
    out_path: Path,
) -> None:
    """Plot 2D heatmap."""
    # Build matrix
    matrix = np.full((len(latent_b_vals), len(latent_a_vals)), np.nan)
    for i, lb in enumerate(latent_b_vals):
        for j, la in enumerate(latent_a_vals):
            key = (la, lb)
            if key in heatmap_dict:
                matrix[i, j] = heatmap_dict[key]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto", origin="lower")

    # Labels
    ax.set_xticks(range(len(latent_a_vals)))
    ax.set_xticklabels([f"{v:.0f}" for v in latent_a_vals])
    ax.set_yticks(range(len(latent_b_vals)))
    ax.set_yticklabels([f"{v:.0f}" for v in latent_b_vals])

    ax.set_xlabel(f"Conditioned: {trait_a} (latent value)", fontsize=12)
    ax.set_ylabel(f"Judged: {trait_b} (latent value)", fontsize=12)
    ax.set_title(
        f"2D Trait Heatmap: {trait_a.capitalize()} conditioning → {trait_b.capitalize()} response",
        fontsize=14,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, label="OCEAN judge score (−4..+4)")

    # Add cell values
    for i in range(len(latent_b_vals)):
        for j in range(len(latent_a_vals)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ Heatmap saved -> {out_path}")


def main():
    """Main: customize traits here."""
    # === CUSTOMIZE THESE ===
    TRAIT_A = "openness"  # X-axis: what we condition on
    TRAIT_B = "neuroticism"  # Y-axis: what we measure
    # =======================

    root = Path(__file__).resolve().parents[2]
    scored_path = root / "scratch/psychadapter_eval/n150/n150_scored.jsonl"
    out_dir = scored_path.parent
    out_path = out_dir / f"heatmap_{TRAIT_A}_x_{TRAIT_B}.png"

    print(f"Loading scored data from {scored_path}...")
    rows = load_scored_data(scored_path)

    print(f"Building heatmap: {TRAIT_A} × {TRAIT_B}...")
    heatmap_dict, lat_a, lat_b = build_2d_heatmap(rows, TRAIT_A, TRAIT_B)

    print(f"  {TRAIT_A} latent values: {lat_a}")
    print(f"  {TRAIT_B} latent values: {lat_b}")

    plot_2d_heatmap(heatmap_dict, lat_a, lat_b, TRAIT_A, TRAIT_B, out_path)


if __name__ == "__main__":
    main()

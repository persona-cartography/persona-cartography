"""Gemma-3-12B replication of the o_plus × n_plus soup heatmaps (Fig. 1 style).

Same two 5×5 heatmaps as ``main_o_n_soup_heatmaps.py`` (o_plus scale on x,
n_plus scale on y, mean Qwen3-235B judge score per cell) but for the
Gemma-3-12B ``ocean_const_paired_dpo`` adapters. Rendering is imported from the
main-figure script so the appendix panels are pixel-identical in style.

Data: ``analysis/grid_summary.jsonl`` of the two Gemma cell sweeps on
``persona-cartography/monorepo``, written by
``scripts_dev.evals.llm_judge_sweep.runner_cells`` with the
``configs/gemma12b_paired_dpo/o_plus_x_n_plus_on_{openness,neuroticism}``
configs (100 rollouts/cell, Qwen3-235B judge):

  combos/gemma-3-12b-it/ocean-neuroticism-amplifier-ocean_const_paired_dpo__\
ocean-openness-amplifier-ocean_const_paired_dpo/llm_judge_lora_scale_sweep/
    a35cb58aaa/analysis/grid_summary.jsonl   (judged openness_v2)
    be0206bb9c/analysis/grid_summary.jsonl   (judged neuroticism_v2)

The summary is used for all 25 cells (mean over the same per-response judge
scores the Llama figure averages from raw files). The 16 combo cells' raw
rollouts/judge files are also on HF under the same sweep roots; the 9
single/baseline cells' raw files were not uploaded by the batched sweep upload.

Paper figures:
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma12b.pdf
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma12b.pdf

Run with:
    uv run python scripts/visualisations/appendix_o_n_soup_heatmaps_gemma12b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
from huggingface_hub import hf_hub_download

# Importing the main-figure module also applies its PAPER_STYLE rcParams, so
# these appendix panels render with exactly the article styling.
from scripts.visualisations.main_o_n_soup_heatmaps import SCALES, render_heatmap
from src.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma12b.pdf",
    "appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma12b.pdf",
]

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "gemma-3-12b-it"
VERSION = "ocean_const_paired_dpo"

O_SLUG = f"ocean-openness-amplifier-{VERSION}"
N_SLUG = f"ocean-neuroticism-amplifier-{VERSION}"
COMBO_SLUG = "__".join(sorted([O_SLUG, N_SLUG]))
SWEEP_ROOT = f"combos/{MODEL_SLUG}/{COMBO_SLUG}/llm_judge_lora_scale_sweep"

# Each entry: (fingerprint, judged trait, paper output filename).
SOUPS = [
    ("a35cb58aaa", "openness",
     "appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma12b.pdf"),
    ("be0206bb9c", "neuroticism",
     "appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma12b.pdf"),
]


def build_grid(fingerprint: str, judged_trait: str) -> np.ndarray:
    """Return a 5×5 array of mean judge scores, shape [n_axis, o_axis]."""
    path = hf_hub_download(
        HF_REPO_ID, f"{SWEEP_ROOT}/{fingerprint}/analysis/grid_summary.jsonl",
        repo_type="dataset",
    )
    metric_name = f"{judged_trait}_v2"
    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("metric") != metric_name:
            continue
        by_slug = {e["slug"]: e["scale"] for e in row.get("cell_entries", [])}
        o_scale, n_scale = by_slug.get(O_SLUG, 0.0), by_slug.get(N_SLUG, 0.0)
        if o_scale not in SCALES or n_scale not in SCALES:
            continue
        grid[SCALES.index(n_scale), SCALES.index(o_scale)] = row["mean"]
        print(f"  ✓ (o={o_scale:+.0f}, n={n_scale:+.0f}): "
              f"mean = {row['mean']:+.3f} (n={row.get('n')})")
    return grid


def main() -> None:
    for fingerprint, judged_trait, out_rel in SOUPS:
        out_path = PAPER_FIGURES_DIR / out_rel
        print(f"\n[heatmap] fp={fingerprint} judged_trait={judged_trait}")
        print(f"           → {out_path}")
        grid = build_grid(fingerprint, judged_trait)
        if np.isnan(grid).any():
            missing = int(np.isnan(grid).sum())
            print(f"  ⚠ {missing} cell(s) missing from grid_summary")
        render_heatmap(
            grid, judged_trait=judged_trait, out_path=out_path,
            title_suffix=" (Gemma-3-12B)",
        )


if __name__ == "__main__":
    main()

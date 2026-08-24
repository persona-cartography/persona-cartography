"""Gemma replications of the o_plus × n_plus soup heatmaps (Fig. 1 style).

Same 5×5 heatmaps as ``main_o_n_soup_heatmaps.py`` (o_plus scale on x, n_plus
scale on y, mean Qwen3-235B judge score per cell) but for the Gemma-3-12B and
Gemma-3-27B ``ocean_const_paired_dpo`` adapters. Rendering is imported from the
main-figure script so the appendix panels are pixel-identical in style.

Data: ``analysis/grid_summary.jsonl`` of the four Gemma cell sweeps on
``persona-cartography/monorepo``, written by
``scripts_dev.evals.llm_judge_sweep.runner_cells`` with the
``configs/gemma{12b,27b}_paired_dpo/o_plus_x_n_plus_on_{openness,neuroticism}``
configs (100 rollouts/cell, Qwen3-235B judge):

  combos/{model}/ocean-neuroticism-amplifier-ocean_const_paired_dpo__\
ocean-openness-amplifier-ocean_const_paired_dpo/llm_judge_lora_scale_sweep/{fp}/
    gemma-3-12b-it: on_openness a35cb58aaa, on_neuroticism be0206bb9c
    gemma-3-27b-it: on_openness 086c6c6f8e, on_neuroticism c345fc5886

The summary is used for all 25 cells (mean over the same per-response judge
scores the Llama figure averages from raw files). The 16 combo cells' raw
rollouts/judge files are also on HF under the same sweep roots; the 9
single/baseline cells' raw files were not uploaded by the batched sweep upload.

Paper figures:
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma12b.pdf
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma12b.pdf
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma27b.pdf
    - paper/figures/appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma27b.pdf

Run with:
    uv run python scripts/visualisations/appendix_o_n_soup_heatmaps_gemma.py
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
    "appendix/model_comparison/fig_o_n_soup_heatmap_openness_gemma27b.pdf",
    "appendix/model_comparison/fig_o_n_soup_heatmap_neuroticism_gemma27b.pdf",
]

HF_REPO_ID = "persona-cartography/monorepo"
EVAL_NAME = "llm_judge_lora_scale_sweep"
VERSION = "ocean_const_paired_dpo"

O_SLUG = f"ocean-openness-amplifier-{VERSION}"
N_SLUG = f"ocean-neuroticism-amplifier-{VERSION}"
COMBO_SLUG = "__".join(sorted([O_SLUG, N_SLUG]))

# (model_slug, title label, {judged trait: fingerprint}, output filename tag)
MODELS = [
    ("gemma-3-12b-it", "Gemma-3-12B-IT",
     {"openness": "a35cb58aaa", "neuroticism": "be0206bb9c"}, "gemma12b"),
    ("gemma-3-27b-it", "Gemma-3-27B-IT",
     {"openness": "086c6c6f8e", "neuroticism": "c345fc5886"}, "gemma27b"),
]


def build_grid(model_slug: str, fingerprint: str, judged_trait: str) -> np.ndarray:
    """Return a 5×5 array of mean judge scores, shape [n_axis, o_axis]."""
    sweep_root = f"combos/{model_slug}/{COMBO_SLUG}/{EVAL_NAME}"
    path = hf_hub_download(
        HF_REPO_ID, f"{sweep_root}/{fingerprint}/analysis/grid_summary.jsonl",
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
    for model_slug, label, fps, tag in MODELS:
        for judged_trait, fingerprint in fps.items():
            out_rel = (
                f"appendix/model_comparison/"
                f"fig_o_n_soup_heatmap_{judged_trait}_{tag}.pdf"
            )
            out_path = PAPER_FIGURES_DIR / out_rel
            print(f"\n[heatmap] {model_slug} fp={fingerprint} "
                  f"judged_trait={judged_trait}")
            print(f"           → {out_path}")
            grid = build_grid(model_slug, fingerprint, judged_trait)
            if np.isnan(grid).any():
                missing = int(np.isnan(grid).sum())
                print(f"  ⚠ {missing} cell(s) missing from grid_summary")
            render_heatmap(
                grid, judged_trait=judged_trait, out_path=out_path,
                title_suffix=f" ({label})",
            )


if __name__ == "__main__":
    main()

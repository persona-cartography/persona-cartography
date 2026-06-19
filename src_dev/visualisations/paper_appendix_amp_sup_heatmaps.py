"""Per-trait amplifier × suppressor heatmaps for Appendix F (OCEAN results).

Five 5×5 LLM-judge heatmaps, one per OCEAN trait, showing the trait score when
the amplifier and suppressor LoRAs of that trait are stacked at varying scales.
Both axes use the LoRA scales {0.0, 0.5, 1.0, 1.5, 2.0}; the diagonal cells
(amp scale = sup scale) test whether the two adapters cancel and recover base
behaviour, while off-diagonal cells show what happens when one direction
dominates. The (0, 0) cell is the base model; the first row/column are the
single-adapter sweeps along each axis.

Cell means come from each sweep's pre-computed
``analysis/grid_summary.jsonl``, which already aggregates the per-message
LLM-judge scores into a mean (with bootstrap CI) for every (amp, sup) pair
including the baseline and single-adapter rows.

Layout: single figure with 5 heatmaps in a 2×3 grid (sixth panel hidden), shared
colorbar to the right. Style matches paper_main_o_n_soup_heatmaps.py (RdBu_r,
[-4, +4] axis range, per-cell numeric annotations).

Paper figures:
    - paper/figures/appendix/fig_F_amp_sup_heatmaps.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_amp_sup_heatmaps
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfFileSystem

from src_dev.visualisations import PAPER_FIGURES_DIR

# Match paper_main_o_n_soup_heatmaps.py styling.
SPINE_COLOR = "#2f3748"
AXIS_FACE = "#fbfbfc"
GRID_COLOR = "#dfe3e8"
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.facecolor": AXIS_FACE,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": SPINE_COLOR,
    "axes.titlecolor": SPINE_COLOR,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.color": SPINE_COLOR,
    "ytick.color": SPINE_COLOR,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

PAPER_FIGURES = [
    "appendix/fig_F_amp_sup_heatmaps.pdf",
]

HF_REPO = "datasets/persona-shattering-lasr/monorepo"
COMBO_BASE = f"{HF_REPO}/combos/llama-3.1-8b-it"

SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]

# (trait_name, judge_metric, combo_dir_segment, sweep_id, panel_letter)
TRAITS = [
    ("openness",          "openness_v2",          "ocean-openness-amplifier-vanton4_paired_dpo__ocean-openness-suppressor-vanton4_paired_dpo",                   "1817b5cf78", "(a)"),
    ("conscientiousness", "conscientiousness_v2", "ocean-conscientiousness-amplifier-vanton4_paired_dpo__ocean-conscientiousness-suppressor-vanton4_paired_dpo", "97743334f6", "(b)"),
    ("extraversion",      "extraversion_v2",      "ocean-extraversion-amplifier-vanton4_paired_dpo__ocean-extraversion-suppressor-vanton4_paired_dpo",           "47a37c39b7", "(c)"),
    ("agreeableness",     "agreeableness_v2",     "ocean-agreeableness-amplifier-vanton4_paired_dpo__ocean-agreeableness-suppressor-vanton4_paired_dpo",         "b2e6755ff3", "(d)"),
    ("neuroticism",       "neuroticism_v2",       "ocean-neuroticism-amplifier-vanton4_paired_dpo__ocean-neuroticism-suppressor-vanton4_paired_dpo",             "8b01e9fa2c", "(e)"),
]


def _parse_cell_tag(tag: str) -> tuple[float, float] | None:
    """Return (amp_scale, sup_scale) or None for cells that aren't in the grid.

    Three possible tag forms:
      - ``scale_+0.00`` (baseline; both 0.0)
      - ``ocean-<trait>-amplifier-vanton4_paired_dpo_scale_+0.50`` (amp only)
      - ``ocean-<trait>-suppressor-vanton4_paired_dpo_scale_+0.50`` (sup only)
      - ``cell_ocean-<trait>-amplifier-vanton4_paired_dpo+0.50_ocean-<trait>-suppressor-vanton4_paired_dpo+0.50``
    """
    if tag == "scale_+0.00":
        return 0.0, 0.0
    m = re.match(r"^ocean-[a-z]+-amplifier-vanton4_paired_dpo_scale_([+\-][\d.]+)$", tag)
    if m:
        return float(m.group(1)), 0.0
    m = re.match(r"^ocean-[a-z]+-suppressor-vanton4_paired_dpo_scale_([+\-][\d.]+)$", tag)
    if m:
        return 0.0, float(m.group(1))
    m = re.match(
        r"^cell_ocean-[a-z]+-amplifier-vanton4_paired_dpo([+\-][\d.]+)_"
        r"ocean-[a-z]+-suppressor-vanton4_paired_dpo([+\-][\d.]+)$",
        tag,
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def build_grid(fs: HfFileSystem, combo_dir: str, sweep_id: str, judge_metric: str) -> np.ndarray:
    """5x5 grid: rows indexed by sup_scale (y), cols by amp_scale (x)."""
    summary_path = (
        f"{COMBO_BASE}/{combo_dir}/llm_judge_lora_scale_sweep/{sweep_id}/"
        "analysis/grid_summary.jsonl"
    )
    text = fs.cat(summary_path).decode()
    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("metric") != judge_metric:
            continue
        coords = _parse_cell_tag(row.get("cell_tag", ""))
        if coords is None:
            print(f"    skip unparseable tag: {row.get('cell_tag')}")
            continue
        amp, sup = coords
        try:
            xi = SCALES.index(amp)
            yi = SCALES.index(sup)
        except ValueError:
            continue
        grid[yi, xi] = row["mean"]
    n_filled = int(np.isfinite(grid).sum())
    print(f"  loaded {n_filled}/{grid.size} cells")
    return grid


def main() -> None:
    fs = HfFileSystem()
    grids: dict[str, np.ndarray] = {}
    for trait, judge, combo_dir, sweep_id, _letter in TRAITS:
        print(f"\n[{trait}] sweep={sweep_id}")
        grids[trait] = build_grid(fs, combo_dir, sweep_id, judge)

    # 2×3 grid; last cell hidden. Shared colorbar.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    vmax = 4.0
    im = None

    for ax, (trait, _judge, _combo_dir, _sweep_id, letter) in zip(axes.flat, TRAITS):
        grid = grids[trait]
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

        ax.set_xticks(range(len(SCALES)))
        ax.set_yticks(range(len(SCALES)))
        ax.set_xticklabels([f"{s:g}" for s in SCALES])
        ax.set_yticklabels([f"{s:g}" for s in SCALES])
        ax.set_xlabel("Amplifier scale")
        ax.set_ylabel("Suppressor scale")
        ax.set_title(f"{letter} {trait.capitalize()}", loc="left", pad=6)

        # Per-cell numeric annotation, plus a "(base)" tag at (0,0).
        base_xi = SCALES.index(0.0)
        base_yi = SCALES.index(0.0)
        for yi in range(len(SCALES)):
            for xi in range(len(SCALES)):
                v = grid[yi, xi]
                if np.isnan(v):
                    label, colour = "—", "gray"
                else:
                    label = f"{v:+.2f}"
                    colour = "white" if abs(v) > 2.0 else "black"
                ax.text(xi, yi, label, ha="center", va="center", fontsize=8.5, color=colour)
                if xi == base_xi and yi == base_yi and not np.isnan(v):
                    ax.text(xi, yi - 0.34, "(base)", ha="center", va="center",
                            fontsize=7.5, color=colour, style="italic")

    # Hide the unused 6th panel.
    axes.flat[-1].set_visible(False)

    # Shared colorbar in the unused panel area.
    if im is not None:
        cbar_ax = fig.add_axes([0.72, 0.10, 0.018, 0.32])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Trait judge score", rotation=270, labelpad=15)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "fig_F_amp_sup_heatmaps.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

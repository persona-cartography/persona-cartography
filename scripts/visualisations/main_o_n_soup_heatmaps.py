"""Pair of heatmaps for the o_plus × n_plus (ocean_const_paired_dpo) 1:1 soup.

Two 5x5 heatmaps (o_plus scale on x, n_plus scale on y), one per judged trait:

 - openness: rollouts from the ``o_plus_x_n_plus_on_openness`` sweep
   (``data/ocean_open_ended/openness.jsonl``, fingerprint ``1817b5cf78``),
   judged on openness_v2.
 - neuroticism: rollouts from the ``o_plus_x_n_plus_on_neuroticism`` sweep
   (``data/ocean_open_ended/neuroticism.jsonl``, fingerprint ``8b01e9fa2c``),
   judged on neuroticism_v2.

Reads the bundled subtree at
``persona-shattering-lasr/monorepo::evals/heatmaps_o_n``. Cells are flat under
``on_<trait>/<cell_label>/judge_runs/qwen3_235b/<trait>_v2.jsonl`` — no
canonical-tier path resolution required.

Paper figures:
    - paper/figures/main/fig_1_o_n_soup_heatmap_openness.pdf
    - paper/figures/main/fig_1_o_n_soup_heatmap_neuroticism.pdf

Run with:
    uv run python scripts/visualisations/main_o_n_soup_heatmaps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv

load_dotenv(project_root / ".env")


# ---------------------------------------------------------------------------
# Paper figure style (mirrors `pab.analysis.science_plots.PAPER_STYLE` —
# inlined here because that module does not live in this repo).
# ---------------------------------------------------------------------------

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
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "xtick.color": SPINE_COLOR,
    "ytick.color": SPINE_COLOR,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

from src.evals.cell_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
)
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.heatmap_common import hydrate_judge_file, mean_score

PAPER_FIGURES = [
    "main/fig_1_o_n_soup_heatmap_openness.pdf",
    "main/fig_1_o_n_soup_heatmap_neuroticism.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
RATER_ID = "qwen3_235b"
BUNDLE_PATH_IN_REPO = "evals/heatmaps_o_n"

ADAPTER_O_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/ocean_const_paired_dpo"
    "/lora/openness_amplifying_full-persona"
)
ADAPTER_N_PLUS = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo"
    "/lora/neuroticism_amplifying_full-persona"
)

SCALES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# (bundle subdir, judged trait, paper output filename)
HEATMAPS = [
    ("on_openness", "openness", "main/fig_1_o_n_soup_heatmap_openness.pdf"),
    ("on_neuroticism", "neuroticism", "main/fig_1_o_n_soup_heatmap_neuroticism.pdf"),
]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "o_n_soup_heatmaps"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _cell_label(o_scale: float, n_scale: float) -> str:
    """Canonical cell label, matching CanonicalCell.variant_label()."""
    cell = CanonicalCell.from_scales(
        [(ADAPTER_O_PLUS, o_scale), (ADAPTER_N_PLUS, n_scale)]
    )
    return cell.variant_label()


def _judge_repo_path(subdir: str, label: str, judged_trait: str) -> str:
    metric_name = f"{judged_trait}_v2"
    return f"{BUNDLE_PATH_IN_REPO}/{subdir}/{label}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


# Hydration + scoring use the shared helpers in
# ``src.visualisations.heatmap_common`` (``hydrate_judge_file`` /
# ``mean_score``); cell-label resolution stays local because this soup reads a
# flat bundle subtree rather than canonical-tier paths.


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_grid(subdir: str, judged_trait: str) -> np.ndarray:
    """Return a 5x5 array of mean scores, shape [n_axis, o_axis].

    Rows (y axis) indexed by SCALES (n_plus), columns (x axis) by SCALES
    (o_plus). NaN for cells that aren't on HF yet.
    """
    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    for yi, n_scale in enumerate(SCALES):
        for xi, o_scale in enumerate(SCALES):
            label = _cell_label(o_scale, n_scale)
            hf_path = _judge_repo_path(subdir, label, judged_trait)
            local = hydrate_judge_file(hf_path, CACHE_DIR, repo_id=HF_REPO_ID)
            if local is None:
                print(f"  ⚠ (o={o_scale:+.0f}, n={n_scale:+.0f}): missing on HF")
                continue
            mean = mean_score(local)
            if mean is None:
                print(f"  ⚠ (o={o_scale:+.0f}, n={n_scale:+.0f}): no valid scores")
                continue
            grid[yi, xi] = mean
            print(f"  ✓ (o={o_scale:+.0f}, n={n_scale:+.0f}): mean = {mean:+.3f}")
    return grid


def render_heatmap(
    grid: np.ndarray,
    *,
    judged_trait: str,
    out_path: Path,
) -> None:
    trait_title = judged_trait.capitalize()

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    # Judge scale is -4..+4; centre the diverging colormap at 0.
    vmax = 4.0
    im = ax.imshow(
        grid,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )

    ax.set_xticks(range(len(SCALES)))
    ax.set_yticks(range(len(SCALES)))
    ax.set_xticklabels([f"{s:+g}" for s in SCALES])
    ax.set_yticklabels([f"{s:+g}" for s in SCALES])
    # Axis labels carry the "← suppress | amplify →" semantics inline. When
    # the y-axis label is rotated 90° CCW, the leading "←" points downward
    # (toward the bottom = suppress) and the trailing "→" points upward
    # (toward the top = amplify), which matches the data orientation.
    ax.set_xlabel("←  suppress       openness       amplify  →")
    ax.set_ylabel("←  suppress       neuroticism       amplify  →")
    ax.set_title(f"{trait_title} judge score", loc="left", pad=8)

    # Per-cell annotations. The (0, 0) baseline gets a "(base)" tag below
    # its numeric value.
    n_y, n_x = grid.shape
    base_xi = SCALES.index(0.0)
    base_yi = SCALES.index(0.0)
    for yi in range(n_y):
        for xi in range(n_x):
            val = grid[yi, xi]
            if np.isnan(val):
                label = "—"
                color = "gray"
            else:
                label = f"{val:+.2f}"
                color = "white" if abs(val) > 2.0 else "black"
            ax.text(xi, yi, label, ha="center", va="center", fontsize=9, color=color)
            if xi == base_xi and yi == base_yi and not np.isnan(val):
                ax.text(
                    xi,
                    yi - 0.32,
                    "(base)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    style="italic",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(judged_trait, rotation=270, labelpad=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Save both .pdf (vector, embedded fonts via fonttype=42) and .png
    # (raster, dpi=300) per the paper style guide.
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ saved {out_path}")
    print(f"✓ saved {png_path}")


def main() -> None:
    print(f"[heatmap] cache dir: {CACHE_DIR}")
    for subdir, judged_trait, out_rel in HEATMAPS:
        out_path = PAPER_FIGURES_DIR / out_rel
        print(f"\n[heatmap] subdir={subdir} judged_trait={judged_trait}")
        print(f"           → {out_path}")
        grid = build_grid(subdir, judged_trait)
        render_heatmap(grid, judged_trait=judged_trait, out_path=out_path)


if __name__ == "__main__":
    main()

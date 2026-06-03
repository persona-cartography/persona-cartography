"""Pair of heatmaps for the c_minus × e_minus (vanton4) 1:1 soup.

Two 5x5 heatmaps (c_minus scale on x, e_minus scale on y), one per judged
trait:

 - conscientiousness: rollouts from the
   ``c_minus_x_e_minus_on_conscientiousness`` sweep
   (``data/ocean_open_ended/conscientiousness.jsonl``, fingerprint
   ``97743334f6``), judged on conscientiousness_v2.
 - extraversion: rollouts from the ``c_minus_x_e_minus_on_extraversion``
   sweep (``data/ocean_open_ended/extraversion.jsonl``, fingerprint
   ``47a37c39b7``), judged on extraversion_v2.

Pulls mean scores for each of 25 canonical cells directly from HF and
caches them locally so subsequent runs are fast. Cells that land in
different canonical tiers live at different HF paths — handled explicitly:
 - (0, 0): baseline at ``combos/{model}/_baseline/...``
 - (x, 0), x != 0: single-adapter c_minus at the c_minus eval dir
 - (0, y), y != 0: single-adapter e_minus at the e_minus eval dir
 - (x, y), both nonzero: combo cell at the combo_slug dir

Paper figures:
    - paper/figures/main/fig_1_c_e_soup_heatmap_conscientiousness.pdf
    - paper/figures/main/fig_1_c_e_soup_heatmap_extraversion.pdf

Run with:
    uv run python -m scripts.figures.main_c_e_soup_heatmaps
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

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.heatmap_common import hydrate_judge_file, mean_score

PAPER_FIGURES = [
    "main/fig_1_c_e_soup_heatmap_conscientiousness.pdf",
    "main/fig_1_c_e_soup_heatmap_extraversion.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"

C_SLUG = "ocean-conscientiousness-suppressor-vanton4"
E_SLUG = "ocean-extraversion-suppressor-vanton4"
C_TRAIT = "conscientiousness"
E_TRAIT = "extraversion"

SCALES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Soups produced by the c_minus × e_minus configs in vanton4_qwen3.
# Each entry: (fingerprint, judged trait name, paper output filename).
SOUPS = [
    ("97743334f6", C_TRAIT, "main/fig_1_c_e_soup_heatmap_conscientiousness.pdf"),
    ("47a37c39b7", E_TRAIT, "main/fig_1_c_e_soup_heatmap_extraversion.pdf"),
]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "c_e_soup_heatmaps"

# Combo slug is adapter slugs joined with "__" in alphabetical order.
COMBO_SLUG = "__".join(sorted([C_SLUG, E_SLUG]))


# ---------------------------------------------------------------------------
# Path helpers (match CanonicalCell.hf_dir)
# ---------------------------------------------------------------------------

def _format_scale(x: float) -> str:
    # Matches format_scale() in src_dev/evals/cell_sweep/cell_identity.py.
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _cell_hf_dir(c_scale: float, e_scale: float, fingerprint: str) -> str:
    """Canonical HF dir for a (c_minus=c_scale, e_minus=e_scale) cell."""
    if c_scale == 0.0 and e_scale == 0.0:
        return f"combos/{MODEL_SLUG}/_baseline/{EVAL_NAME}/{fingerprint}"
    if e_scale == 0.0:
        return (
            f"fine_tuning/{MODEL_SLUG}/ocean/{C_TRAIT}/suppressor/vanton4"
            f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_format_scale(c_scale)}"
        )
    if c_scale == 0.0:
        return (
            f"fine_tuning/{MODEL_SLUG}/ocean/{E_TRAIT}/suppressor/vanton4"
            f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_format_scale(e_scale)}"
        )
    # Combo: both scales non-zero. Slugs sorted alphabetically; conscientiousness
    # comes before extraversion.
    spec = f"cell_{C_SLUG}{_format_scale(c_scale)}_{E_SLUG}{_format_scale(e_scale)}"
    return f"combos/{MODEL_SLUG}/{COMBO_SLUG}/{EVAL_NAME}/{fingerprint}/{spec}"


def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


# Hydration + scoring use the shared helpers in
# ``src.visualisations.heatmap_common`` (``hydrate_judge_file`` /
# ``mean_score``); cell-path resolution stays local because the two soup
# scripts resolve HF layouts differently.


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_grid(fingerprint: str, judged_trait: str) -> np.ndarray:
    """Return a 5x5 array of mean scores, shape [e_axis, c_axis].

    Rows (y axis) indexed by SCALES (e_minus), columns (x axis) by SCALES
    (c_minus). NaN for cells that aren't on HF yet.
    """
    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    metric_name = f"{judged_trait}_v2"
    for yi, e_scale in enumerate(SCALES):
        for xi, c_scale in enumerate(SCALES):
            hf_path = _judge_hf_path(
                _cell_hf_dir(c_scale, e_scale, fingerprint),
                metric_name,
            )
            local = hydrate_judge_file(hf_path, CACHE_DIR, repo_id=HF_REPO_ID)
            if local is None:
                print(f"  ⚠ (c={c_scale:+.0f}, e={e_scale:+.0f}): missing on HF")
                continue
            mean = mean_score(local)
            if mean is None:
                print(f"  ⚠ (c={c_scale:+.0f}, e={e_scale:+.0f}): no valid scores")
                continue
            grid[yi, xi] = mean
            print(f"  ✓ (c={c_scale:+.0f}, e={e_scale:+.0f}): mean = {mean:+.3f}")
    return grid


def render_heatmap(
    grid: np.ndarray,
    *,
    judged_trait: str,
    out_path: Path,
) -> None:
    trait_title = judged_trait.capitalize()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
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
    ax.set_xlabel(f"{C_SLUG.split('-')[1]} suppressor scale (c_minus)")
    ax.set_ylabel(f"{E_SLUG.split('-')[1]} suppressor scale (e_minus)")
    ax.set_title(f"{trait_title} judge score on {judged_trait} prompts")

    # Per-cell annotations.
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            val = grid[yi, xi]
            if np.isnan(val):
                label = "—"
                color = "gray"
            else:
                label = f"{val:+.2f}"
                # White text on dark cells, black on light.
                color = "white" if abs(val) > 2.0 else "black"
            ax.text(xi, yi, label, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"mean {judged_trait}_v2 (Qwen3-235B judge)", rotation=270, labelpad=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


def main() -> None:
    print(f"[heatmap] cache dir: {CACHE_DIR}")
    for fingerprint, judged_trait, out_rel in SOUPS:
        out_path = PAPER_FIGURES_DIR / out_rel
        print(f"\n[heatmap] soup fp={fingerprint} judged_trait={judged_trait}")
        print(f"           → {out_path}")
        grid = build_grid(fingerprint, judged_trait)
        render_heatmap(grid, judged_trait=judged_trait, out_path=out_path)


if __name__ == "__main__":
    main()

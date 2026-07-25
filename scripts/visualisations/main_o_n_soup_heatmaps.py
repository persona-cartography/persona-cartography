"""Pair of heatmaps for the o_plus × n_plus (ocean_const_paired_dpo) 1:1 soup.

Two 5x5 heatmaps (o_plus scale on x, n_plus scale on y), one per judged trait:

 - openness: rollouts from the ``o_plus_x_n_plus_on_openness`` sweep
   (``data/ocean_open_ended/openness.jsonl``, fingerprint ``1817b5cf78``),
   judged on openness_v2.
 - neuroticism: rollouts from the ``o_plus_x_n_plus_on_neuroticism`` sweep
   (``data/ocean_open_ended/neuroticism.jsonl``, fingerprint ``8b01e9fa2c``),
   judged on neuroticism_v2.

Reads the bundled subtree at
``persona-cartography/monorepo::evals/heatmaps_o_n``. Cells are flat under
``on_<trait>/<cell_label>/judge_runs/qwen3_235b/<trait>_v2.jsonl`` — no
canonical-tier path resolution required.

Paper figures:
    - paper/figures/main/fig_1_o_n_soup_heatmap_openness.pdf
    - paper/figures/main/fig_1_o_n_soup_heatmap_neuroticism.pdf

Run with:
    uv run python scripts/visualisations/main_o_n_soup_heatmaps.py
"""

from __future__ import annotations

import re
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
    "font.family": "sans-serif",
    # Force DejaVu Serif (bundled with matplotlib, always present) so this
    # figure reproduces its committed appearance: DejaVu renders the ←/→
    # axis-arrow glyphs and a true bold title, both of which Times — when it
    # happens to be installed — silently drops.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.facecolor": AXIS_FACE,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": SPINE_COLOR,
    "axes.titlecolor": SPINE_COLOR,
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

from huggingface_hub import HfFileSystem

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.heatmap_common import hydrate_judge_file, mean_score

PAPER_FIGURES = [
    "main/fig_1_o_n_soup_heatmap_openness.pdf",
    "main/fig_1_o_n_soup_heatmap_neuroticism.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-cartography/monorepo"
RATER_ID = "qwen3_235b"
BUNDLE_PATH_IN_REPO = "evals/heatmaps_o_n"

# The soup is openness-amplifier × neuroticism-amplifier. Cells are discovered
# by listing the bundle subtree (see ``_discover_cells``) rather than by
# constructing names, so the figure is robust to the on-disk version token.
# The bundle was uploaded by ``scripts_dev/visualisations/bundle_o_n_heatmaps.py``
# under the legacy paired-DPO version name and is frozen (not regenerated).
CANONICAL_VERSION = "ocean_const_paired_dpo"

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


_OPENNESS_RE = re.compile(r"openness-amplifier-[^+\-]*([+-]\d+\.\d+)")
_NEUROTICISM_RE = re.compile(r"neuroticism-amplifier-[^+\-]*([+-]\d+\.\d+)")


def _cell_scales(dirname: str) -> tuple[float, float] | None:
    """Parse ``(o_scale, n_scale)`` from a cell dir name, version-token-agnostic.

    Returns ``None`` if the dir is not part of the openness×neuroticism
    amplifier soup (a suppressor cell, or a different trait pair bundled in).
    The base cell applies no adapter and is named ``scale_+0.00`` → ``(0, 0)``.
    """
    if "suppressor" in dirname:
        return None
    other_traits = ("conscientiousness", "extraversion", "agreeableness")
    if any(t in dirname for t in other_traits):
        return None
    o = _OPENNESS_RE.search(dirname)
    n = _NEUROTICISM_RE.search(dirname)
    if o is None and n is None:
        return (0.0, 0.0) if "scale_+0.00" in dirname else None
    return (float(o.group(1)) if o else 0.0, float(n.group(1)) if n else 0.0)


def _discover_cells(subdir: str) -> dict[tuple[float, float], str]:
    """Map each grid ``(o, n)`` to the real cell dir under ``on_<trait>/``.

    Lists the bundle subtree and reads scales out of whatever cell labels exist
    (legacy or renamed paired-DPO version token), so the figure works regardless
    of the on-disk naming and never hardcodes the legacy token. Guards against
    the "someone ran more cells" cases:

      * only the openness×neuroticism amplifier soup is accepted;
      * only cells on the fixed ``SCALES`` grid are kept (extras are ignored);
      * if more than one dir maps to the same ``(o, n)``, the
        ``ocean_const_paired_dpo`` one wins; a genuine same-priority collision
        raises rather than silently plotting the wrong cell.
    """
    fs = HfFileSystem()
    root = f"datasets/{HF_REPO_ID}/{BUNDLE_PATH_IN_REPO}/{subdir}"
    names = [p.split("/")[-1] for p in fs.ls(root, detail=False)]
    candidates: dict[tuple[float, float], list[str]] = {}
    extras = 0
    for name in names:
        scales = _cell_scales(name)
        if scales is None or scales[0] not in SCALES or scales[1] not in SCALES:
            extras += 1
            continue
        candidates.setdefault(scales, []).append(name)
    chosen: dict[tuple[float, float], str] = {}
    for cell, dirs in candidates.items():
        preferred = [d for d in dirs if CANONICAL_VERSION in d] or dirs
        if len(preferred) > 1:
            raise RuntimeError(
                f"ambiguous cell {cell} in {subdir!r}: {preferred} — refusing to guess"
            )
        chosen[cell] = preferred[0]
    if extras:
        side = len(SCALES)
        print(f"  (ignored {extras} dir(s) outside the {side}x{side} O×N grid)")
    return chosen


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_grid(subdir: str, judged_trait: str) -> np.ndarray:
    """Return a 5x5 array of mean scores, shape [n_axis, o_axis].

    Rows (y axis) indexed by SCALES (n_plus), columns (x axis) by SCALES
    (o_plus). NaN for cells that aren't on HF yet.
    """
    grid = np.full((len(SCALES), len(SCALES)), np.nan, dtype=float)
    cells = _discover_cells(subdir)
    metric_name = f"{judged_trait}_v2"
    for yi, n_scale in enumerate(SCALES):
        for xi, o_scale in enumerate(SCALES):
            dirname = cells.get((o_scale, n_scale))
            if dirname is None:
                print(f"  ⚠ (o={o_scale:+.0f}, n={n_scale:+.0f}): missing on HF")
                continue
            hf_path = (
                f"{BUNDLE_PATH_IN_REPO}/{subdir}/{dirname}"
                f"/judge_runs/{RATER_ID}/{metric_name}.jsonl"
            )
            local = hydrate_judge_file(hf_path, CACHE_DIR, repo_id=HF_REPO_ID)
            if local is None:
                print(f"  ⚠ (o={o_scale:+.0f}, n={n_scale:+.0f}): judge file missing")
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
    title_suffix: str = "",
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
    ax.set_title(f"{trait_title} judge score{title_suffix}", loc="left", pad=8)

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
            ax.text(xi, yi, label, ha="center", va="center", fontsize=13, color=color)
            if xi == base_xi and yi == base_yi and not np.isnan(val):
                ax.text(
                    xi,
                    yi - 0.32,
                    "(base)",
                    ha="center",
                    va="center",
                    fontsize=11,
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

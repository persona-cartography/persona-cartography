"""Appendix Pareto plot for the E+ induction comparison.

For each (method, intervention strength) pair we measured on neutral psychometric
prompts, plot one point in (extraversion, coherence) space. This visualises the
trait/coherence trade-off curve that each method traces as its strength grows.

Methods covered:
  - LoRA E+ at scales {0.25, 0.5, 0.75, 1.0}
  - Actcap E+ at fractions {0.25, 0.5, 0.75, 0.85, 1.0}
  - Sysprompt-induce E+ (single point — no scale knob)
  - User-roleplay E+ scenarios (single point)
  - Base (single point — the origin)

Each method has its own colour; the points within a method are connected by a
faint line in scale order so the trajectory through (ext, coh) space is visible.
A dashed reference at coh=base marks the "no coherence cost" line.

Data sources (all under HF ``persona-shattering-lasr/monorepo``):
  base:        rollout_baseline_t0.7_steering/base/baseline/run_info.json
  LoRA sweep:  rollout_sweep_lora_t0.7_steering/scale_+0.{25,50,75},+1.00/baseline/run_info.json
  actcap:      rollout_sweep_activation_capping_t0.7_steering/frac_0.{25,50,75,85},1.00/baseline/run_info.json
  sysprompt:   rollout_sysprompt_elicit_t0.7_steering/base/sysprompt_elicit_extraversion_high/run_info.json
  scenarios:   rollout_scenarios/subset_3e141037_t0.7_steering/high/base/scenarios_extraversion_high/run_info.json

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_pareto_eplus.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_pareto
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/induction/fig_G_induction_pareto_eplus.pdf",
]

HF_REPO_FS = "datasets/persona-shattering-lasr/monorepo"
_AMP = (
    f"{HF_REPO_FS}/fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/rollouts"
)


# Each method: (label, colour, marker, list of (variant_label, run_info_path))
# The variant_label appears next to the marker. Path is the run_info.json
# whose aggregates carry overall extraversion_v2 and coherence_v2 means.
METHODS: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "Base",
        "#000000",
        "o",
        [
            ("", f"{_AMP}/rollout_baseline_t0.7_steering/base/baseline/run_info.json"),
        ],
    ),
    (
        "Sysprompt-induce E↑",
        "#0f7f3f",
        "s",
        [
            (
                "",
                f"{_AMP}/rollout_sysprompt_elicit_t0.7_steering/base/sysprompt_elicit_extraversion_high/run_info.json",
            ),
        ],
    ),
    (
        "E↑ LoRA",
        "#c91546",
        "^",
        [
            (f"coeff={s}", f"{_AMP}/rollout_sweep_lora_t0.7_steering/scale_+{s}/baseline/run_info.json")
            for s in ["0.25", "0.50", "0.75", "1.00"]
        ],
    ),
    (
        "E↑ activation capping",
        "#3c7fb1",
        "D",
        [
            (
                f"coeff={f}",
                f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_{f}/baseline/run_info.json",
            )
            for f in ["0.25", "0.50", "0.75", "0.85", "1.00"]
        ],
    ),
]


def _load_aggregates(path: str) -> tuple[float, float]:
    """Return (ext_mean, coh_mean) from a run_info.json on HF."""
    fs = HfFileSystem()
    d = json.loads(fs.cat(path).decode())
    a = d["aggregates"]
    ext = float(a["overall/extraversion_v2.score/mean"])
    coh = float(a["overall/coherence_v2.score/mean"])
    return ext, coh


def main() -> None:
    print("Loading method points from HF...")
    method_points: list[tuple[str, str, str, list[tuple[str, float, float]]]] = []
    base_coh: float | None = None
    for label, colour, marker, variants in METHODS:
        points: list[tuple[str, float, float]] = []
        for v_label, path in variants:
            try:
                ext, coh = _load_aggregates(path)
                points.append((v_label, ext, coh))
                print(f"  {label} [{v_label}]: ext={ext:+.2f} coh={coh:.2f}")
                if label == "Base":
                    base_coh = coh
            except Exception as e:
                print(f"  {label} [{v_label}]: ERR {e.__class__.__name__}: {e}")
        method_points.append((label, colour, marker, points))

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # base reference line at base coherence
    if base_coh is not None:
        ax.axhline(
            base_coh,
            color="grey",
            linewidth=0.8,
            linestyle=":",
            alpha=0.6,
            zorder=0,
        )
        ax.text(
            -1.95, base_coh + 0.06, "base coherence",
            fontsize=8, color="grey", style="italic", va="bottom",
        )

    # Per-point label-offset hints to avoid overlap. Keys: (method_label, variant_label).
    # Default to (8, 4) when not specified.
    OFFSET_OVERRIDES: dict[tuple[str, str], tuple[int, int]] = {
        # E↑ LoRA cluster — push labels away from the origin pile-up
        ("E↑ LoRA", "coeff=0.25"): (8, -14),
        ("E↑ LoRA", "coeff=0.50"): (-65, 4),
        ("E↑ LoRA", "coeff=0.75"): (-12, 12),
        ("E↑ LoRA", "coeff=1.00"): (10, 0),
        # actcap cluster — push labels below points so they don't collide with
        # LoRA labels above (esp. near the (1.1, 8.4) crossover) and stay clear
        # of the "base coherence" text on the left edge
        ("E↑ activation capping", "coeff=0.25"): (10, 8),
        ("E↑ activation capping", "coeff=0.50"): (10, 8),
        ("E↑ activation capping", "coeff=0.75"): (10, -14),
        ("E↑ activation capping", "coeff=0.85"): (10, -14),
        ("E↑ activation capping", "coeff=1.00"): (10, -4),
    }

    for label, colour, marker, points in method_points:
        if not points:
            continue
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        # connect points within method order so the curve through space is visible
        if len(points) > 1:
            ax.plot(
                xs, ys,
                color=colour, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1,
            )
        # Single-point methods (sysprompt) get a hollow marker so they don't
        # occlude an overlapping LoRA/actcap point at the same coordinate.
        if len(points) == 1 and label != "Base":
            ax.scatter(
                xs, ys,
                facecolors="none", edgecolors=colour, marker=marker,
                s=110, linewidths=1.6,
                label=label, zorder=3,
            )
        else:
            ax.scatter(
                xs, ys,
                color=colour, marker=marker, s=80,
                edgecolors="#2f3748", linewidths=0.5,
                label=label, zorder=2,
            )
        # annotate variant labels at each point — only if non-empty.
        # Single-point methods (base, sysprompt, user-roleplay) have empty labels
        # because their identity is already in the legend.
        for v_label, x, y in points:
            if not v_label:
                continue
            offset = OFFSET_OVERRIDES.get((label, v_label), (8, 4))
            ax.annotate(
                v_label,
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=colour,
                alpha=0.9,
            )

    ax.set_xlabel("Extraversion judge score (mean across all turns)", fontsize=11)
    ax.set_ylabel("Coherence judge score (mean across all turns)", fontsize=11)
    ax.set_xlim(-2, 4)
    ax.set_ylim(0, 10)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.4, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / "fig_G_induction_pareto_eplus.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

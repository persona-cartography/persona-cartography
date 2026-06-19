"""Appendix figure: O± Pareto plot.

Mirror of fig_G_induction_pareto_eplus.pdf for openness, with both directions
on a single plot. Each marker is one (method, intervention strength) cell.
Within-method points connected with faint lines.

Methods covered (both directions):
  - Base on neutral (single point)
  - LoRA E+ at coefficients {0.25, 0.50, 0.75, 1.00}
  - Actcap E+ at coefficients {0.25, 0.50, 0.75, 1.00}
  - Sysprompt-induce E↑ / E↓ (one point each)

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_o_pareto.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_o_pareto
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/induction/fig_G_induction_o_pareto.pdf",
]

HF_REPO_FS = "datasets/persona-shattering-lasr/monorepo"
_AMP = (
    f"{HF_REPO_FS}/fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4_paired_dpo/rollouts"
)
_SUPP = (
    f"{HF_REPO_FS}/fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4_paired_dpo/rollouts"
)


# Each method: (label, colour, marker, list of (variant_label, run_info_path))
METHODS: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "Base",
        "#000000",
        "o",
        [("", f"{_AMP}/rollout_baseline_t0.7_steering_o/base/baseline/run_info.json")],
    ),
    (
        "Sysprompt-induce O↑",
        "#0f7f3f",
        "s",
        [("", f"{_AMP}/rollout_sysprompt_elicit_t0.7_steering_o/base/sysprompt_elicit_openness_high/run_info.json")],
    ),
    (
        "Sysprompt-induce O↓",
        "#5b2abf",
        "P",
        [("", f"{_SUPP}/rollout_sysprompt_elicit_t0.7_steering_o/base/sysprompt_elicit_openness_low/run_info.json")],
    ),
    (
        "O↑ LoRA",
        "#c91546",
        "^",
        [(f"coeff={s}", f"{_AMP}/rollout_sweep_lora_t0.7_steering_o/scale_+{s}/baseline/run_info.json")
         for s in ["0.25", "0.50", "0.75", "1.00"]],
    ),
    (
        "O↑ activation capping",
        "#3c7fb1",
        "D",
        [(f"coeff={f}", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering_o/frac_{f}/baseline/run_info.json")
         for f in ["0.25", "0.50", "0.75", "1.00"]],
    ),
    (
        # Drop coeff=3.00 — coh has collapsed to 1.33 and the point pulls
        # the y-axis down without adding a useful claim. The story (LoRA
        # breaks the floor at 1.5/2.0) is fully visible without it.
        "O↓ LoRA",
        "#df6f4f",
        "v",
        [(f"coeff={s}", f"{_SUPP}/rollout_sweep_lora_t0.7_steering_o/scale_+{s}/baseline/run_info.json")
         for s in ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00"]],
    ),
    (
        # Drop coeff={2.00, 3.00} — coh near zero, trait retreats.
        "O↓ activation capping",
        "#f39a22",
        "X",
        [(f"coeff={f}", f"{_SUPP}/rollout_sweep_activation_capping_t0.7_steering_o/frac_{f}/baseline/run_info.json")
         for f in ["0.25", "0.50", "0.75", "1.00", "1.50"]],
    ),
]


def _load_aggregates(path: str) -> tuple[float, float]:
    fs = HfFileSystem()
    d = json.loads(fs.cat(path).decode())
    a = d["aggregates"]
    op = float(a["overall/openness_v2.score/mean"])
    coh = float(a["overall/coherence_v2.score/mean"])
    return op, coh


def main() -> None:
    print("Loading method points from HF...")
    method_points: list[tuple[str, str, str, list[tuple[str, float, float]]]] = []
    base_coh: float | None = None
    for label, colour, marker, variants in METHODS:
        points: list[tuple[str, float, float]] = []
        for v_label, path in variants:
            try:
                op, coh = _load_aggregates(path)
                points.append((v_label, op, coh))
                print(f"  {label} [{v_label or '(single)'}]: op={op:+.2f} coh={coh:.2f}")
                if label == "Base":
                    base_coh = coh
            except Exception as e:
                print(f"  {label} [{v_label}]: ERR {e.__class__.__name__}")
        method_points.append((label, colour, marker, points))

    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    if base_coh is not None:
        ax.axhline(base_coh, color="grey", linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=0)
        ax.text(-3.95, base_coh + 0.06, "base coherence",
                fontsize=8, color="grey", style="italic", va="bottom")

    # Per-point label-offset hints to avoid overlap.
    # The right side (O↑ high coeffs) is dense — labels for actcap and LoRA at
    # high coeffs cluster around (3.6-4.0, 7.5-8.5). Fan them out.
    OFFSET_OVERRIDES: dict[tuple[str, str], tuple[int, int]] = {
        # O↑ LoRA — points at op {1.28, 2.12, 3.61, 3.97} (low coh varies)
        # Wait, those are actcap. LoRA is at {2.23, 3.22, 3.82, 3.97}.
        ("O↑ LoRA", "coeff=0.25"): (-50, 8),    # left of point
        ("O↑ LoRA", "coeff=0.50"): (-50, 8),
        ("O↑ LoRA", "coeff=0.75"): (8, 10),     # above-right
        ("O↑ LoRA", "coeff=1.00"): (8, 10),     # above-right (overlaps actcap 1.0; offset further)
        ("O↑ activation capping", "coeff=0.25"): (8, -14),    # below-right
        ("O↑ activation capping", "coeff=0.50"): (8, -14),
        ("O↑ activation capping", "coeff=0.75"): (8, -14),
        ("O↑ activation capping", "coeff=1.00"): (8, -22),    # well below to avoid overlap with LoRA 1.0
        # O↓ LoRA — points at op {0.79, 0.21, -0.18, -0.36}
        ("O↓ LoRA", "coeff=0.25"): (-50, 10),
        ("O↓ LoRA", "coeff=0.50"): (-50, 10),
        ("O↓ LoRA", "coeff=0.75"): (8, 10),
        ("O↓ LoRA", "coeff=1.00"): (8, 10),
        # O↓ actcap — points at op {0.77, 0.53, 0.12, -0.12} (coh drops to 7.18 at 1.0)
        ("O↓ activation capping", "coeff=0.25"): (8, -14),
        ("O↓ activation capping", "coeff=0.50"): (8, -14),
        ("O↓ activation capping", "coeff=0.75"): (-58, -14),
        ("O↓ activation capping", "coeff=1.00"): (8, -14),
    }

    # Draw sweep methods first, then single-point markers (sysprompt/base) on top
    # so they're not hidden under the LoRA/actcap clusters. Sysprompt points get
    # hollow markers so they don't occlude an overlapping LoRA/actcap point at
    # the same coordinate (consistent with the E↑ Pareto convention).
    SINGLE_POINT_LABELS = {"Base", "Sysprompt-induce O↑", "Sysprompt-induce O↓"}
    for label, colour, marker, points in method_points:
        if not points or label in SINGLE_POINT_LABELS:
            continue
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        if len(points) > 1:
            ax.plot(xs, ys, color=colour, linestyle="--", linewidth=1.2,
                    alpha=0.5, zorder=1)
        ax.scatter(xs, ys, color=colour, marker=marker, s=80,
                   edgecolors="#2f3748", linewidths=0.5, label=label, zorder=2)
        for v_label, x, y in points:
            if not v_label:
                continue
            offset = OFFSET_OVERRIDES.get((label, v_label), (8, 4))
            ax.annotate(v_label, xy=(x, y), xytext=offset,
                        textcoords="offset points", fontsize=7.5,
                        color=colour, alpha=0.9)
    # Singletons on top — base solid (origin marker), sysprompt hollow.
    for label, colour, marker, points in method_points:
        if label not in SINGLE_POINT_LABELS or not points:
            continue
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        if label == "Base":
            ax.scatter(xs, ys, color=colour, marker=marker, s=110,
                       edgecolors="#2f3748", linewidths=0.8,
                       label=label, zorder=4)
        else:
            ax.scatter(xs, ys, facecolors="none", edgecolors=colour, marker=marker,
                       s=110, linewidths=1.6, label=label, zorder=3)

    ax.set_xlabel("Openness judge score (mean across all turns)", fontsize=11)
    ax.set_ylabel("Coherence judge score (mean across all turns)", fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 10)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.4, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)

    fig.tight_layout()
    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / "fig_G_induction_o_pareto.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

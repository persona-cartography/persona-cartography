"""Appendix figures: per-method sweep trajectories for E↑ induction.

Two figures, both per-turn extraversion+coherence trajectories like the main-body
fig_3_4_eplus_induction_comparison.pdf, but each shows ONE method swept across
its scale axis (so the reader can see how the trait/coherence trade-off
develops monotonically with intervention strength):

  1. LoRA E↑ scale sweep on neutral: base + scales {0.25, 0.5, 0.75, 1.0}
  2. Actcap E↑ fraction sweep on neutral: base + fractions {0.25, 0.5, 0.75, 0.85, 1.0}

These are the per-method counterparts to the main-body figure (which shows the
chosen contender from each method) — they justify the contender pick (LoRA 0.75,
actcap 0.85) by displaying the full sweep.

Layout matches the main-body E↑ headline figure: extraversion (left) and
coherence (right) panels side by side, single shared legend below, no title.

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_lora_sweep.pdf
    - paper/figures/appendix/induction/fig_G_induction_actcap_sweep.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_method_sweeps
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/induction/fig_G_induction_lora_sweep.pdf",
    "appendix/induction/fig_G_induction_actcap_sweep.pdf",
]

HF_REPO_FS = "datasets/persona-shattering-lasr/monorepo"
_AMP = (
    f"{HF_REPO_FS}/fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/rollouts"
)

BASE_PATH = f"{_AMP}/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl"


def _ramp(cmap_name: str, n: int, lo: float = 0.30, hi: float = 0.92) -> list[str]:
    """Return n hex colours sampled from a sequential colormap, light → dark."""
    cmap = cm.get_cmap(cmap_name)
    out: list[str] = []
    for i in range(n):
        frac = lo + (hi - lo) * (i / max(n - 1, 1))
        rgba = cmap(frac)
        out.append("#{:02x}{:02x}{:02x}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
    return out


def _build_cells(
    method_paths: list[tuple[str, str]],  # (label, eval_jsonl_path)
    cmap_name: str,
) -> list[tuple[str, str, str, str, str]]:
    """Add Base + sequential colours per coefficient. Returns (label, path, colour, linestyle, marker) tuples."""
    colours = _ramp(cmap_name, len(method_paths))
    cells: list[tuple[str, str, str, str, str]] = [("Base", BASE_PATH, "#000000", "-", "o")]
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2, 1, 2))]
    markers = ["s", "^", "D", "v", "P"]
    for i, (label, path) in enumerate(method_paths):
        cells.append((label, path, colours[i], linestyles[i % len(linestyles)], markers[i % len(markers)]))
    return cells


# Sweep configurations.
# LoRA: red ramp (matches the LoRA colour in the headline figure).
# Actcap: blue ramp (matches the actcap colour in the headline figure).
LORA_CELLS = _build_cells(
    [
        ("coeff=0.25", f"{_AMP}/rollout_sweep_lora_t0.7_steering/scale_+0.25/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=0.50", f"{_AMP}/rollout_sweep_lora_t0.7_steering/scale_+0.50/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=0.75", f"{_AMP}/rollout_sweep_lora_t0.7_steering/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=1.00", f"{_AMP}/rollout_sweep_lora_t0.7_steering/scale_+1.00/baseline/evals/rollouts_evaluated.jsonl"),
    ],
    "Reds",
)

ACTCAP_CELLS = _build_cells(
    [
        ("coeff=0.25", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_0.25/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=0.50", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_0.50/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=0.75", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_0.75/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=0.85", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_0.85/baseline/evals/rollouts_evaluated.jsonl"),
        ("coeff=1.00", f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering/frac_1.00/baseline/evals/rollouts_evaluated.jsonl"),
    ],
    "Blues",
)


JUDGES: list[tuple[str, str, tuple[float, float]]] = [
    ("extraversion_v2", "Extraversion score", (-4, 4)),
    ("coherence_v2", "Coherence score", (0, 10)),
]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load_evaluated(hf_path: str) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    text = fs.cat(hf_path).decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _per_turn_scores(
    entries: list[dict[str, Any]], judge: str
) -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for entry in entries:
        for _r_idx, msgs in entry.get("messages", {}).items():
            for msg in msgs:
                if msg.get("role") != "assistant":
                    continue
                turn = msg.get("turn_index")
                if turn is None:
                    continue
                scores = msg.get("scores") or {}
                obj = scores.get(judge)
                raw = obj.get("score") if isinstance(obj, dict) else obj
                if raw is None:
                    continue
                try:
                    by_turn[int(turn)].append(float(raw))
                except (TypeError, ValueError):
                    continue
    return by_turn


def _bootstrap_ci(values: list[float], n_iter: int, seed: int) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = mean(values)
    if n == 1:
        return m, m, m
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    return m, boots[int(0.025 * n_iter)], boots[int(0.975 * n_iter)]


def _aggregate(by_turn: dict[int, list[float]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for turn, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap_ci(vs, BOOTSTRAP_N, BOOTSTRAP_SEED + turn)
        out[turn] = {"mean": m, "lo": lo, "hi": hi, "n": len(vs)}
    return out


def _render(cells: list[tuple[str, str, str, str, str]], out_name: str) -> None:
    print(f"\n=== {out_name} ===")
    cell_data = []
    for label, path, colour, linestyle, marker in cells:
        print(f"  {label}: {path.rsplit('/rollouts/', 1)[-1]}")
        entries = _load_evaluated(path)
        cell_data.append((label, entries, colour, linestyle, marker))

    n_judges = len(JUDGES)
    fig, axes = plt.subplots(1, n_judges, figsize=(7.5 * n_judges, 4.0), sharex=True)
    if n_judges == 1:
        axes = [axes]

    for ax, (judge, ylabel, ylim) in zip(axes, JUDGES):
        for label, entries, colour, linestyle, marker in cell_data:
            agg = _aggregate(_per_turn_scores(entries, judge))
            if not agg:
                continue
            turns = sorted(agg.keys())
            means = [agg[t]["mean"] for t in turns]
            lo = [agg[t]["lo"] for t in turns]
            hi = [agg[t]["hi"] for t in turns]
            yerr_lo = [m - l for m, l in zip(means, lo)]
            yerr_hi = [h - m for m, h in zip(means, hi)]
            ax.errorbar(
                turns, means,
                yerr=[yerr_lo, yerr_hi],
                color=colour, linestyle=linestyle, marker=marker,
                linewidth=2, markersize=6, label=label,
                capsize=4, capthick=1.2, elinewidth=1.2,
            )
        if ylim is not None and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Turn index", fontsize=11)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles), fontsize=9, frameon=True,
    )

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / out_name
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def main() -> None:
    _render(LORA_CELLS, "fig_G_induction_lora_sweep.pdf")
    _render(ACTCAP_CELLS, "fig_G_induction_actcap_sweep.pdf")


if __name__ == "__main__":
    main()

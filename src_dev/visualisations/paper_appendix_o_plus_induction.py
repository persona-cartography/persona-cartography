"""Appendix figure: headline-style O↑ induction comparison.

Mirror of fig_3_4_eplus_induction_comparison.pdf for the openness amplifier
direction. Demonstrates the cross-trait generality of the steering matrix.
4 lines: base, sysprompt-induce-O↑, O↑ LoRA at coeff=0.75, O↑ actcap at
coeff=0.75 (chosen to mirror the E↑ contender for direct comparability).

Data sources (HF persona-shattering-lasr/monorepo, all under
fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4_paired_dpo/rollouts/):
  base:        rollout_baseline_t0.7_steering_o/base/baseline/evals/rollouts_evaluated.jsonl
  sysprompt:   rollout_sysprompt_elicit_t0.7_steering_o/base/sysprompt_elicit_openness_high/evals/rollouts_evaluated.jsonl
  LoRA 0.75:   rollout_sweep_lora_t0.7_steering_o/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl
  actcap 0.75: rollout_sweep_activation_capping_t0.7_steering_o/frac_0.75/baseline/evals/rollouts_evaluated.jsonl

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_oplus.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_o_plus_induction
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

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/induction/fig_G_induction_oplus.pdf",
]

HF_REPO_FS = "datasets/persona-shattering-lasr/monorepo"
_AMP = (
    f"{HF_REPO_FS}/fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4_paired_dpo/rollouts"
)

CELLS: list[tuple[str, str, str, str, str]] = [
    (
        "Base (no intervention)",
        f"{_AMP}/rollout_baseline_t0.7_steering_o/base/baseline/evals/rollouts_evaluated.jsonl",
        "#000000", "-", "o",
    ),
    (
        "Sysprompt-induce O↑",
        f"{_AMP}/rollout_sysprompt_elicit_t0.7_steering_o/base/sysprompt_elicit_openness_high/evals/rollouts_evaluated.jsonl",
        "#0f7f3f", "--", "s",
    ),
    (
        "O↑ LoRA (coeff=1.00)",
        f"{_AMP}/rollout_sweep_lora_t0.7_steering_o/scale_+1.00/baseline/evals/rollouts_evaluated.jsonl",
        "#c91546", "-.", "^",
    ),
    (
        "O↑ activation capping (coeff=0.75)",
        f"{_AMP}/rollout_sweep_activation_capping_t0.7_steering_o/frac_0.75/baseline/evals/rollouts_evaluated.jsonl",
        "#3c7fb1", ":", "D",
    ),
]

JUDGES: list[tuple[str, str, tuple[float, float]]] = [
    ("openness_v2", "Openness score", (-4, 4)),
    ("coherence_v2", "Coherence score", (0, 10)),
]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load(path: str) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    text = fs.cat(path).decode()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _per_turn(entries: list[dict[str, Any]], judge: str) -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        for r, msgs in e.get("messages", {}).items():
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                t = m.get("turn_index")
                obj = (m.get("scores") or {}).get(judge, {})
                v = obj.get("score") if isinstance(obj, dict) else obj
                if t is not None and v is not None:
                    by_turn[int(t)].append(float(v))
    return by_turn


def _bootstrap(values: list[float], seed: int, n_iter: int = BOOTSTRAP_N) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = mean(values)
    if n == 1:
        return m, m, m
    rng = random.Random(seed)
    boots = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_iter))
    return m, boots[int(0.025 * n_iter)], boots[int(0.975 * n_iter)]


def _aggregate(by_turn: dict[int, list[float]]) -> dict[int, dict[str, float]]:
    out = {}
    for t, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap(vs, BOOTSTRAP_SEED + t)
        out[t] = {"mean": m, "lo": lo, "hi": hi}
    return out


def main() -> None:
    print("Loading O↑ cells from HF...")
    cell_data = []
    for label, path, colour, linestyle, marker in CELLS:
        print(f"  {label}")
        entries = _load(path)
        cell_data.append((label, entries, colour, linestyle, marker))

    n_judges = len(JUDGES)
    fig, axes = plt.subplots(1, n_judges, figsize=(7.5 * n_judges, 4.0), sharex=True)
    if n_judges == 1:
        axes = [axes]

    for ax, (judge, ylabel, ylim) in zip(axes, JUDGES):
        for label, entries, colour, linestyle, marker in cell_data:
            agg = _aggregate(_per_turn(entries, judge))
            if not agg:
                continue
            ts = sorted(agg.keys())
            ms = [agg[t]["mean"] for t in ts]
            lo = [agg[t]["lo"] for t in ts]
            hi = [agg[t]["hi"] for t in ts]
            yerr_lo = [m - l for m, l in zip(ms, lo)]
            yerr_hi = [h - m for m, h in zip(ms, hi)]
            ax.errorbar(
                ts, ms, yerr=[yerr_lo, yerr_hi],
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

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / "fig_G_induction_oplus.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

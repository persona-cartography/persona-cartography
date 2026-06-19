"""Drift prevention figure for the appendix subsection.

Per-turn extraversion + coherence trajectory under user-side scenario pressure
(E↓ pressure scenarios), with three contender lines that demonstrate "the same
LoRA at the same coefficient produces a curved equilibrium when applied
against opposing user pressure":

  1. Neutral baseline (no pressure, no intervention) — flat reference at the
     model's natural register on neutral psychometric prompts.
  2. Base under E↓ pressure — drift toward introversion, equilibrium at ~-2.5.
  3. E↑ LoRA at coefficient 0.5 under pressure — partial drift prevention.
  4. E↑ activation capping at coefficient 0.75 under pressure — partial
     drift prevention.

Same trajectories as the steering-on-neutral headline, but with the user-side
pressure applied via scenarios from datasets/scenarios/extraversion_pressure_v1.json.

Data sources (HF persona-shattering-lasr/monorepo):
  Neutral baseline: amplifier/.../rollout_baseline_t0.7_steering/base/baseline/
  Base on E↓ scenarios: filtered subset of the 4 v1 winners via the existing
    rollout_scenarios pipeline (subset_bbf9d326 contains the 4 winners)
  E↑ LoRA scale 0.50 on E↓ scenarios: subset_bbf9d326 + subset_5fcc3ba1 merged
  E↑ actcap frac 0.75 on E↓ scenarios: subset_b2595342

These are the data we already had from the prevention experiments before the
steering pivot — same paths.

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_drift_prevention.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_drift_prevention
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
    "appendix/induction/fig_G_induction_drift_prevention.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean/extraversion"
_AMP = f"{HF_FS}/amplifier/vanton4_paired_dpo/rollouts"

# Cells: each line is (label, list[hf_path], colour, linestyle, marker).
# Multi-path lists are merged at load time when the same logical cell was
# split across multiple subset hashes (the prevention experiments were run on
# v1-winners and v2-additional in two batches).
# 4 v1-winner scenario IDs to filter the original 5-scenario sanity base
# down to the same 4 scenarios the LoRA cells were run on.
V1_WINNER_SEED_IDS = {
    "sample_095a3d2405d0b789be223c6e",  # solo cabin
    "sample_0a9830553572427c9d55520d",  # astronomy
    "sample_34cc05cff062e7b63a3ac9f4",  # walking-clearing
    "sample_7c7995e989150d6f2718c4be",  # tides line / rainy reading
}

CELLS: list[tuple[str, list[str], str, str, str]] = [
    (
        "Neutral baseline (no pressure)",
        [f"{_AMP}/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl"],
        "#000000", "-", "o",
    ),
    (
        "Base under E↓ pressure (drift)",
        [
            # v1-winners base lives in the original sanity-sweep path; we filter
            # it to 4 winners at load time (see _load_paths). Combined with the
            # 5-scenario v2 base for the full 9-scenario base under pressure.
            f"{_AMP}/rollout_scenarios/low/base/scenarios_extraversion_low/evals/rollouts_evaluated.jsonl",
            f"{HF_FS}/suppressor/vanton4_paired_dpo/rollouts/rollout_scenarios/subset_5fcc3ba1/low/base/scenarios_extraversion_low/evals/rollouts_evaluated.jsonl",
        ],
        "#7f8c9b", "--", "v",
    ),
    (
        "E↑ LoRA (coeff=0.5) under pressure",
        [
            f"{_AMP}/rollout_scenarios/subset_bbf9d326/low/scale_+0.50/scenarios_extraversion_low/evals/rollouts_evaluated.jsonl",
            f"{_AMP}/rollout_scenarios/subset_5fcc3ba1/low/scale_+0.50/scenarios_extraversion_low/evals/rollouts_evaluated.jsonl",
        ],
        "#c91546", "-.", "^",
    ),
    (
        "E↑ activation capping (coeff=0.75) under pressure",
        [
            f"{_AMP}/rollout_scenarios/subset_b2595342/low/frac_0.75/scenarios_extraversion_low/evals/rollouts_evaluated.jsonl",
        ],
        "#3c7fb1", ":", "D",
    ),
]

JUDGES: list[tuple[str, str, tuple[float, float]]] = [
    ("extraversion_v2", "Extraversion score", (-4, 4)),
    ("coherence_v2", "Coherence score", (0, 10)),
]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load_paths(paths: list[str]) -> list[dict[str, Any]]:
    """Load and concatenate evaluated rollouts from one or more HF paths.

    The original v1-winners base run was generated on 5 scenarios. We filter
    it to the 4 winners (V1_WINNER_SEED_IDS) at load time so the merged
    "Base under E↓ pressure" line aggregates exactly the same 9 scenarios as
    the corresponding LoRA/actcap cells.
    """
    fs = HfFileSystem()
    out: list[dict[str, Any]] = []
    for p in paths:
        text = fs.cat(p).decode()
        for line in text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            # Filter the 5-scenario sanity base to the 4 v1 winners only.
            if "rollouts/rollout_scenarios/low/base/" in p:
                if entry.get("seed_id") not in V1_WINNER_SEED_IDS:
                    continue
            out.append(entry)
    return out


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
    print("Loading drift-prevention cells from HF...")
    cell_data = []
    for label, paths, colour, linestyle, marker in CELLS:
        print(f"  {label}")
        try:
            entries = _load_paths(paths)
            cell_data.append((label, entries, colour, linestyle, marker))
        except Exception as e:
            print(f"    ERR: {e.__class__.__name__}: {e}")

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

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / "fig_G_induction_drift_prevention.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

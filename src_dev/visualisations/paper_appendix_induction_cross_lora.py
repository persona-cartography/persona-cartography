"""Appendix figure: cross-LoRA controls for E↑ induction (per-turn trajectories).

Single 4-panel figure, one panel per control. Each panel shows per-turn
extraversion (mean across rollouts) for several lines:

  - Base on neutral (reference, black solid)
  - Canonical E↑ LoRA at coeff=0.75 (reference, faint grey dashed)
  - The control adapter at multiple coefficients (one line each)

So a reader can read each panel as a per-turn trajectory of how the *control*
moves the trait at varying intervention strengths, with the canonical E↑ LoRA
contender (the one we used in the main-body figure) as the visual anchor for
"what an actual E↑-targeting LoRA does at the chosen strength".

Panels:
  (a) E↑ LoRA without DPO step (vanton4 SFT-only)
  (b) C↓ LoRA on extraversion (cross-trait bleed; uses extraversion judge added
      via cross_judge_eval.py to the conscientiousness rollouts)
  (c) Control LoRA (no trait signal)
  (d) E↑/E↓ soup at fixed E↑=0.5 with varying E↓ component

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_cross_lora_controls.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_cross_lora
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
    "appendix/induction/fig_G_induction_cross_lora_controls.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean"

BASE_PATH = (
    f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/"
    "rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl"
)
EPLUS_REF_PATH = (
    f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/"
    "rollout_sweep_lora_t0.7_steering/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl"
)

# Coefficient sweep colours: a single neutral ramp (light → dark) per panel.
# We deliberately avoid the canonical-LoRA red so readers don't confuse a
# control sweep with the canonical E↑ contender, which appears as a faint
# grey reference line in every panel.
COEFF_COLORS = {
    "0.25": "#a07cb6",  # light purple
    "0.50": "#7a4da3",  # mid purple
    "0.75": "#c75d2e",  # burnt orange
    "1.00": "#5a3a1f",  # dark brown
}
COEFF_MARKERS = {
    "0.25": "s",
    "0.50": "^",
    "0.75": "D",
    "1.00": "v",
}
COEFF_STYLES = {
    "0.25": "--",
    "0.50": "-.",
    "0.75": ":",
    "1.00": (0, (3, 1, 1, 1)),
}

# (a) E↑ no-DPO (vanton4 parent dir, no _paired_dpo)
NO_DPO_PATHS = {
    s: f"{HF_FS}/extraversion/amplifier/vanton4/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/evals/rollouts_evaluated.jsonl"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (b) C↓ — uses cross-judge-merged eval file (extraversion judge added)
CMINUS_PATHS = {
    s: f"{HF_FS}/conscientiousness/suppressor/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/evals/rollouts_evaluated.jsonl"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (c) Control LoRA (lands at routing-quirk path)
CONTROL_PATHS = {
    s: f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo_s1vs2/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/evals/rollouts_evaluated.jsonl"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (d) Soup (E↑+E↓ at fixed E↑=0.5, varying E↓ component)
SOUP_PATHS = {
    "E↓=0.25": f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_combo_t0.7_crossLoRA/ep05_em025/baseline/evals/rollouts_evaluated.jsonl",
    "E↓=0.50": f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_combo_t0.7_crossLoRA/ep05_em05/baseline/evals/rollouts_evaluated.jsonl",
    "E↓=0.75": f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_combo_t0.7_crossLoRA/ep05_em075/baseline/evals/rollouts_evaluated.jsonl",
}
SOUP_COLOURS = {
    "E↓=0.25": "#a07cb6",
    "E↓=0.50": "#7a4da3",
    "E↓=0.75": "#c75d2e",
}
SOUP_MARKERS = {"E↓=0.25": "s", "E↓=0.50": "P", "E↓=0.75": "v"}
SOUP_STYLES = {"E↓=0.25": "--", "E↓=0.50": "-.", "E↓=0.75": ":"}

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load(path: str) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    text = fs.cat(path).decode()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _per_turn(entries: list[dict[str, Any]], judge: str = "extraversion_v2") -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        for r, msgs in e.get("messages", {}).items():
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                t = m.get("turn_index")
                s = (m.get("scores") or {}).get(judge, {})
                v = s.get("score") if isinstance(s, dict) else s
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
    out: dict[int, dict[str, float]] = {}
    for t, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap(vs, BOOTSTRAP_SEED + t)
        out[t] = {"mean": m, "lo": lo, "hi": hi}
    return out


def _plot_lines(ax: plt.Axes, lines: list[tuple[str, dict[int, dict[str, float]], str, str, Any]]) -> None:
    """Plot a set of (label, agg, color, marker, linestyle) lines on ax."""
    for label, agg, colour, marker, linestyle in lines:
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
            color=colour, marker=marker, linestyle=linestyle,
            linewidth=1.8, markersize=5, label=label,
            capsize=3, capthick=1.0, elinewidth=1.0,
        )


def _build_lines_for_sweep(paths: dict[str, str], coeff_label: str) -> list[tuple[str, dict, str, str, Any]]:
    """Return coeff-keyed lines for a single-method sweep."""
    out = []
    for coeff, p in paths.items():
        try:
            entries = _load(p)
        except Exception as e:
            print(f"  ERR loading {coeff}: {e.__class__.__name__}")
            continue
        agg = _aggregate(_per_turn(entries))
        out.append((
            f"{coeff_label}={coeff}",
            agg,
            COEFF_COLORS[coeff],
            COEFF_MARKERS[coeff],
            COEFF_STYLES[coeff],
        ))
    return out


def main() -> None:
    print("Loading reference cells...")
    base_agg = _aggregate(_per_turn(_load(BASE_PATH)))
    eplus_ref_agg = _aggregate(_per_turn(_load(EPLUS_REF_PATH)))

    REFERENCE_LINES = [
        ("Base (no intervention)", base_agg, "#000000", "o", "-"),
        ("E↑ LoRA (coeff=0.75, ref.)", eplus_ref_agg, "#7f8c9b", ".", ":"),
    ]

    print("Loading panel data...")
    nodpo_lines = _build_lines_for_sweep(NO_DPO_PATHS, "coeff")
    cminus_lines = _build_lines_for_sweep(CMINUS_PATHS, "coeff")
    ctrl_lines = _build_lines_for_sweep(CONTROL_PATHS, "coeff")

    soup_lines = []
    for label, p in SOUP_PATHS.items():
        try:
            agg = _aggregate(_per_turn(_load(p)))
            soup_lines.append((label, agg, SOUP_COLOURS[label], SOUP_MARKERS[label], SOUP_STYLES[label]))
        except Exception as e:
            print(f"  ERR loading soup {label}: {e.__class__.__name__}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)

    panels = [
        ("(a) E↑ old-pipeline LoRA", REFERENCE_LINES + nodpo_lines, axes[0][0]),
        ("(b) C↓ LoRA on E judge", REFERENCE_LINES + cminus_lines, axes[0][1]),
        ("(c) Control LoRA", REFERENCE_LINES + ctrl_lines, axes[1][0]),
        ("(d) E↑/E↓ soup (E↑ fixed at 0.5)", REFERENCE_LINES + soup_lines, axes[1][1]),
    ]

    for title, lines, ax in panels:
        _plot_lines(ax, lines)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_ylim(-4, 4)
        ax.set_title(title, loc="left", pad=8, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    for ax in axes[:, 0]:
        ax.set_ylabel("Extraversion judge score", fontsize=11)
    for ax in axes[-1, :]:
        ax.set_xlabel("Turn index", fontsize=11)

    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / "fig_G_induction_cross_lora_controls.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

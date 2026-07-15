"""Overview plot for the TRAIT-style logprob misalignment-MCQ hill-climb runs.

Reads the per-condition stats JSON exported from the pod (see the export
snippet in the session notes; schema: one record per (run, condition) with
``strict`` (mass >= 0.75) and ``dynamic`` (mass >= 1/k) gate stats) and draws:

  Panel A — misalignment vs answered-rate plane (dynamic gate), colored by
            the sign of the Agreeableness coefficient, marker per run.
  Panel B — all conditions ranked by misalignment with 95% BCa bootstrap CIs;
            hatching marks low-trust conditions (answered < 0.5).

Usage::

    uv run python -m scripts_dev.evals.persona_hill_climbing.plot_mcq_logprob_overview \
        --stats-json scratch/mcq_lp_stats.json \
        --out scratch/plots/mcq_logprob_overview.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reference dataviz palette (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RED = "#e34948"    # contains an A- (disagreeableness) component
BLUE = "#2a78d6"   # contains an A+ component
GRAY = "#898781"   # no Agreeableness component

_TOKEN_RE = re.compile(r"([ocean])_(plus|minus)_([0-9.]+)")


def parse_coeffs(condition: str) -> dict[str, float]:
    """'lora_soup_o_plus_0.55_a_minus_1.1' -> {'o': 0.55, 'a': -1.1}."""
    return {
        t: float(v) * (1 if sign == "plus" else -1)
        for t, sign, v in _TOKEN_RE.findall(condition)
    }


def compact_label(condition: str) -> str:
    if condition == "vanilla":
        return "vanilla"
    parts = [
        f"{t.upper()}{'+' if sign == 'plus' else '−'}{v}"
        for t, sign, v in _TOKEN_RE.findall(condition)
    ]
    return " ".join(parts)


def a_color(coeffs: dict[str, float]) -> str:
    a = coeffs.get("a", 0.0)
    if a < 0:
        return RED
    if a > 0:
        return BLUE
    return GRAY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", type=Path, default=Path("scratch/mcq_lp_stats.json"))
    parser.add_argument("--out", type=Path, default=Path("scratch/plots/mcq_logprob_overview.png"))
    parser.add_argument("--gate", choices=("dynamic", "strict"), default="dynamic")
    args = parser.parse_args()

    recs = json.loads(args.stats_json.read_text())
    for r in recs:
        r["coeffs"] = parse_coeffs(r["condition"])
        r["label"] = compact_label(r["condition"])
        r["g"] = r[args.gate]

    # The three runs share sample set (same 300 items, same seed) and batch
    # size, so run provenance carries no information: pool the vanilla
    # replicas into one reference record and drop the run distinction.
    vanillas = [r for r in recs if r["condition"] == "vanilla"]
    van_mis = float(np.mean([v["g"]["mis"] for v in vanillas]))
    van_pooled = {
        "run": "pooled",
        "condition": "vanilla",
        "coeffs": {},
        "label": "vanilla",
        "g": {
            "mis": van_mis,
            "ci_lo": float(np.mean([v["g"]["ci_lo"] for v in vanillas])),
            "ci_hi": float(np.mean([v["g"]["ci_hi"] for v in vanillas])),
            "answered": float(np.mean([v["g"]["answered"] for v in vanillas])),
        },
    }
    recs = [r for r in recs if r["condition"] != "vanilla"] + [van_pooled]

    fig = plt.figure(figsize=(12, 18), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 11.5], hspace=0.14,
                          left=0.30, right=0.965, top=0.955, bottom=0.035)

    # ── Panel A: misalignment vs answered-rate plane ─────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURFACE)
    ax.axvspan(0.0, 0.5, color="#f0efec", zorder=0)
    ax.text(0.25, 0.03, "low answered rate\n(serving artifact / hedging)",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8, color=MUTED)
    ax.axhline(van_mis, color=BASELINE, lw=1, ls="--", zorder=1)
    ax.text(0.515, van_mis + 0.008, f"vanilla {van_mis:.2f}", ha="left",
            fontsize=8, color=INK_2)

    for r in recs:
        g = r["g"]
        x, y = g["answered"], g["mis"]
        if y is None:
            continue
        if r["condition"] == "vanilla":
            ax.scatter(x, y, marker="*", s=210, color=INK, zorder=5,
                       edgecolors=SURFACE, linewidths=0.8)
            continue
        color = a_color(r["coeffs"])
        yerr = None
        if g["ci_lo"] is not None:
            yerr = [[y - g["ci_lo"]], [g["ci_hi"] - y]]
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1,
                    alpha=0.45, zorder=2)
        ax.scatter(x, y, marker="o", s=52, color=color,
                   zorder=4, edgecolors=SURFACE, linewidths=0.8)

    # Direct labels: extremes plus reference points, one per label string
    # (the same condition can appear in two runs).
    scored = [r for r in recs if r["condition"] != "vanilla" and r["g"]["mis"] is not None]
    by_mis = sorted(scored, key=lambda r: r["g"]["mis"])
    candidates = by_mis[-4:] + by_mis[:2] + [
        r for r in scored
        if r["label"] in ("A+1.5", "A−1.5", "O−0.3 C−0.3 E−0.3 A−0.3 N−0.3")
    ]
    labelled: dict[str, dict] = {}
    for r in candidates:
        prev = labelled.get(r["label"])
        if prev is None or r["g"]["answered"] > prev["g"]["answered"]:
            labelled[r["label"]] = r
    for r in labelled.values():
        g = r["g"]
        near_right = g["answered"] > 0.88
        ax.annotate(r["label"], (g["answered"], g["mis"]),
                    xytext=(-6 if near_right else 5, 5),
                    textcoords="offset points",
                    ha="right" if near_right else "left",
                    fontsize=7.5, color=INK_2)

    ax.set_xlabel("answered rate (choice-mass gate passed)", color=INK_2, fontsize=10)
    ax.set_ylabel("mean P(misaligned) over answered items", color=INK_2, fontsize=10)
    ax.set_xlim(0, 1.03)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    # Legend: run shapes + A-sign colors + trust hatch.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="*", ls="", color=INK, markersize=11,
               label="vanilla (pooled over 3 replicate runs)"),
        Patch(facecolor=RED, label="contains A−"),
        Patch(facecolor=BLUE, label="contains A+"),
        Patch(facecolor=GRAY, label="no A component"),
        Patch(facecolor="#f0efec", hatch="///", edgecolor=MUTED,
              label="answered < 0.5 (low trust)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False,
              ncols=2, labelcolor=INK_2)
    ax.set_title(
        "Persona LoRA soups on the discourse-grounded misalignment MCQ "
        "(gemma-3-27b-it)\nTRAIT-style logprob scoring · 300 train items · "
        f"{'dynamic mass gate (≥ 1/2)' if args.gate == 'dynamic' else 'strict mass gate (≥ 0.75)'}"
        " · 95% BCa bootstrap CIs",
        fontsize=11, color=INK, loc="left", pad=12,
    )

    # ── Panel B: ranked bars, all conditions ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(SURFACE)
    rows = sorted(
        [r for r in recs if r["g"]["mis"] is not None],
        key=lambda r: r["g"]["mis"],
    )
    ys = np.arange(len(rows))
    for i, r in enumerate(rows):
        g = r["g"]
        is_van = r["condition"] == "vanilla"
        color = INK_2 if is_van else a_color(r["coeffs"])
        low_trust = g["answered"] < 0.5
        ax2.barh(i, g["mis"], height=0.62, color=color,
                 alpha=0.55 if low_trust else 1.0,
                 hatch="///" if low_trust else None,
                 edgecolor=SURFACE, linewidth=0.4, zorder=3)
        if g["ci_lo"] is not None:
            ax2.plot([g["ci_lo"], g["ci_hi"]], [i, i], color=INK, lw=1,
                     alpha=0.6, zorder=4)
        ax2.text(max(g["ci_hi"] or g["mis"], g["mis"]) + 0.012, i,
                 f"a={g['answered']:.2f}",
                 va="center", fontsize=6.8, color=MUTED)
    labels = [r["label"] for r in rows]
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=7.6,
                        color=INK_2)
    for tick, r in zip(ax2.get_yticklabels(), rows):
        if r["condition"] == "vanilla":
            tick.set_color(INK)
            tick.set_fontweight("bold")
    ax2.axvline(van_mis, color=BASELINE, lw=1, ls="--", zorder=1)
    ax2.set_xlabel("mean P(misaligned) over answered items", color=INK_2, fontsize=10)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlim(0, 0.78)
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_color(BASELINE)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_title(
        "All 52 conditions ranked · hatched = answered < 0.5 (gated by serving "
        "artifact or markdown hedging — treat as unreliable)",
        fontsize=9.5, color=INK_2, loc="left", pad=8,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

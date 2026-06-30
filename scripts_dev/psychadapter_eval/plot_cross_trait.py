"""Cross-trait dose-response: condition each trait -5..+5, judge on ALL 5 traits.

Re-plots from the merged scored generations (v1 + sweep + sweep_extreme) — every
generation was already judged on all five OCEAN traits, so no re-judging needed.

Produces a 5-panel figure: one panel per CONDITIONED trait, each with 5 lines
(judge score on O/C/E/A/N vs latent std). The line whose colour matches the
conditioned trait is the "own-trait" curve (drawn bold); the others show
cross-trait leakage.

    uv run python scripts_dev/psychadapter_eval/plot_cross_trait.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scratch/psychadapter_eval"
SCORED = [
    OUT_DIR / "scored.jsonl",                 # v1: +/-3 + baseline
    OUT_DIR / "sweep" / "scored.jsonl",       # +/-1, +/-2 + baseline
    OUT_DIR / "sweep_extreme" / "scored.jsonl",  # +/-4, +/-5 + baseline
]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
METRICS = [f"{t}_v2" for t in TRAITS]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_rows() -> list[dict]:
    rows = []
    for p in SCORED:
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def build_curves(rows: list[dict]) -> dict:
    """curves[cond_trait][metric][std] = mean judge score. std 0 = baseline rows."""
    # baseline (shared 0 point), per metric
    base = {m: [] for m in METRICS}
    # per conditioned trait + std
    buck: dict = {t: {} for t in TRAITS}
    for r in rows:
        pm = r.get("persona_metrics") or {}
        scores = {m: pm.get(f"{m}.score") for m in METRICS}
        if r["trait"] == "baseline":
            for m in METRICS:
                if scores[m] is not None:
                    base[m].append(float(scores[m]))
            continue
        std = float(r.get("latent_value", 0.0))
        d = buck[r["trait"]].setdefault(std, {m: [] for m in METRICS})
        for m in METRICS:
            if scores[m] is not None:
                d[m].append(float(scores[m]))
    base_mean = {m: float(np.mean(v)) for m, v in base.items() if v}

    curves: dict = {}
    for t in TRAITS:
        curves[t] = {m: {0.0: base_mean[m]} for m in METRICS}
        for std, md in buck[t].items():
            for m in METRICS:
                if md[m]:
                    curves[t][m][std] = float(np.mean(md[m]))
    return curves


def plot(curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    for ci, ct in enumerate(TRAITS):
        ax = axes[ci]
        for mi, (m, mt) in enumerate(zip(METRICS, TRAITS)):
            pts = curves[ct][m]
            xs = sorted(pts)
            ys = [pts[x] for x in xs]
            own = mt == ct
            ax.plot(
                xs, ys, marker="o", color=COLORS[mi], label=mt,
                lw=2.6 if own else 1.2, zorder=3 if own else 2,
                markersize=6 if own else 4,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"conditioned: {ct}", fontsize=11)
        ax.set_xlabel("latent std")
        if ci == 0:
            ax.set_ylabel("OCEAN judge score (−4..+4)")
    axes[-1].legend(title="judged trait", fontsize=8, loc="upper left")
    fig.suptitle(
        "PsychAdapter big5: cross-trait dose-response (judge on all 5 traits; bold = own trait)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "TRAIT_cross_dose_response_5x5.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote -> {out}")

    # CSV: rows = (conditioned, judged), cols = stds
    allstd = sorted({s for t in TRAITS for m in METRICS for s in curves[t][m]})
    lines = ["conditioned,judged," + ",".join(str(s) for s in allstd)]
    for ct in TRAITS:
        for m, mt in zip(METRICS, TRAITS):
            lines.append(
                f"{ct},{mt}," + ",".join(f"{curves[ct][m].get(s, float('nan')):.3f}" for s in allstd)
            )
    (OUT_DIR / "TRAIT_cross_dose_response_5x5.csv").write_text("\n".join(lines) + "\n")


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} scored rows")
    curves = build_curves(rows)
    plot(curves)


if __name__ == "__main__":
    main()

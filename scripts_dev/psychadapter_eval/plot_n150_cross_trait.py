"""Cross-trait dose-response for n150 rollouts.

Reads scratch/psychadapter_eval/n150/generations.jsonl, scores each generation
on all 5 OCEAN traits, and produces a 5-panel cross-trait plot showing how
conditioning one trait at different intensities affects all 5 traits.

Run in the REPO env:
    uv run python scripts_dev/psychadapter_eval/plot_n150_cross_trait.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig, PersonaMetricsConfig
from src_dev.persona_metrics.run import run_persona_metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scratch/psychadapter_eval/n150/generations.jsonl"
OUT_DIR = GEN_PATH.parent

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
METRICS = [f"{t}_v2" for t in TRAITS]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_generations() -> list[dict]:
    rows = [json.loads(line) for line in GEN_PATH.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if (r.get("generation") or "").strip()]
    if not rows:
        raise SystemExit(f"No usable generations in {GEN_PATH}")
    print(f"Loaded {len(rows)} generations")
    return rows


def score(rows: list[dict]) -> Dataset:
    ds = Dataset.from_list(
        [{"response": r["generation"], "question": r["prompt"], **r} for r in rows]
    )
    config = PersonaMetricsConfig(
        evaluations=METRICS,
        response_column="response",
        question_column="question",
        judge=JudgeLLMConfig(
            max_retries=5,  # Increased from 3
            backoff_factor=2.0,
            timeout=90,  # Increased from 60
            max_concurrent=5,  # Reduced from 10 to be gentler on API
        ),
    )
    print(f"Scoring {len(ds)} rows on {len(METRICS)} traits (with enhanced retries)...")
    scored_ds, result = run_persona_metrics(config, ds)
    return scored_ds


def build_curves(scored_ds: Dataset) -> dict:
    """curves[cond_trait][metric][latent_value] = mean judge score."""
    base = {m: [] for m in METRICS}
    buck: dict = {t: {} for t in TRAITS}

    for row in scored_ds:
        pm = row.get("persona_metrics") or {}
        scores = {m: pm.get(f"{m}.score") for m in METRICS}

        if row["trait"] == "baseline":
            for m in METRICS:
                if scores[m] is not None:
                    base[m].append(float(scores[m]))
            continue

        latent_val = float(row.get("latent_value", 0.0))
        trait = row["trait"]
        d = buck[trait].setdefault(latent_val, {m: [] for m in METRICS})
        for m in METRICS:
            if scores[m] is not None:
                d[m].append(float(scores[m]))

    # Filter out -99 (failed evaluations) from baseline before averaging
    base_mean = {m: float(np.mean([x for x in v if x != -99])) if [x for x in v if x != -99] else 0.0 for m, v in base.items()}

    curves: dict = {}
    for t in TRAITS:
        curves[t] = {m: {0.0: base_mean[m]} for m in METRICS}
        for latent_val, md in buck[t].items():
            for m in METRICS:
                if md[m]:
                    curves[t][m][latent_val] = float(np.mean(md[m]))

    return curves


def plot(curves: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})")
        return

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    for ci, ct in enumerate(TRAITS):
        ax = axes[ci]
        for mi, (m, mt) in enumerate(zip(METRICS, TRAITS)):
            pts = curves[ct][m]
            xs = sorted(pts)
            # Filter out placeholder values (-99.0) when plotting
            ys = [pts[x] for x in xs if pts[x] > -99.0]
            xs_valid = [x for x in xs if pts[x] > -99.0]
            own = mt == ct
            ax.plot(
                xs_valid, ys, marker="o", color=COLORS[mi], label=mt,
                lw=2.6 if own else 1.2, zorder=3 if own else 2,
                markersize=6 if own else 4,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_ylim(-4, 4)
        ax.set_title(f"conditioned: {ct}", fontsize=11)
        ax.set_xlabel("latent value")
        if ci == 0:
            ax.set_ylabel("OCEAN judge score (−4..+4)")

    axes[-1].legend(title="judged trait", fontsize=8, loc="upper left")
    fig.suptitle(
        "Judge sweep on prefix-tuning adaptors",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "n150_TRAIT_cross_latent.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote plot -> {out}")

    # CSV: rows = (conditioned, judged), cols = latent values
    allvals = sorted({s for t in TRAITS for m in METRICS for s in curves[t][m]})
    lines = ["conditioned,judged,stat," + ",".join(str(s) for s in allvals)]
    for ct in TRAITS:
        for m, mt in zip(METRICS, TRAITS):
            vals = [curves[ct][m].get(s, float('nan')) for s in allvals]
            lines.append(f"{ct},{mt},mean," + ",".join(f"{v:.3f}" for v in vals))

    csv_out = OUT_DIR / "n150_TRAIT_cross_latent.csv"
    csv_out.write_text("\n".join(lines) + "\n")
    print(f"Wrote CSV -> {csv_out}")


def main():
    rows = load_generations()
    scored_ds = score(rows)

    # Save scored dataset
    scored_out = OUT_DIR / "n150_scored.jsonl"
    scored_out.write_text(
        "\n".join(json.dumps(r, default=str) for r in scored_ds.to_list())
    )
    print(f"Wrote scored -> {scored_out}")

    curves = build_curves(scored_ds)
    plot(curves)


if __name__ == "__main__":
    main()

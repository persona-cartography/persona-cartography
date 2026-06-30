"""Stage 2 — score PsychAdapter generations with OUR OCEAN judge.

Reads ``scratch/psychadapter_eval/generations.jsonl`` (Stage 1) and runs the
repo's OCEAN judge (``run_persona_metrics``, OpenRouter API, no GPU) over every
generation for all five traits. Produces:

  * scored.jsonl              — per-generation judge scores (-4..+4) for O/C/E/A/N
  * trait_judge_matrix.csv    — mean judged score per (conditioned trait, direction)
  * fig_psychadapter_ocean.png — heatmap + diagonal steering effect

The key result is the DIAGONAL: conditioning trait T high vs low should move the
judge's score on trait T (high - low > 0). Off-diagonal = cross-trait leakage.

Run in the REPO env (needs OPENROUTER_API_KEY in .env):

    uv run python scripts_dev/psychadapter_eval/score_ocean.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv

from src_dev.persona_metrics.config import PersonaMetricsConfig
from src_dev.persona_metrics.run import run_persona_metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

load_dotenv()

import os

ROOT = Path(__file__).resolve().parents[2]
# Honor PA_OUT (the Stage 1 generations path) so a sweep run scores/writes into
# its own dir instead of clobbering the endpoint (v1) outputs.
GEN_PATH = Path(os.environ.get("PA_OUT", str(ROOT / "scratch/psychadapter_eval/generations.jsonl")))
OUT_DIR = GEN_PATH.parent

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
METRIC_NAMES = [f"{t}_v2" for t in TRAITS]  # registered judge keys


def load_generations() -> list[dict]:
    rows = [json.loads(line) for line in GEN_PATH.read_text().splitlines() if line.strip()]
    # Drop empty / whitespace-only generations (judge can't score them).
    rows = [r for r in rows if (r.get("generation") or "").strip()]
    if not rows:
        raise SystemExit(f"No usable generations in {GEN_PATH}")
    return rows


def score(rows: list[dict]) -> Dataset:
    ds = Dataset.from_list(
        [{"response": r["generation"], "question": r["prompt"], **r} for r in rows]
    )
    config = PersonaMetricsConfig(
        evaluations=METRIC_NAMES,
        response_column="response",
        question_column="question",
    )
    scored_ds, result = run_persona_metrics(config, ds)
    print("Aggregates:", json.dumps(result.aggregates, indent=2, default=str)[:1000])
    return scored_ds


def summarize(scored_ds: Dataset) -> dict:
    """mean[conditioned_trait][direction][judged_metric] over generations."""
    bucket: dict = {}
    for row in scored_ds:
        key = (row["trait"], row["direction"])
        pm = row.get("persona_metrics") or {}
        for m in METRIC_NAMES:
            v = pm.get(f"{m}.score")
            if v is None:
                continue
            bucket.setdefault(key, {}).setdefault(m, []).append(float(v))
    return {k: {m: float(np.mean(vs)) for m, vs in d.items()} for k, d in bucket.items()}


def dose_response(scored_ds: Dataset) -> dict:
    """Per trait: {std_position: mean own-trait judge score}.

    Baseline rows (latent all-zero) supply the shared 0-std point for every trait.
    Returns {} when there is no sweep (only endpoint/baseline data).
    """
    trait_to_metric = dict(zip(TRAITS, METRIC_NAMES))
    by_trait: dict = {t: {} for t in TRAITS}
    baseline: dict = {}  # metric -> [scores] at std 0
    for row in scored_ds:
        pm = row.get("persona_metrics") or {}
        pos = float(row.get("latent_value", 0.0))
        if row["trait"] == "baseline":
            for m in METRIC_NAMES:
                v = pm.get(f"{m}.score")
                if v is not None:
                    baseline.setdefault(m, []).append(float(v))
            continue
        m = trait_to_metric[row["trait"]]
        v = pm.get(f"{m}.score")
        if v is not None:
            by_trait[row["trait"]].setdefault(pos, []).append(float(v))

    base_mean = {m: float(np.mean(vs)) for m, vs in baseline.items()}
    curves: dict = {}
    for t in TRAITS:
        pts = {pos: float(np.mean(vs)) for pos, vs in by_trait[t].items()}
        if base_mean:
            pts[0.0] = base_mean[trait_to_metric[t]]
        curves[t] = dict(sorted(pts.items()))
    # Only meaningful as a sweep if some trait has >2 distinct positions.
    if max((len(v) for v in curves.values()), default=0) <= 2:
        return {}
    return curves


def _plot_dose_response(curves: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip dose-response plot: {e})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in TRAITS:
        xs = sorted(curves[t])
        ys = [curves[t][x] for x in xs]
        ax.plot(xs, ys, marker="o", label=t)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("latent conditioning (std units)")
    ax.set_ylabel("OCEAN judge score on the conditioned trait")
    ax.set_title("PsychAdapter big5: trait dose–response (own-trait judge score)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "fig_psychadapter_dose_response.png"
    fig.savefig(out, dpi=150)
    csv = ["trait," + ",".join(str(x) for x in sorted({p for c in curves.values() for p in c}))]
    allpos = sorted({p for c in curves.values() for p in c})
    for t in TRAITS:
        csv.append(t + "," + ",".join(f"{curves[t].get(p, float('nan')):.3f}" for p in allpos))
    (OUT_DIR / "dose_response.csv").write_text("\n".join(csv) + "\n")
    print(f"Wrote dose-response -> {out}")
    print("\n=== Dose-response (own-trait judge score vs std) ===")
    print("trait            " + "".join(f"{p:>8g}" for p in allpos))
    for t in TRAITS:
        print(f"{t:<16}" + "".join(f"{curves[t].get(p, float('nan')):>8.2f}" for p in allpos))


def write_outputs(scored_ds: Dataset, means: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scored.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in scored_ds.to_list())
    )

    # CSV: rows = (trait, direction), cols = judged metrics.
    lines = ["conditioned_trait,direction," + ",".join(METRIC_NAMES)]
    for trait in ["baseline"] + TRAITS:
        for direction in (["neutral"] if trait == "baseline" else ["low", "high"]):
            row = means.get((trait, direction))
            if not row:
                continue
            lines.append(
                f"{trait},{direction}," + ",".join(f"{row.get(m, float('nan')):.3f}" for m in METRIC_NAMES)
            )
    (OUT_DIR / "trait_judge_matrix.csv").write_text("\n".join(lines) + "\n")

    # Diagonal steering effect: judged trait T, high - low.
    print("\n=== Diagonal steering effect (judge score on the conditioned trait) ===")
    print(f"{'trait':<18}{'low':>8}{'high':>8}{'Δ(high-low)':>14}")
    diag = {}
    for i, trait in enumerate(TRAITS):
        m = METRIC_NAMES[i]
        lo = means.get((trait, "low"), {}).get(m, float("nan"))
        hi = means.get((trait, "high"), {}).get(m, float("nan"))
        diag[trait] = hi - lo
        print(f"{trait:<18}{lo:>8.2f}{hi:>8.2f}{hi - lo:>14.2f}")

    _plot(means, diag)


def _plot(means: dict, diag: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return

    # Heatmap of judged scores at HIGH conditioning (rows=conditioned, cols=judged).
    mat = np.full((len(TRAITS), len(TRAITS)), np.nan)
    for i, trait in enumerate(TRAITS):
        for j, m in enumerate(METRIC_NAMES):
            mat[i, j] = means.get((trait, "high"), {}).get(m, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-4, vmax=4)
    ax1.set_xticks(range(len(TRAITS)))
    ax1.set_xticklabels([t[:4] for t in TRAITS], rotation=45)
    ax1.set_yticks(range(len(TRAITS)))
    ax1.set_yticklabels(TRAITS)
    ax1.set_xlabel("judged trait")
    ax1.set_ylabel("conditioned trait (high)")
    ax1.set_title("Avg OCEAN judge score @ high conditioning")
    for i in range(len(TRAITS)):
        for j in range(len(TRAITS)):
            if not np.isnan(mat[i, j]):
                ax1.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax1, fraction=0.046, label="avg judge score (−4..+4)")

    # Right: plain average own-trait judge score at low vs high (no deltas).
    x = np.arange(len(TRAITS))
    lo = [means.get((t, "low"), {}).get(m, np.nan) for t, m in zip(TRAITS, METRIC_NAMES)]
    hi = [means.get((t, "high"), {}).get(m, np.nan) for t, m in zip(TRAITS, METRIC_NAMES)]
    ax2.bar(x - 0.2, lo, 0.4, label="low conditioning", color="#4C72B0")
    ax2.bar(x + 0.2, hi, 0.4, label="high conditioning", color="#C44E52")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:4] for t in TRAITS], rotation=45)
    ax2.set_ylabel("avg judge score on conditioned trait (−4..+4)")
    ax2.set_title("Own-trait average judge score")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "fig_psychadapter_ocean.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote figure -> {out}")


def main():
    rows = load_generations()
    print(f"Scoring {len(rows)} generations across {len(METRIC_NAMES)} traits...")
    scored_ds = score(rows)
    means = summarize(scored_ds)
    write_outputs(scored_ds, means)
    curves = dose_response(scored_ds)
    if curves:
        _plot_dose_response(curves)


if __name__ == "__main__":
    main()

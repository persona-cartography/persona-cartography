"""Analyse the choice-mass distribution of logprob-scored questionnaire cells.

For every completed Stage-2 run (local or hydrated from HF), reads
``raw_responses.jsonl``, collects the per-cell ``choice_mass`` (the
probability the model put on the expected target tokens — digits 1..5
for Likert, letters A..D for trait_mcq, A..B for fc_pair), and reports
what a cutoff filter would cost us in retention.

Three framings, because each answers a different question:

    (i)   per-cell retention     — what fraction of cells survive a cutoff
    (ii)  per-rollout retention  — what fraction of PERSONAS survive the cutoff
                                   *assuming we drop any rollout with any
                                   below-threshold cell* (current
                                   preprocess_response_matrix semantics)
    (iii) per-item retention     — what fraction of ITEMS keep > X% of their
                                   rollouts above the cutoff

Inputs
------

The script auto-scans ``scratch/psychometric_fa`` for every
``questionnaire-*-lp*`` run and includes those. Set ``HYDRATE_FROM_HF``
to also fetch explicitly-named run-ids from the HF monorepo (for re-
running on a different machine or picking up runs you don't have
locally yet).

Outputs land under ``scratch/psychometric_fa/choice_mass_analysis/<tag>/``
— CSVs + four plots:

    1. Per-cell choice_mass distribution, overlaid by pair.
    2. Per-rollout MIN choice_mass distribution, overlaid by pair.
    3. Retention curves — frac rollouts surviving at each cutoff,
       one line per pair.
    4. Per-pair summary table as a heatmap (rows=pairs,
       cols=thresholds).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

SCRATCH_ROOT = Path("scratch/psychometric_fa")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
OUTPUT_DIR = SCRATCH_ROOT / "choice_mass_analysis" / "latest"

# Set True to also hydrate explicitly-named run-ids from HF when they're
# missing locally. Useful when moving between machines.
HYDRATE_FROM_HF: bool = False
HYDRATE_RUN_IDS: list[str] = []  # e.g. ["questionnaire-rollouts-external-..."]

# Cutoff thresholds to evaluate.
THRESHOLDS: list[float] = [0.0, 0.1, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95, 0.99]


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunSource:
    """One (preset, questionnaire) run's source data."""
    run_id: str
    preset_slug: str       # shortened, for display
    questionnaire_slug: str
    run_dir: Path
    n_expected_cells: int  # inferred from the raw_responses.jsonl K × N_items


def _parse_run_id(run_id: str) -> tuple[str, str]:
    """Extract (preset_slug, questionnaire_slug) from a questionnaire run-id.

    Run-ids look like:
      questionnaire-rollouts-external-<source>-<modelslug>-<N>p-seed<S>[-f_<tag>]
        -q_<version>-<blocks>-<phrasing>-lp<K>[-reset_*][-qm_*]

    We truncate to a compact display form:
      preset_slug = "<source>/<modelslug>[-<tag>]"
      questionnaire_slug = "<version>"
    """
    # Drop the leading "questionnaire-rollouts-[external-]" prefix.
    tail = run_id
    for prefix in ("questionnaire-rollouts-external-", "questionnaire-rollouts-"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break

    # The format after prefix: <source>-<modelslug>-<N>p-seed<S>[-f_<tag>]-q_<ver>-<rest>
    m = re.match(
        r"^(?P<source>[^-]+)-(?P<model>[a-z0-9]+)-\d+p-seed\d+"
        r"(?:-f_(?P<tag>[a-z0-9_]+))?"
        r"-q_(?P<version>[a-z0-9_]+)-.*$",
        tail,
    )
    if not m:
        # Fall back to a generic truncation.
        return tail[:40], tail[-30:]
    source = m.group("source")
    model = m.group("model")
    tag = m.group("tag") or ""
    version = m.group("version")
    preset_slug = f"{source}/{model}"
    if tag:
        preset_slug += f"-{tag}"
    return preset_slug, version


def _find_local_runs(root: Path) -> list[Path]:
    """Return every Stage-2 dir with a non-empty raw_responses.jsonl."""
    out: list[Path] = []
    for p in root.glob("questionnaire-*-lp*"):
        raw = p / "questionnaire" / "raw_responses.jsonl"
        if raw.exists() and raw.stat().st_size > 0:
            out.append(p)
    return sorted(out)


def _maybe_hydrate_from_hf(run_ids: list[str]) -> None:
    for rid in run_ids:
        dest = SCRATCH_ROOT / rid
        if (dest / "questionnaire" / "raw_responses.jsonl").exists():
            continue
        print(f"[Hydrate] {rid} from HF…")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_runs_path(rid, "questionnaire"),
            local_dir=dest / "questionnaire",
            required=False,
        )


def load_run(run_dir: Path) -> tuple[RunSource, pd.DataFrame]:
    """Read raw_responses.jsonl → DataFrame of (k, item_id, choice_mass, probs)."""
    raw = run_dir / "questionnaire" / "raw_responses.jsonl"
    rows: list[dict] = []
    with raw.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append({
                "k": entry["k"],
                "item_id": entry["item_id"],
                "choice_mass": entry.get("choice_mass"),
                "parsed_choice": entry.get("parsed_choice"),
                "n_probs": len(entry.get("probs") or {}),
            })
    df = pd.DataFrame(rows)
    preset_slug, questionnaire_slug = _parse_run_id(run_dir.name)
    src = RunSource(
        run_id=run_dir.name,
        preset_slug=preset_slug,
        questionnaire_slug=questionnaire_slug,
        run_dir=run_dir,
        n_expected_cells=len(df),
    )
    return src, df


# ═════════════════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════════════════


def per_cell_summary(df: pd.DataFrame, src: RunSource) -> dict[str, Any]:
    cm = df["choice_mass"].astype(float)
    cm = cm[~cm.isna()]
    out: dict[str, Any] = {
        "run_id": src.run_id,
        "preset_slug": src.preset_slug,
        "questionnaire_slug": src.questionnaire_slug,
        "n_cells": int(len(df)),
        "n_with_choice_mass": int(len(cm)),
    }
    if not len(cm):
        for t in THRESHOLDS:
            out[f"frac_cells_below_{t}"] = np.nan
        return out
    out["cm_mean"] = float(cm.mean())
    out["cm_median"] = float(cm.median())
    out["cm_p10"] = float(cm.quantile(0.10))
    out["cm_p25"] = float(cm.quantile(0.25))
    for t in THRESHOLDS:
        out[f"frac_cells_below_{t}"] = float((cm < t).mean())
    return out


def per_rollout_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Rollout-level stats: for each persona, what's their min choice_mass?

    A rollout survives a cutoff t iff every cell has choice_mass >= t. The
    dropping semantics mirror ``preprocess_response_matrix`` which removes
    any row with any NaN (here we'd NaN any cell below the cutoff).
    """
    tmp = df.dropna(subset=["choice_mass"])
    rollout_min = tmp.groupby("k")["choice_mass"].min().rename("min_choice_mass")
    return rollout_min.reset_index()


def retention_curve(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """At each threshold, fraction of cells / rollouts / items surviving."""
    per_rollout = per_rollout_retention(df)
    total_rollouts = per_rollout.shape[0]
    total_cells = df["choice_mass"].notna().sum()
    # Per-item: at each threshold, how many of the item's rollouts survive.
    rows = []
    for t in thresholds:
        frac_cells = (df["choice_mass"].fillna(0) >= t).mean()
        frac_rollouts = (per_rollout["min_choice_mass"] >= t).mean()
        # Per-item: distribution of "frac of rollouts surviving for this item"
        per_item_survive = (
            df.assign(survive=(df["choice_mass"].fillna(0) >= t))
              .groupby("item_id")["survive"].mean()
        )
        rows.append({
            "threshold": t,
            "frac_cells_surviving": float(frac_cells),
            "frac_rollouts_surviving": float(frac_rollouts),
            "per_item_p10_survival": float(per_item_survive.quantile(0.10)),
            "per_item_p50_survival": float(per_item_survive.quantile(0.50)),
            "per_item_p90_survival": float(per_item_survive.quantile(0.90)),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════


def _colors(labels: list[str]):
    import matplotlib.pyplot as plt
    return plt.cm.tab20(np.linspace(0, 1, max(len(labels), 3)))


def plot_per_cell_distribution(
    per_run: dict[str, pd.DataFrame], out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    labels = list(per_run.keys())
    colors = _colors(labels)
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 1, 41)
    for (label, df), c in zip(per_run.items(), colors):
        cm = df["choice_mass"].dropna().to_numpy()
        if not len(cm):
            continue
        ax_hist.hist(cm, bins=bins, histtype="step", linewidth=1.8,
                     color=c, label=label)
        s = np.sort(cm)
        y = np.arange(1, len(s) + 1) / len(s)
        ax_cdf.plot(s, y, color=c, lw=1.5, label=label)
    ax_hist.set_xlabel("choice_mass (per-cell)")
    ax_hist.set_ylabel("# cells")
    ax_hist.set_title("(A) Per-cell choice_mass distribution")
    ax_hist.legend(fontsize=7, loc="upper left")
    ax_cdf.set_xlabel("choice_mass (per-cell)")
    ax_cdf.set_ylabel("cumulative fraction")
    ax_cdf.set_title("(B) Per-cell choice_mass CDF")
    for t in [0.5, 0.85, 0.95]:
        ax_cdf.axvline(t, color="gray", lw=0.6, ls=":")
    ax_cdf.legend(fontsize=7, loc="lower right")
    ax_cdf.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_per_rollout_min(
    per_run: dict[str, pd.DataFrame], out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    labels = list(per_run.keys())
    colors = _colors(labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)
    for (label, df), c in zip(per_run.items(), colors):
        per = per_rollout_retention(df)
        if not per.shape[0]:
            continue
        ax.hist(per["min_choice_mass"].to_numpy(), bins=bins, histtype="step",
                linewidth=1.8, color=c, label=label)
    ax.set_xlabel("MIN choice_mass across the rollout's cells")
    ax.set_ylabel("# rollouts")
    ax.set_title(
        "Per-rollout minimum choice_mass — if cutoff > this value, "
        "the rollout gets dropped"
    )
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_retention_curves(
    retention_by_run: dict[str, pd.DataFrame], out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    labels = list(retention_by_run.keys())
    colors = _colors(labels)
    fig, (ax_cells, ax_rollouts) = plt.subplots(1, 2, figsize=(14, 5))
    for (label, rc), c in zip(retention_by_run.items(), colors):
        ax_cells.plot(rc["threshold"], rc["frac_cells_surviving"],
                      marker="o", color=c, label=label)
        ax_rollouts.plot(rc["threshold"], rc["frac_rollouts_surviving"],
                         marker="o", color=c, label=label)
    for ax, title in [
        (ax_cells, "(A) Frac CELLS surviving cutoff"),
        (ax_rollouts, "(B) Frac ROLLOUTS surviving cutoff (all cells ≥ t)"),
    ]:
        ax.set_xlabel("choice_mass cutoff")
        ax.set_ylabel("fraction surviving")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
    ax_cells.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_retention_heatmap(
    retention_by_run: dict[str, pd.DataFrame], out_path: Path,
) -> None:
    """Rows = runs, cols = thresholds, values = frac rollouts surviving."""
    import matplotlib.pyplot as plt
    labels = list(retention_by_run.keys())
    if not labels:
        return
    thresholds = retention_by_run[labels[0]]["threshold"].tolist()
    mat = np.array([
        retention_by_run[l]["frac_rollouts_surviving"].to_numpy()
        for l in labels
    ])
    fig, ax = plt.subplots(figsize=(1.2 * len(thresholds) + 4, 0.4 * len(labels) + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if 0.2 < mat[i, j] < 0.8 else "white")
    ax.set_xticks(range(len(thresholds)), [f"{t:g}" for t in thresholds])
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("choice_mass cutoff")
    ax.set_title("Frac ROLLOUTS surviving cutoff — per run")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if HYDRATE_FROM_HF and HYDRATE_RUN_IDS:
        _maybe_hydrate_from_hf(HYDRATE_RUN_IDS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = _find_local_runs(SCRATCH_ROOT)
    print(f"Found {len(run_dirs)} logprob-mode Stage-2 runs locally.\n")
    if not run_dirs:
        print("Nothing to analyse. Set HYDRATE_FROM_HF=True + HYDRATE_RUN_IDS to pull.")
        return

    per_run: dict[str, pd.DataFrame] = {}
    retention_by_run: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    for rd in run_dirs:
        try:
            src, df = load_run(rd)
        except Exception as e:
            logger.warning("Failed to load %s: %s", rd, e)
            continue
        if not len(df) or df["choice_mass"].isna().all():
            print(f"[skip] {rd.name}: no choice_mass (non-logprob run)")
            continue
        label = f"{src.preset_slug} × {src.questionnaire_slug}  (n={len(df)})"
        per_run[label] = df
        retention_by_run[label] = retention_curve(df, THRESHOLDS)
        summaries.append(per_cell_summary(df, src))

    if not per_run:
        print("No logprob-mode runs with choice_mass found.")
        return

    # Write summary CSVs.
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "per_cell_summary.csv", index=False)
    print(f"[Write] {OUTPUT_DIR / 'per_cell_summary.csv'}")

    retention_rows = []
    for label, rc in retention_by_run.items():
        for _, row in rc.iterrows():
            retention_rows.append({"run": label, **row.to_dict()})
    pd.DataFrame(retention_rows).to_csv(
        OUTPUT_DIR / "retention_by_threshold.csv", index=False,
    )
    print(f"[Write] {OUTPUT_DIR / 'retention_by_threshold.csv'}")

    # Compact printout.
    print("\n=== Summary ===")
    print(f"{'run':<70} {'cm_p50':>8} {'cm_p10':>8} {'frac≥0.5':>9} {'frac≥0.9':>9}")
    for s in summaries:
        print(
            f"{s['run_id'][:68]:<70} "
            f"{s.get('cm_median', float('nan')):>8.3f} "
            f"{s.get('cm_p10', float('nan')):>8.3f} "
            f"{1 - s.get('frac_cells_below_0.5', 0):>9.3f} "
            f"{1 - s.get('frac_cells_below_0.9', 0):>9.3f}"
        )

    print("\n=== Rollout-level retention at selected thresholds ===")
    print(f"{'run':<70} {'t≥0.5':>8} {'t≥0.85':>8} {'t≥0.95':>8}")
    for label, rc in retention_by_run.items():
        row = {r["threshold"]: r["frac_rollouts_surviving"] for _, r in rc.iterrows()}
        print(
            f"{label[:68]:<70} "
            f"{row.get(0.5, float('nan')):>8.3f} "
            f"{row.get(0.85, float('nan')):>8.3f} "
            f"{row.get(0.95, float('nan')):>8.3f}"
        )

    # Plots.
    plot_per_cell_distribution(per_run, OUTPUT_DIR / "per_cell_choice_mass.png")
    plot_per_rollout_min(per_run, OUTPUT_DIR / "per_rollout_min.png")
    plot_retention_curves(retention_by_run, OUTPUT_DIR / "retention_curves.png")
    plot_retention_heatmap(retention_by_run, OUTPUT_DIR / "retention_heatmap.png")

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Cross-model psychometric-response comparison.

Loads persona × item response matrices produced by the psychometric-FA
pipeline for multiple models (administered on the same rollouts), aligns
them by sample_id, and computes:

    1. Marginal distribution comparison — per-item mean/std, KS tests,
       histograms grid.
    2. Per-persona agreement — per-item Spearman/Pearson correlation of
       paired responses, per-persona mean-response correlation, exact-
       match rate, disagreement matrix.
    3. Matrix-level summary — flattened paired-cell correlation.

Inputs are HF dataset-repo run-ids (hydrated locally on first use). Outputs
land under ``scratch/psychometric_comparison/<tag>/``.

Factor-analysis comparison is out of scope for v1 — it needs per-model FA
outputs, which we produce separately via ``psychometric_rollout_fa.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit in place to change the comparison
# ═════════════════════════════════════════════════════════════════════════════

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
SCRATCH_CACHE = Path("scratch/psychometric_fa")
OUTPUT_ROOT = Path("scratch/psychometric_comparison")


@dataclass(frozen=True)
class ModelRun:
    """One (model, questionnaire) result to include in the comparison."""
    label: str              # short tag used in plots and filenames
    questionnaire: str      # e.g. "v5", "trait_ocean_v1"
    run_id: str             # HF run-id under ``runs/<run_id>/questionnaire``


# The base rollout run-id prefix — shared by every entry below.
# Kept as a single constant so new models are one-liners.
_B_PREFIX = (
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6"
)

RUNS: list[ModelRun] = [
    # v5 direct-gen (Likert integer, 1-5). The original v5 comparison.
    ModelRun(
        label="llama-direct",
        questionnaire="v5",
        run_id=f"{_B_PREFIX}-q_v5-likert-direct",
    ),
    ModelRun(
        label="qwen2.5-direct",
        questionnaire="v5",
        run_id=f"{_B_PREFIX}-q_v5-likert-direct-qm_qwen257binstruct",
    ),
    # v5 logprob (expected Likert value on [1, 5] via Σd·P(d)). The
    # modality-controlled comparison that disentangles direct-gen vs logprob
    # from the v5 Likert / trait_mcq ABCD asymmetry seen earlier.
    ModelRun(
        label="llama-logprob",
        questionnaire="v5",
        run_id=f"{_B_PREFIX}-q_v5-likert-direct-lp20",
    ),
    ModelRun(
        label="qwen2.5-logprob",
        questionnaire="v5",
        run_id=f"{_B_PREFIX}-q_v5-likert-direct-lp20-qm_qwen257binstruct",
    ),
]

# Which questionnaires to compare. Each is compared across every model
# present in RUNS that has a matching ``questionnaire`` tag. If only one
# model has a given questionnaire, it's skipped with a warning.
QUESTIONNAIRES_TO_COMPARE: list[str] = ["v5"]

# Output subdirectory tag; results written to OUTPUT_ROOT / TAG.
TAG = "B_v5_modality_x_model"


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LoadedRun:
    label: str
    questionnaire: str
    run_id: str
    matrix: np.ndarray          # (n_personas, n_items)
    sample_ids: list[str]
    items: list[dict]


def _ensure_local(run_id: str) -> Path:
    """Download the questionnaire subtree from HF if not already cached."""
    local_dir = SCRATCH_CACHE / run_id / "questionnaire"
    expected = local_dir / "response_matrix.npy"
    if not expected.exists():
        print(f"[Load] Hydrating {run_id} from HF…")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_runs_path(run_id, "questionnaire"),
            local_dir=local_dir,
            required=True,
        )
    return local_dir


def load_run(run: ModelRun) -> LoadedRun:
    d = _ensure_local(run.run_id)
    matrix = np.load(d / "response_matrix.npy")
    with open(d / "metadata.jsonl") as f:
        meta = [json.loads(l) for l in f if l.strip()]
    with open(d / "items.json") as f:
        items = json.load(f)
    sample_ids = [m["sample_id"] for m in meta]
    if len(sample_ids) != matrix.shape[0]:
        raise ValueError(f"{run.label}: metadata/matrix row mismatch")
    if len(items) != matrix.shape[1]:
        raise ValueError(f"{run.label}: items/matrix column mismatch")
    return LoadedRun(
        label=run.label,
        questionnaire=run.questionnaire,
        run_id=run.run_id,
        matrix=matrix,
        sample_ids=sample_ids,
        items=items,
    )


def align_runs(runs: list[LoadedRun]) -> tuple[np.ndarray, list[str], list[dict], list[np.ndarray]]:
    """Align runs to the intersection of sample_ids and shared columns.

    All runs in a single call must share the same questionnaire (i.e.
    matching item_ids). Returns (persona_ids, item_ids, items, matrices-per-run).
    """
    qs = {r.questionnaire for r in runs}
    if len(qs) != 1:
        raise ValueError(f"align_runs requires a single questionnaire, got {qs}")

    # Column alignment — by item_id (must be identical set; same questionnaire).
    item_id_sets = [[it["item_id"] for it in r.items] for r in runs]
    shared_item_ids = sorted(set(item_id_sets[0]).intersection(*item_id_sets[1:]))
    if not shared_item_ids:
        raise ValueError("No overlapping items across runs")
    if any(set(s) != set(shared_item_ids) for s in item_id_sets):
        print(f"[Align] Columns differ across runs; using intersection "
              f"({len(shared_item_ids)} items)")

    # Use the first run's item records (textually identical across runs).
    iid_to_item = {it["item_id"]: it for it in runs[0].items}
    items = [iid_to_item[iid] for iid in shared_item_ids]

    # Row alignment — intersection of sample_ids.
    shared_samples = set(runs[0].sample_ids)
    for r in runs[1:]:
        shared_samples &= set(r.sample_ids)
    persona_ids = sorted(shared_samples)
    print(
        f"[Align] {len(persona_ids)} shared personas across "
        f"{len(runs)} runs (input sizes: "
        f"{[len(r.sample_ids) for r in runs]})"
    )

    aligned: list[np.ndarray] = []
    for r in runs:
        sid_to_row = {sid: i for i, sid in enumerate(r.sample_ids)}
        iid_to_col = {iid: i for i, iid in enumerate(item_id_sets[runs.index(r)])}
        rows = [sid_to_row[sid] for sid in persona_ids]
        cols = [iid_to_col[iid] for iid in shared_item_ids]
        aligned.append(r.matrix[np.ix_(rows, cols)])

    return np.array(persona_ids, dtype=object), shared_item_ids, items, aligned


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSES
# ═════════════════════════════════════════════════════════════════════════════


def marginal_distribution_table(
    runs: list[LoadedRun],
    items: list[dict],
    matrices: list[np.ndarray],
) -> pd.DataFrame:
    """Per-item marginal stats for each model + KS test across model pairs."""
    rows = []
    for j, it in enumerate(items):
        row = {"item_id": it["item_id"], "text": it["text"]}
        cols_by_model = {}
        for r, m in zip(runs, matrices):
            col = m[:, j]
            col = col[~np.isnan(col)]
            cols_by_model[r.label] = col
            row[f"{r.label}:mean"] = float(np.mean(col))
            row[f"{r.label}:std"] = float(np.std(col))
        # Pairwise KS between models (useful when more than 2 models present)
        for a, b in itertools.combinations([r.label for r in runs], 2):
            ks = stats.ks_2samp(cols_by_model[a], cols_by_model[b])
            row[f"KS({a},{b})"] = float(ks.statistic)
            row[f"KS_p({a},{b})"] = float(ks.pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_agreement_table(
    runs: list[LoadedRun],
    items: list[dict],
    matrices: list[np.ndarray],
) -> pd.DataFrame:
    """Per-item per-pair Spearman / Pearson / exact-match rate on
    sample-id-aligned responses."""
    rows = []
    labels = [r.label for r in runs]
    pairs = list(itertools.combinations(range(len(runs)), 2))
    for j, it in enumerate(items):
        row = {"item_id": it["item_id"], "text": it["text"]}
        for ia, ib in pairs:
            a, b = matrices[ia][:, j], matrices[ib][:, j]
            mask = ~(np.isnan(a) | np.isnan(b))
            a_, b_ = a[mask], b[mask]
            la, lb = labels[ia], labels[ib]
            if a_.size < 2:
                continue
            # Spearman — rank-based, handles ties. Guard on zero-variance.
            if np.std(a_) == 0 or np.std(b_) == 0:
                row[f"spearman({la},{lb})"] = np.nan
                row[f"pearson({la},{lb})"] = np.nan
            else:
                row[f"spearman({la},{lb})"] = float(stats.spearmanr(a_, b_).statistic)
                row[f"pearson({la},{lb})"] = float(np.corrcoef(a_, b_)[0, 1])
            row[f"exact_match({la},{lb})"] = float(np.mean(a_ == b_))
            row[f"|diff|_mean({la},{lb})"] = float(np.mean(np.abs(a_ - b_)))
            row[f"mean_a({la},{lb})"] = float(np.mean(a_))
            row[f"mean_b({la},{lb})"] = float(np.mean(b_))
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_level_summary(
    runs: list[LoadedRun],
    matrices: list[np.ndarray],
) -> pd.DataFrame:
    """Flattened cell-level + per-persona mean-response correlation per pair."""
    rows = []
    labels = [r.label for r in runs]
    for ia, ib in itertools.combinations(range(len(runs)), 2):
        A, B = matrices[ia], matrices[ib]
        mask = ~(np.isnan(A) | np.isnan(B))
        a_, b_ = A[mask], B[mask]
        persona_mean_a = np.nanmean(A, axis=1)
        persona_mean_b = np.nanmean(B, axis=1)
        rows.append({
            "a": labels[ia],
            "b": labels[ib],
            "n_cells": int(mask.sum()),
            "cell_spearman": float(stats.spearmanr(a_, b_).statistic),
            "cell_pearson": float(np.corrcoef(a_, b_)[0, 1]),
            "cell_exact_match": float(np.mean(a_ == b_)),
            "cell_|diff|_mean": float(np.mean(np.abs(a_ - b_))),
            "persona_mean_pearson": float(np.corrcoef(persona_mean_a, persona_mean_b)[0, 1]),
            "persona_mean_spearman": float(
                stats.spearmanr(persona_mean_a, persona_mean_b).statistic
            ),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════


def _response_is_discrete(matrices: list[np.ndarray], max_unique: int = 10) -> bool:
    """Return True when responses form a small discrete set (Likert, ABCD, etc.).

    Logprob-scored trait_mcq produces near-continuous floats so treat as
    continuous and use fixed-bin histograms instead of per-value bars.
    """
    vals = np.concatenate([m.ravel() for m in matrices])
    vals = vals[~np.isnan(vals)]
    return len(np.unique(vals)) <= max_unique


def plot_per_item_histograms(
    runs: list[LoadedRun],
    items: list[dict],
    matrices: list[np.ndarray],
    out_path: Path,
    max_items: int = 100,
) -> None:
    """Grid of per-item response histograms, one cluster/overlay per model.

    Discrete case (Likert 1-5, ABCD): bars at each unique value.
    Continuous case (trait_mcq logprob probabilities): fixed 20-bin histogram
    rendered as step plots (bars would overlap unreadably with 3 models).
    """
    n = min(len(items), max_items)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 1.8),
                             sharex=True, sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    discrete = _response_is_discrete(matrices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    if discrete:
        all_vals = np.concatenate([m.ravel() for m in matrices])
        all_vals = all_vals[~np.isnan(all_vals)]
        unique_vals = sorted(np.unique(all_vals).tolist())
        bin_edges = np.array(unique_vals + [unique_vals[-1] + 1]) - 0.5
        centers = np.array(unique_vals)
        bar_w = 0.8 / len(runs)
    else:
        vmin = min(np.nanmin(m) for m in matrices)
        vmax = max(np.nanmax(m) for m in matrices)
        bin_edges = np.linspace(vmin, vmax, 21)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for j in range(n):
        ax = axes[j // ncols, j % ncols]
        for k, (r, m) in enumerate(zip(runs, matrices)):
            col = m[:, j]
            col = col[~np.isnan(col)]
            counts, _ = np.histogram(col, bins=bin_edges)
            counts = counts / max(counts.sum(), 1)
            if discrete:
                ax.bar(centers + (k - (len(runs) - 1) / 2) * bar_w, counts,
                       width=bar_w, color=colors[k], label=r.label)
            else:
                ax.plot(bin_centers, counts, color=colors[k], label=r.label, lw=1.2)
                ax.fill_between(bin_centers, 0, counts, color=colors[k], alpha=0.2)
        ax.set_title(f"#{items[j]['item_id']}", fontsize=7)
        ax.tick_params(labelsize=6)
        if discrete:
            ax.set_ylim(0, 1)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    # Single shared legend
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(
        f"Per-item response distributions — "
        f"{', '.join(r.label for r in runs)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_agreement_bars(
    agreement_df: pd.DataFrame,
    runs: list[LoadedRun],
    out_path: Path,
) -> None:
    """Per-item Spearman bars, one panel per model-pair, sorted high→low."""
    labels = [r.label for r in runs]
    pairs = list(itertools.combinations(labels, 2))
    fig, axes = plt.subplots(len(pairs), 1, figsize=(max(8, len(agreement_df) * 0.12),
                                                      3.2 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (la, lb) in zip(axes, pairs):
        col = f"spearman({la},{lb})"
        if col not in agreement_df.columns:
            ax.axis("off")
            continue
        s = agreement_df[["item_id", col]].dropna().sort_values(col, ascending=False)
        ax.bar(range(len(s)), s[col].values, color="steelblue")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"Per-item Spearman: {la} vs {lb}  "
                     f"(median={s[col].median():.3f}, mean={s[col].mean():.3f})")
        ax.set_xlabel("Item rank (high→low agreement)")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(-0.2, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_persona_mean_scatter(
    runs: list[LoadedRun],
    matrices: list[np.ndarray],
    out_path: Path,
) -> None:
    """Scatter of per-persona mean responses, one panel per model pair."""
    labels = [r.label for r in runs]
    pairs = list(itertools.combinations(range(len(runs)), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4), squeeze=False)
    for ax, (ia, ib) in zip(axes[0], pairs):
        a = np.nanmean(matrices[ia], axis=1)
        b = np.nanmean(matrices[ib], axis=1)
        ax.scatter(a, b, s=4, alpha=0.4)
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], color="red", lw=0.8)
        rho = float(stats.spearmanr(a, b).statistic)
        ax.set_xlabel(f"{labels[ia]} — mean response per persona")
        ax.set_ylabel(f"{labels[ib]} — mean response per persona")
        ax.set_title(f"ρ = {rho:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_disagreement_heatmap(
    runs: list[LoadedRun],
    items: list[dict],
    matrices: list[np.ndarray],
    out_path: Path,
    pair_idx: tuple[int, int] = (0, 1),
) -> None:
    """Confusion-matrix-style heatmap of (a_response, b_response) cell counts,
    aggregated across all items and personas. Useful for spotting systematic
    shifts (e.g. one model consistently answers higher)."""
    ia, ib = pair_idx
    A, B = matrices[ia], matrices[ib]
    mask = ~(np.isnan(A) | np.isnan(B))
    discrete = _response_is_discrete(matrices)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    if discrete:
        a_, b_ = A[mask].astype(int), B[mask].astype(int)
        vals = sorted(set(a_.tolist()) | set(b_.tolist()))
        vmap = {v: i for i, v in enumerate(vals)}
        H = np.zeros((len(vals), len(vals)), dtype=int)
        for x, y in zip(a_, b_):
            H[vmap[x], vmap[y]] += 1
        im = ax.imshow(H, cmap="Blues", origin="lower")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j, i, f"{H[i, j]:,}", ha="center", va="center",
                        fontsize=8, color="black" if H[i, j] < H.max() * 0.5 else "white")
        ax.set_xticks(range(len(vals)), vals)
        ax.set_yticks(range(len(vals)), vals)
    else:
        a_, b_ = A[mask], B[mask]
        H, xe, ye = np.histogram2d(a_, b_, bins=40)
        im = ax.imshow(H.T, cmap="Blues", origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto")
        ax.plot([xe[0], xe[-1]], [xe[0], xe[-1]], color="red", lw=0.6, ls="--")
    ax.set_xlabel(f"{runs[ib].label} response")
    ax.set_ylabel(f"{runs[ia].label} response")
    ax.set_title(f"Joint response distribution (n={mask.sum():,} cells)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def plot_summary_overview(
    runs: list[LoadedRun],
    items: list[dict],
    matrices: list[np.ndarray],
    agreement_df: pd.DataFrame,
    marginal_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    pair_idx: tuple[int, int] = (0, 1),
) -> None:
    """One-page overview of the cross-model comparison.

    Six panels:
      (A) Persona-mean scatter — the "do personas rank similarly?" view.
      (B) Per-item Spearman histogram with mean/median markers.
      (C) Joint cell-level response heatmap (both marginals rendered above/right).
      (D) Per-item mean-shift histogram (systematic bias in either direction).
      (E) Top-10 agreeing items (horizontal bars with item text).
      (F) Bottom-10 agreeing items.
    """
    ia, ib = pair_idx
    la, lb = runs[ia].label, runs[ib].label
    A, B = matrices[ia], matrices[ib]

    # Canonical column names for this pair.
    sp_col = f"spearman({la},{lb})"
    mean_shift = marginal_df[f"{lb}:mean"] - marginal_df[f"{la}:mean"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.05, 1.05, 1.4], hspace=0.45, wspace=0.35
    )

    # ── (A) Persona-mean scatter ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    a = np.nanmean(A, axis=1)
    b = np.nanmean(B, axis=1)
    ax.scatter(a, b, s=5, alpha=0.3, color="steelblue")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    ax.plot([lo, hi], [lo, hi], color="red", lw=0.8, label="y=x")
    rho = summary_df.iloc[0]["persona_mean_spearman"]
    pr = summary_df.iloc[0]["persona_mean_pearson"]
    ax.set_xlabel(f"{la} — mean response per persona")
    ax.set_ylabel(f"{lb} — mean response per persona")
    ax.set_title(f"(A) Persona means  ρ={rho:.3f}  r={pr:.3f}", fontsize=11)
    ax.legend(fontsize=8)

    # ── (B) Per-item Spearman histogram ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    sp = agreement_df[sp_col].dropna()
    ax.hist(sp, bins=25, color="steelblue", edgecolor="white")
    ax.axvline(float(sp.mean()), color="red", lw=1.2,
               label=f"mean={sp.mean():.3f}")
    ax.axvline(float(sp.median()), color="orange", lw=1.2, ls="--",
               label=f"median={sp.median():.3f}")
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("Per-item Spearman ρ")
    ax.set_ylabel("Items")
    ax.set_title(
        f"(B) Per-item agreement (n={len(sp)})\n"
        f"ρ>0.5: {(sp > 0.5).mean():.0%}   ρ<0.2: {(sp < 0.2).mean():.0%}",
        fontsize=11,
    )
    ax.legend(fontsize=8)

    # ── (C) Joint cell-level response heatmap ──────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    mask = ~(np.isnan(A) | np.isnan(B))
    discrete = _response_is_discrete(matrices)
    if discrete:
        a_, b_ = A[mask].astype(int), B[mask].astype(int)
        vals_ = sorted(set(a_.tolist()) | set(b_.tolist()))
        vmap = {v: i for i, v in enumerate(vals_)}
        H = np.zeros((len(vals_), len(vals_)), dtype=float)
        for x, y in zip(a_, b_):
            H[vmap[x], vmap[y]] += 1
        H_norm = H / H.sum()
        im = ax.imshow(H_norm, cmap="Blues", origin="lower")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                frac = H_norm[i, j]
                ax.text(j, i, f"{frac:.1%}", ha="center", va="center",
                        fontsize=8,
                        color="black" if frac < H_norm.max() * 0.5 else "white")
        ax.set_xticks(range(len(vals_)), vals_)
        ax.set_yticks(range(len(vals_)), vals_)
        diag = np.trace(H_norm)
        ax.set_title(
            f"(C) Joint cell distribution (n={mask.sum():,})\n"
            f"exact match on diagonal: {diag:.1%}",
            fontsize=11,
        )
    else:
        a_, b_ = A[mask], B[mask]
        vmin = min(a_.min(), b_.min())
        vmax = max(a_.max(), b_.max())
        H, xe, ye = np.histogram2d(a_, b_, bins=40, range=[[vmin, vmax], [vmin, vmax]])
        im = ax.imshow(H.T, cmap="Blues", origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto")
        ax.plot([vmin, vmax], [vmin, vmax], color="red", lw=0.6, ls="--")
        # Near-diagonal concentration: fraction within ±0.1 of y=x.
        near = (np.abs(a_ - b_) < 0.1).mean()
        ax.set_title(
            f"(C) Joint cell density (n={mask.sum():,})\n"
            f"within ±0.1 of y=x: {near:.1%}",
            fontsize=11,
        )
    ax.set_xlabel(f"{lb} response")
    ax.set_ylabel(f"{la} response")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # ── (D) Per-item mean-shift histogram ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(mean_shift, bins=25, color="purple", edgecolor="white", alpha=0.7)
    ax.axvline(0, color="gray", lw=0.6)
    ax.axvline(float(mean_shift.mean()), color="red", lw=1.2,
               label=f"mean={mean_shift.mean():+.3f}")
    ax.axvline(float(mean_shift.median()), color="orange", lw=1.2, ls="--",
               label=f"median={mean_shift.median():+.3f}")
    ax.set_xlabel(f"Per-item mean shift  ({lb} − {la})")
    ax.set_ylabel("Items")
    ax.set_title(
        f"(D) Marginal shift per item\n"
        f"{lb} higher: {(mean_shift > 0).sum()}   "
        f"{la} higher: {(mean_shift < 0).sum()}",
        fontsize=11,
    )
    ax.legend(fontsize=8)

    # ── (E) KS statistic per item (marginal distribution shape) ────────────
    ax = fig.add_subplot(gs[1, 1])
    ks_col = f"KS({la},{lb})"
    ks = marginal_df[ks_col].dropna() if ks_col in marginal_df.columns else pd.Series(dtype=float)
    if len(ks):
        ax.hist(ks, bins=25, color="teal", edgecolor="white")
        ax.axvline(float(ks.median()), color="orange", lw=1.2, ls="--",
                   label=f"median={ks.median():.3f}")
        ax.set_xlabel("KS statistic (marginals)")
        ax.set_ylabel("Items")
        ax.set_title(f"(E) Marginal shape divergence\n(higher = more different)",
                     fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    # ── (F) Exact-match rate (discrete) or per-item |diff| (continuous) ────
    ax = fig.add_subplot(gs[1, 2])
    if discrete:
        em_col = f"exact_match({la},{lb})"
        em = agreement_df[em_col].dropna()
        ax.hist(em, bins=25, color="darkgreen", edgecolor="white", alpha=0.7)
        ax.axvline(float(em.mean()), color="red", lw=1.2, label=f"mean={em.mean():.3f}")
        ax.axvline(float(em.median()), color="orange", lw=1.2, ls="--",
                   label=f"median={em.median():.3f}")
        # Marginals-based chance (sum of per-value product probabilities).
        a_int = A[mask].astype(int)
        b_int = B[mask].astype(int)
        uniq = sorted(set(a_int.tolist()) | set(b_int.tolist()))
        chance = float(sum((a_int == v).mean() * (b_int == v).mean() for v in uniq))
        ax.axvline(chance, color="black", lw=1.0, ls=":",
                   label=f"marginals chance={chance:.3f}")
        ax.set_xlabel("Exact-match rate")
        ax.set_title("(F) Exact-match rate per item\n(vs marginals-based chance)",
                     fontsize=11)
    else:
        diff_col = f"|diff|_mean({la},{lb})"
        if diff_col in agreement_df.columns:
            dd = agreement_df[diff_col].dropna()
            ax.hist(dd, bins=25, color="darkgreen", edgecolor="white", alpha=0.7)
            ax.axvline(float(dd.mean()), color="red", lw=1.2,
                       label=f"mean={dd.mean():.3f}")
            ax.axvline(float(dd.median()), color="orange", lw=1.2, ls="--",
                       label=f"median={dd.median():.3f}")
        ax.set_xlabel(f"Per-item mean |a − b|")
        ax.set_title("(F) Mean absolute per-item disagreement", fontsize=11)
    ax.set_ylabel("Items")
    ax.legend(fontsize=8)

    # ── (G) Top/Bottom agreeing items — horizontal bars ────────────────────
    ax = fig.add_subplot(gs[2, :])
    topk = 10
    srt = agreement_df.sort_values(sp_col)
    bottom = srt.head(topk)
    top = srt.tail(topk).iloc[::-1]
    rows = pd.concat([top, bottom])
    colors = (["#2E7D32"] * len(top)) + (["#B71C1C"] * len(bottom))
    labels = [
        f"#{r.item_id}: {r.text[:120] + ('…' if len(r.text) > 120 else '')}"
        for r in rows.itertuples()
    ]
    y = np.arange(len(rows))
    ax.barh(y, rows[sp_col].values, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.6)
    ax.axvline(float(sp.mean()), color="red", lw=1.0, ls="--",
               label=f"overall mean ρ={sp.mean():.3f}")
    ax.set_xlabel("Per-item Spearman ρ")
    ax.set_title(
        f"(G) Top-{topk} agreeing (green) vs bottom-{topk} (red) items — "
        f"{la} vs {lb}",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Cross-model response comparison — {la} vs {lb}  "
        f"(questionnaire: {runs[0].questionnaire},  "
        f"n personas = {A.shape[0]:,},  n items = {A.shape[1]})",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═════════════════════════════════════════════════════════════════════════════


def run_comparison(
    runs: list[LoadedRun],
    questionnaire: str,
    out_dir: Path,
) -> None:
    print(f"\n{'=' * 70}\nComparing {len(runs)} runs on questionnaire {questionnaire!r}\n"
          f"Models: {[r.label for r in runs]}\n{'=' * 70}")

    _persona_ids, _item_ids, items, matrices = align_runs(runs)
    out_dir.mkdir(parents=True, exist_ok=True)

    marg = marginal_distribution_table(runs, items, matrices)
    marg.to_csv(out_dir / f"marginals_{questionnaire}.csv", index=False)
    print(f"[Write] {out_dir / f'marginals_{questionnaire}.csv'}")

    agree = paired_agreement_table(runs, items, matrices)
    agree.to_csv(out_dir / f"agreement_{questionnaire}.csv", index=False)
    print(f"[Write] {out_dir / f'agreement_{questionnaire}.csv'}")

    summary = matrix_level_summary(runs, matrices)
    summary.to_csv(out_dir / f"summary_{questionnaire}.csv", index=False)
    print(f"\n[Summary] {questionnaire}")
    print(summary.to_string(index=False))

    plot_per_item_histograms(runs, items, matrices,
                             out_dir / f"hists_{questionnaire}.png")
    plot_agreement_bars(agree, runs,
                        out_dir / f"agreement_bars_{questionnaire}.png")
    plot_persona_mean_scatter(runs, matrices,
                              out_dir / f"persona_scatter_{questionnaire}.png")
    if len(runs) >= 2:
        plot_disagreement_heatmap(runs, items, matrices,
                                  out_dir / f"joint_heatmap_{questionnaire}.png")

    # One multi-panel overview figure per model pair (usually just one pair).
    for ia, ib in itertools.combinations(range(len(runs)), 2):
        la = runs[ia].label.replace("/", "_")
        lb = runs[ib].label.replace("/", "_")
        pair_summary = matrix_level_summary([runs[ia], runs[ib]], [matrices[ia], matrices[ib]])
        plot_summary_overview(
            runs, items, matrices, agree, marg, pair_summary,
            out_dir / f"overview_{questionnaire}_{la}_vs_{lb}.png",
            pair_idx=(ia, ib),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG, help="output subdir name")
    parser.add_argument(
        "--questionnaires", nargs="+", default=QUESTIONNAIRES_TO_COMPARE,
        help="Which questionnaire tags to compare (must match ModelRun.questionnaire)",
    )
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = [load_run(r) for r in RUNS]
    for r in loaded:
        print(f"  [{r.label} / {r.questionnaire}] matrix={r.matrix.shape} "
              f"nan_frac={np.isnan(r.matrix).mean():.4f}")

    for q in args.questionnaires:
        subset = [r for r in loaded if r.questionnaire == q]
        if len(subset) < 2:
            print(f"[Skip] Questionnaire {q!r}: only {len(subset)} model(s) "
                  f"available in RUNS — need ≥2.")
            continue
        run_comparison(subset, q, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()

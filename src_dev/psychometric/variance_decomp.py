"""Variance decomposition of FA factor scores by archetype / scenario.

Given factor scores from a fitted FA on persona-rollouts plus the rollout
metadata that records each persona's interviewer-archetype and scenario-id,
this module reports how much of the per-factor score variance is
attributable to each grouping.

Reported quantities (per factor):
    - One-way η² for ``archetype`` and ``scenario_id`` separately.
    - Two-way ANOVA-style decomposition (archetype + scenario +
      interaction + residual) summing to 1.0. Uses cell/marginal/grand
      means; the partition is informative as a variance share even when
      cells are unbalanced or sparse.
    - Per-archetype mean factor scores, written as CSV + heatmap PNG.

The original implementation lived in
``scripts_dev/unsupervised_embeddings/factor_variance_decomp.py`` (since
deleted — see git history); this module exposes the same logic as a
callable so the v2 paper-analysis pipeline (``run_variance_decomp``) drops
the decomposition into its canonical output tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.interpretation import prompt_effects


__all__ = [
    "build_archetype_scenario_lookup",
    "two_way_eta2",
    "per_archetype_means",
    "plot_factor_means_heatmap",
    "run_variance_decomposition",
]


def build_archetype_scenario_lookup(
    rollout_dir: Path,
    scenarios_file: Path,
) -> dict[str, dict]:
    """Build ``sample_id -> {archetype, scenario_id, row_index}``.

    Resolution path mirrors the rollout pipeline's bookkeeping:
    ``canonical_samples.jsonl`` carries each sample_id's row_index +
    target system prompt; ``archetype_assignments.json`` keys archetype
    by row_index; the scenarios file lets us reverse-lookup the
    scenario_id from the system-prompt text.
    """
    rollout_dir = Path(rollout_dir)
    arch = json.loads((rollout_dir / "archetype_assignments.json").read_text())
    scen_data = json.loads(Path(scenarios_file).read_text())
    prompt_to_scenario: dict[str, str] = {
        sc["target_system_prompt"]: sc["id"] for sc in scen_data["scenarios"]
    }
    out: dict[str, dict] = {}
    canonical_path = rollout_dir / "datasets" / "canonical_samples.jsonl"
    with canonical_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            row_idx = rec["source_info"]["row_index"]
            sys_prompt = rec["input"]["messages"][0]["content"]
            scen_id = prompt_to_scenario.get(sys_prompt)
            archetype = arch.get(str(row_idx))
            if scen_id is None or archetype is None:
                continue
            out[sid] = {
                "archetype": archetype,
                "scenario_id": scen_id,
                "row_index": row_idx,
            }
    return out


def two_way_eta2(
    scores: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> dict:
    """Two-way variance decomposition without statsmodels.

    Returns η² for factor A (archetype), B (scenario), AB interaction,
    plus residual share, summing to ~1.0 per factor. Uses the
    unbalanced-design ANOVA partition based on cell, marginal, and grand
    means — the standard method when cell counts differ.
    """
    out: dict[str, list[float]] = {
        "eta2_archetype": [],
        "eta2_scenario": [],
        "eta2_interaction": [],
        "eta2_residual": [],
        "n_cells": [],
        "n_arch": [],
        "n_scen": [],
        "n_obs": [],
    }
    arch_levels = np.unique(a)
    scen_levels = np.unique(b)
    for f in range(scores.shape[1]):
        y = scores[:, f]
        grand = y.mean()
        ss_total = ((y - grand) ** 2).sum()
        ss_a = 0.0
        for la in arch_levels:
            m = a == la
            if m.sum() == 0:
                continue
            ss_a += m.sum() * (y[m].mean() - grand) ** 2
        ss_b = 0.0
        for lb in scen_levels:
            m = b == lb
            if m.sum() == 0:
                continue
            ss_b += m.sum() * (y[m].mean() - grand) ** 2
        ss_cells = 0.0
        n_cells = 0
        for la in arch_levels:
            for lb in scen_levels:
                m = (a == la) & (b == lb)
                if m.sum() == 0:
                    continue
                n_cells += 1
                ss_cells += m.sum() * (y[m].mean() - grand) ** 2
        ss_inter = max(0.0, ss_cells - ss_a - ss_b)
        ss_resid = max(0.0, ss_total - ss_cells)
        out["eta2_archetype"].append(float(ss_a / ss_total) if ss_total > 0 else 0.0)
        out["eta2_scenario"].append(float(ss_b / ss_total) if ss_total > 0 else 0.0)
        out["eta2_interaction"].append(float(ss_inter / ss_total) if ss_total > 0 else 0.0)
        out["eta2_residual"].append(float(ss_resid / ss_total) if ss_total > 0 else 0.0)
        out["n_cells"].append(int(n_cells))
        out["n_arch"].append(int(len(arch_levels)))
        out["n_scen"].append(int(len(scen_levels)))
        out["n_obs"].append(int(len(y)))
    return out


def per_archetype_means(
    scores: np.ndarray,
    archetypes: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-archetype mean factor scores. Returns (levels, means, counts)."""
    levels = sorted(set(archetypes.tolist()))
    means = np.zeros((len(levels), scores.shape[1]))
    counts = np.zeros(len(levels), dtype=int)
    for i, la in enumerate(levels):
        m = archetypes == la
        means[i] = scores[m].mean(axis=0)
        counts[i] = int(m.sum())
    return levels, means, counts


def plot_factor_means_heatmap(
    levels: list[str],
    means: np.ndarray,
    counts: np.ndarray,
    out_png: Path,
    *,
    title: str,
    factor_labels: list[str] | None = None,
) -> None:
    """Heatmap: archetype × factor mean factor-score, sorted by F0 mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(means[:, 0])
    M = means[order]
    L = [f"{levels[i]}  (n={counts[i]})" for i in order]
    fig, ax = plt.subplots(figsize=(0.95 * M.shape[1] + 4, 0.32 * M.shape[0] + 1.5))
    vmax = float(np.abs(M).max()) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels(L)
    ax.set_xticks(range(M.shape[1]))
    if factor_labels is None:
        factor_labels = [f"F{i}" for i in range(M.shape[1])]
    ax.set_xticklabels(factor_labels, rotation=15, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j, i, f"{M[i, j]:+.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if abs(M[i, j]) < 0.5 * vmax else "white",
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def run_variance_decomposition(
    *,
    scores: np.ndarray,
    metadata: list[dict],
    rollout_dir: Path,
    scenarios_file: Path,
    out_dir: Path,
    model_label: str = "",
    factor_labels: list[str] | None = None,
) -> dict:
    """Run the full archetype/scenario variance decomposition for one model.

    Joins ``scores`` (aligned with ``metadata`` rows by position) against
    a sample_id → archetype/scenario lookup built from ``rollout_dir``,
    drops rows that don't resolve, computes one-way η², two-way η², and
    per-archetype means, and writes:

        ``{out_dir}/eta2_oneway.json``
        ``{out_dir}/eta2_twoway.json``
        ``{out_dir}/per_archetype_factor_means.csv``
        ``{out_dir}/factor_means_by_archetype.{png,pdf}``

    Returns the same payload as a dict for downstream consumers (e.g. a
    cross-model paper figure that aggregates the per-model results).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = build_archetype_scenario_lookup(rollout_dir, scenarios_file)
    if not lookup:
        raise RuntimeError(
            f"No archetype/scenario lookup entries built from {rollout_dir}; "
            "check canonical_samples.jsonl + archetype_assignments.json + "
            f"scenarios at {scenarios_file}."
        )

    enriched_meta: list[dict] = []
    for row in metadata:
        sid = row.get("sample_id")
        hit = lookup.get(sid) if sid is not None else None
        if hit is None:
            enriched_meta.append(dict(row))
            continue
        enriched_meta.append({**row, **hit})

    keep = np.array(
        [("archetype" in r and "scenario_id" in r) for r in enriched_meta],
        dtype=bool,
    )
    if keep.sum() < len(enriched_meta):
        # Common when a few rollout rows fall outside the canonical lookup
        # (e.g. one Qwen context-length outlier dropped earlier in the
        # pipeline). Logged via the returned ``n_unresolved`` field.
        pass
    scores_k = scores[keep]
    meta_k = [r for r, k in zip(enriched_meta, keep) if k]

    eta_arch = prompt_effects(scores_k, meta_k, group_field="archetype").tolist()
    eta_scen = prompt_effects(scores_k, meta_k, group_field="scenario_id").tolist()

    a = np.array([r["archetype"] for r in meta_k])
    b = np.array([r["scenario_id"] for r in meta_k])
    twoway = two_way_eta2(scores_k, a, b)

    levels, means, counts = per_archetype_means(scores_k, a)

    n_factors = int(scores.shape[1])
    oneway_payload = {
        "model": model_label,
        "n_factors": n_factors,
        "n_personas_resolved": int(keep.sum()),
        "n_personas_total": int(len(metadata)),
        "n_unresolved": int(len(metadata) - keep.sum()),
        "n_archetypes": int(len(set(a))),
        "n_scenarios": int(len(set(b))),
        "factor_labels": factor_labels,
        "eta2_archetype_per_factor": eta_arch,
        "eta2_scenario_per_factor": eta_scen,
    }
    (out_dir / "eta2_oneway.json").write_text(json.dumps(oneway_payload, indent=2))

    twoway_payload = {"model": model_label, **twoway, "factor_labels": factor_labels}
    (out_dir / "eta2_twoway.json").write_text(json.dumps(twoway_payload, indent=2))

    with (out_dir / "per_archetype_factor_means.csv").open("w") as fh:
        cols = factor_labels or [f"F{i}" for i in range(n_factors)]
        fh.write("archetype,n," + ",".join(f"{c}_mean" for c in cols) + "\n")
        for la, n_, row in zip(levels, counts, means):
            fh.write(f"{la},{n_}," + ",".join(f"{v:.4f}" for v in row) + "\n")

    plot_factor_means_heatmap(
        levels, means, counts,
        out_dir / "factor_means_by_archetype.png",
        title=f"{model_label}: mean factor score per archetype (sorted by F0)",
        factor_labels=factor_labels,
    )

    return {
        "oneway": oneway_payload,
        "twoway": twoway_payload,
        "per_archetype": {
            "levels": list(levels),
            "counts": counts.tolist(),
            "means": means.tolist(),
        },
    }

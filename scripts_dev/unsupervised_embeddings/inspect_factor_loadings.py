"""One-off inspection: dump full per-factor loadings for the Section 4.2 Llama FA.

Reproduces the paper FA fit on Llama-3.1-8B-Instruct (seed=436, k=4, oblimin,
principal) from the per-questionnaire HF artifacts, then writes
``scratch/factor_inspect/llama_k4_factor_loadings.txt`` — every item in every
factor, sorted by signed loading, so the user can read the *full* picture
when designing constitution prose.

Run: ``uv run python scripts_dev/unsupervised_embeddings/inspect_factor_loadings.py``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

from src_dev.factor_analysis import run_factor_analysis
from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
LLAMA_RUNS = [
    # (run_subdir_on_hf, version_tag_for_combined_items_namespace)
    (
        hf_runs_path(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_v5-likert-direct-lp20"
        ),
        "v5",
    ),
    (
        hf_runs_path(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_trait_ocean_natural_v1-trait_mcq-aside-lp20-"
            "p2-pf2-tmv2"
        ),
        "trait_ocean_natural_v1",
    ),
]
HYDRATE_ROOT = Path("scratch/factor_inspect/hydrated")
OUT_DIR = Path("scratch/factor_inspect")
K = 4
ROTATION = "oblimin"
METHOD = "principal"
MIN_ITEM_VARIANCE = 0.1


def hydrate_one(hf_subdir: str) -> Path:
    """Mirror one ``runs/questionnaire-...`` subtree into scratch."""
    local = HYDRATE_ROOT / Path(hf_subdir).name
    needs_q = (local / "questionnaire" / "response_matrix.npy").exists()
    if not needs_q:
        print(f"hydrate {hf_subdir} -> {local}")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_subdir,
            local_dir=local,
            required=True,
        )
    return local / "questionnaire"


def load_questionnaire(q_dir: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    M = np.load(q_dir / "response_matrix.npy").astype(float)
    meta = [json.loads(l) for l in (q_dir / "metadata.jsonl").read_text().splitlines() if l.strip()]
    items = json.loads((q_dir / "items.json").read_text())
    assert M.shape[0] == len(meta)
    assert M.shape[1] == len(items)
    return M, meta, items


def combine(parts: list[tuple[np.ndarray, list[dict], list[dict], str]]):
    """Align by sample_id intersection across questionnaires; hstack columns.

    Each item in ``parts`` is (M, meta, items, version_tag). Item dicts are
    namespaced with their version_tag in a ``version`` field so the same
    item-text in two questionnaire versions doesn't collide.
    """
    sid_sets = [{m["sample_id"] for m in meta} for _, meta, _, _ in parts]
    common = sorted(set.intersection(*sid_sets))
    print(f"common sample_ids across {len(parts)} questionnaires: {len(common)}")
    sid_to_idx = {sid: i for i, sid in enumerate(common)}

    # Pick metadata from the first part, restricted+ordered to common.
    M0, meta0, _, _ = parts[0]
    common_set = set(common)
    base_meta = [m for m in meta0 if m["sample_id"] in common_set]
    base_meta.sort(key=lambda m: sid_to_idx[m["sample_id"]])

    # Per-part: reorder rows to match common.
    aligned_M: list[np.ndarray] = []
    aligned_items: list[dict] = []
    for M, meta, items, version in parts:
        sid_to_row = {m["sample_id"]: r for r, m in enumerate(meta)}
        order = [sid_to_row[s] for s in common]
        aligned_M.append(M[order])
        for it in items:
            it = dict(it)
            it["version"] = version
            aligned_items.append(it)

    M_full = np.hstack(aligned_M)
    print(f"combined matrix: {M_full.shape}, items: {len(aligned_items)}")
    return M_full, base_meta, aligned_items


def short(text: str, n: int = 120) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def format_item(loading: float, item: dict) -> str:
    block = item.get("block", "?")
    rev = " (REVERSE)" if item.get("reverse_keyed") else ""
    dim = item.get("dimension", "")
    dim_tag = f" [{dim}]" if dim else ""
    text = item.get("text", "<no text>")
    return f"  {loading:+.3f}  <{block}{dim_tag}>{rev}  {short(text)}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for hf_subdir, version in LLAMA_RUNS:
        q_dir = hydrate_one(hf_subdir)
        M, meta, items = load_questionnaire(q_dir)
        print(f"loaded {hf_subdir.split('/')[-1]}: {M.shape} (version={version})")
        parts.append((M, meta, items, version))

    M, meta, items = combine(parts)

    # Drop rows with any NaN.
    nan_rows = np.isnan(M).any(axis=1)
    if nan_rows.any():
        print(f"dropping {int(nan_rows.sum())} rows with NaN")
        M = M[~nan_rows]
        meta = [m for m, n in zip(meta, nan_rows) if not n]

    # Same preprocessing as analysis_for_paper.py: per-block relative-variance
    # floor of 0.1 of the block median.
    Mp, meta_p, items_p, _ = preprocess_response_matrix(
        M, meta, items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=0.0,
        do_residualize=False,
    )
    print(f"preprocessed: {Mp.shape}")

    # Fit FA — deterministic with seed.
    fa = run_factor_analysis(Mp, n_factors=K, method=METHOD, rotation=ROTATION)
    loadings = fa["loadings"]
    prop_var = fa["proportion_variance"]
    fcm = fa.get("factor_correlation_matrix")

    # Save raw fit + items (for downstream use).
    np.savez(
        OUT_DIR / "fa_fit.npz",
        loadings=loadings,
        scores=fa["scores"],
        proportion_variance=prop_var,
        factor_correlation_matrix=fcm if fcm is not None else np.array([]),
    )
    (OUT_DIR / "items.json").write_text(json.dumps(items_p, indent=2))
    print(f"saved fit -> {OUT_DIR/'fa_fit.npz'}")

    # Write the full sorted-loadings dump.
    out = OUT_DIR / "llama_k4_factor_loadings.txt"
    lines: list[str] = []
    lines.append(f"Llama-3.1-8B-Instruct  k={K}  rotation={ROTATION}  method={METHOD}")
    lines.append(f"n_personas={Mp.shape[0]}  n_items={Mp.shape[1]}  seed={SEED}")
    lines.append("Variance explained per factor:")
    for fi in range(K):
        lines.append(f"  F{fi}: prop={prop_var[fi]:.3f}")
    if fcm is not None and getattr(fcm, "shape", None) == (K, K):
        lines.append("")
        lines.append("Factor correlation matrix (oblique):")
        for i in range(K):
            row = "  ".join(f"{fcm[i, j]:+.3f}" for j in range(K))
            lines.append(f"  F{i}   {row}")
    lines.append("")

    for fi in range(K):
        lines.append("=" * 100)
        lines.append(f"F{fi}  —  ALL items sorted by signed loading (positive first)")
        lines.append("=" * 100)
        order = np.argsort(-loadings[:, fi])  # descending
        for idx in order:
            it = items_p[idx]
            line = format_item(float(loadings[idx, fi]), it)
            lines.append(line)
            # MCQ: also include option-text under the item.
            if it.get("block") == "trait_mcq":
                opts = it.get("options") or {}
                am = it.get("answer_mapping") or {}
                if opts:
                    for letter in ["A", "B", "C", "D"]:
                        opt_text = opts.get(letter, "")
                        score = am.get(letter)
                        score_tag = ""
                        if isinstance(score, (int, float)):
                            score_tag = " [HIGH-pole]" if score >= 0.66 else (
                                " [LOW-pole]" if score <= 0.34 else " [MID]"
                            )
                        lines.append(f"      {letter}{score_tag}  {short(opt_text, 140)}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"wrote {out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()

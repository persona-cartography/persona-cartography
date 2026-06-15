"""Rescue script — regenerate ``factor_extremes.html`` for an already-computed
FA run that was missing its rollout exports at the time the main pipeline ran.

``fa_full_stages_filtered.py`` writes HTML via
:func:`src_dev.psychometric.factor_extremes_html.export_factor_extremes_html`,
which silently skips when ``{rollout_dir}/exports/conversation_training.jsonl``
is not present on disk. When the rollouts only live on HF (not local scratch),
the FA stage runs successfully but no HTML is produced.

This script:

1. Hydrates the preset-B rollout's ``exports/`` subtree from the
   ``persona-shattering-lasr/psychometric-fa-runs`` dataset repo into the
   expected local path.
2. Loads each saved FA variant (``raw_oblimin``, ``raw_varimax``, the
   per-block variants) from disk via
   :func:`src_dev.factor_analysis.persistence.load_factor_analysis`.
3. Calls ``export_factor_extremes_html`` for each, using the existing
   ``labeling/`` item-label cache for fallback factor descriptions.

Nothing is recomputed — the FA fits and validations from the original run
are reused verbatim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np

from src_dev.factor_analysis.persistence import load_factor_analysis
from src_dev.psychometric.factor_extremes_html import export_factor_extremes_html
from src_dev.psychometric.fa_run_catalogue import FA_RUN_REGISTRY
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import login_from_env

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Targets — mirror the paths fa_full_stages_filtered.py uses.
# ═════════════════════════════════════════════════════════════════════════════

RUN_DIR = Path(
    "scratch/psychometric_fa/"
    "filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4"
)

# Source rollout run for preset B (the combined matrix was built from this).
_ROLLOUT_RUN = FA_RUN_REGISTRY["llama_base_rollouts"]
ROLLOUT_DIR = _ROLLOUT_RUN.scratch_dir()

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
HF_ROLLOUT_PATH = _ROLLOUT_RUN.hf_path

# FA knobs from the original run — used to locate the saved .npz files and
# to reproduce the preprocess filter that yielded the loadings/scores shape.
N_FACTORS = 4
METHOD = "principal"
ROTATIONS = ("oblimin", "varimax")
MIN_ITEM_VARIANCE = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Steps
# ═════════════════════════════════════════════════════════════════════════════


def _ensure_rollout_export(rollout_dir: Path) -> bool:
    """Hydrate ``rollout_dir/exports/`` from HF if the conversation export is missing."""
    export_path = rollout_dir / "exports" / "conversation_training.jsonl"
    if export_path.exists():
        print(f"[Hydrate] Rollout export already local: {export_path}")
        return True

    try:
        login_from_env()
    except RuntimeError as e:
        print(f"[Hydrate] {e} — cannot pull rollouts from HF.")
        return False

    print(f"[Hydrate] Pulling {HF_REPO_ID}:{HF_ROLLOUT_PATH}/exports → {rollout_dir}/exports")
    got = hydrate_dataset_subtree(
        repo_id=HF_REPO_ID,
        path_in_repo=f"{HF_ROLLOUT_PATH}/exports",
        local_dir=rollout_dir / "exports",
        required=False,
    )
    if not got:
        print(f"[Hydrate] FAILED — no files under {HF_REPO_ID}:{HF_ROLLOUT_PATH}/exports")
        return False

    if not export_path.exists():
        print(f"[Hydrate] FAILED — expected file not materialized: {export_path}")
        return False

    print(f"[Hydrate] OK — {export_path}")
    return True


def _load_questionnaire(run_dir: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    q = run_dir / "questionnaire"
    response_matrix = np.load(q / "response_matrix.npy")
    with open(q / "items.json") as f:
        items = json.load(f)
    with open(q / "metadata.jsonl") as f:
        meta = [json.loads(line) for line in f if line.strip()]
    print(
        f"[Load]  response_matrix={response_matrix.shape}, "
        f"items={len(items)}, metadata rows={len(meta)}"
    )
    return response_matrix, items, meta


def _apply_preprocess(
    response_matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Re-run the same row+col preprocess filter the FA stage applied.

    Returns ``(cols_filtered, meta_filtered)`` aligned with the loadings /
    scores arrays in the saved .npz files.
    """
    _data, meta_f, cols_f, _ = preprocess_response_matrix(
        response_matrix,
        metadata,
        items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=False,
    )
    print(f"[Preprocess] cols {len(items)}→{len(cols_f)}, rows {len(metadata)}→{len(meta_f)}")
    return cols_f, meta_f


def _regen_for_variant(
    *,
    npz_base: Path,
    save_dir: Path,
    label: str,
    column_defs: list[dict],
    metadata: list[dict],
    labeling_dir: Path,
    rollout_dirs: list[Path],
) -> None:
    """Load one FA npz and call the HTML exporter."""
    if not Path(str(npz_base) + ".npz").exists():
        print(f"[Skip] {label}: no npz at {npz_base}.npz")
        return
    fa_result = load_factor_analysis(npz_base)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HTML] {label} → {save_dir}/factor_extremes.html")
    export_factor_extremes_html(
        fa_result=fa_result,
        column_defs=column_defs,
        metadata=metadata,
        label=label,
        save_dir=save_dir,
        rollout_dirs=rollout_dirs,
        labeling_dir=labeling_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Expected FA run dir not found: {RUN_DIR}")

    if not _ensure_rollout_export(ROLLOUT_DIR):
        raise SystemExit(
            "Cannot regenerate HTML without rollout exports. See messages above."
        )

    response_matrix, items, metadata = _load_questionnaire(RUN_DIR)
    # Canonical labels location per FA pipeline convention: sibling of
    # ``questionnaire/`` under the run root. Matches
    # ``cfg.ctx.effective_questionnaire_dir / "labeling"`` used by
    # src_dev/psychometric/stages/{labeling,factor_analysis}.py.
    labeling_dir = RUN_DIR / "labeling"
    fa_root = RUN_DIR / "factor_analysis"

    # ── raw_{rotation} variants (full item set) ─────────────────────────────
    print("\n[Preprocess] raw (full item set)")
    cols_filtered, meta_filtered = _apply_preprocess(response_matrix, metadata, items)
    for rot in ROTATIONS:
        _regen_for_variant(
            npz_base=fa_root / "raw" / f"fa_{N_FACTORS}_{METHOD}_{rot}",
            save_dir=fa_root / f"raw_{rot}",
            label=f"raw_{rot}",
            column_defs=cols_filtered,
            metadata=meta_filtered,
            labeling_dir=labeling_dir,
            rollout_dirs=[ROLLOUT_DIR],
        )

    # ── per_block variants — subset item set per block ──────────────────────
    per_block_root = fa_root / "per_block"
    if per_block_root.exists():
        for block_dir in sorted(p for p in per_block_root.iterdir() if p.is_dir()):
            block = block_dir.name
            keep_idx = [i for i, c in enumerate(items) if str(c.get("block", "")) == block]
            if not keep_idx:
                print(f"[Skip] per_block/{block}: no matching items in items.json")
                continue
            sub_matrix = response_matrix[:, keep_idx]
            sub_items = [items[i] for i in keep_idx]
            print(f"\n[Preprocess] per_block/{block}")
            sub_cols_filtered, sub_meta_filtered = _apply_preprocess(
                sub_matrix, metadata, sub_items,
            )
            for rot in ROTATIONS:
                _regen_for_variant(
                    npz_base=block_dir / "raw" / f"fa_{N_FACTORS}_{METHOD}_{rot}",
                    save_dir=block_dir / f"raw_{rot}",
                    label=f"block_{block}_raw_{rot}",
                    column_defs=sub_cols_filtered,
                    metadata=sub_meta_filtered,
                    labeling_dir=labeling_dir,
                    rollout_dirs=[ROLLOUT_DIR],
                )

    print("\nDone.")


if __name__ == "__main__":
    main()

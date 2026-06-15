"""Validate a trained unsup_k4_v7_pf3 LoRA by re-administering the v7 fc_pair
questionnaire on a 200-persona subsample.

Companion to scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py, adapted to
the k=4 oblimin solution on the v7 fc_pair questionnaire (factors:
F0_Initiative, F1_Pedagogy, F2_Warmth, F3_Hedging) instead of the v5 +
trait_ocean_natural_v1 combined questionnaire (factors: Conviction,
Exuberance, Warmth, Didacticism).

The expectation is that the target factor's score moves in the direction
the LoRA was trained for (up for amplifier, down for suppressor), while
the other three factors shift much less. The script reports per-factor
paired diffs for all four factors regardless of ``--target``; ``--target``
only controls which row is highlighted in the console output and the
default label.

Usage::

    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/validate_lora.py \\
        --target initiative \\
        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/initiative/amplifier/vunsup_k4_v7_pf3_paired_dpo/lora/<adapter-subfolder> \\
        --n-personas 200 \\
        --label initiative_amp

Pre-reqs:
    - The HF dataset ``persona-shattering-lasr/psychometric-fa-runs`` is
      reachable; the script will hydrate the rollout dir and the v7
      fc_pair questionnaire dir on first run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
from itertools import permutations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import torch  # noqa: E402

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from factor_analyzer import FactorAnalyzer  # noqa: E402

from src_dev.psychometric.combine import load_pair_outputs  # noqa: E402
from src_dev.psychometric.config import QuestionnaireStageConfig  # noqa: E402
from src_dev.psychometric.fa_run_catalogue import FA_RUN_REGISTRY  # noqa: E402
from src_dev.psychometric.preprocessing import preprocess_response_matrix  # noqa: E402
from src_dev.psychometric.questionnaire_inference import (  # noqa: E402
    run_questionnaire_inference_async,
)
from src_dev.psychometric.questionnaire_io import load_questionnaire  # noqa: E402
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
_HYDRATE_ROOT = Path("scratch/factor_inspect_v7_pf3/hydrated")
_ROLLOUT_RUN = FA_RUN_REGISTRY["llama_base_rollouts"]
_V7_FC_RUN = FA_RUN_REGISTRY["llama_q_v7_fc_pair_pf3"]

ROLLOUT_HF_PATH = _ROLLOUT_RUN.hf_path
ROLLOUT_LOCAL = _ROLLOUT_RUN.scratch_dir(_HYDRATE_ROOT)

V7_FC_HF_PATH = _V7_FC_RUN.hf_path
V7_FC_LOCAL = _V7_FC_RUN.scratch_dir(_HYDRATE_ROOT)

V7_FC_QUESTIONNAIRE = Path(
    "datasets/psychometric_questionnaires/psychometric_questionnaire_v7_fc_pair.json"
)

QUESTIONNAIRE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NUM_CONVERSATION_TURNS = 15  # match the B preset used in the FA fit
QUESTIONNAIRE_VERSION = "v7_fc_pair"

VALIDATE_ROOT = Path("scratch/factor_inspect_v7_pf3/validate")


# ── Subsampling the rollout dir ────────────────────────────────────────────


def _filter_jsonl(src: Path, dst: Path, kept_ids: set[str]) -> int:
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid is None or sid in kept_ids:
                fout.write(line)
                n += 1
    return n


def write_subsample_rollout(src_dir: Path, dst_dir: Path, kept_ids: set[str]) -> Path:
    """Mirror ``src_dir`` at ``dst_dir`` keeping only rows whose sample_id is in
    ``kept_ids``. Filters every .jsonl that has a ``sample_id`` field; copies
    everything else verbatim. Identical to the unsup_4fac implementation.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for child in src_dir.iterdir():
        if child.is_file():
            shutil.copy(child, dst_dir / child.name)

    src_ds = src_dir / "datasets"
    dst_ds = dst_dir / "datasets"
    if src_ds.exists():
        dst_ds.mkdir(exist_ok=True)
        for child in src_ds.iterdir():
            if child.suffix == ".jsonl":
                _filter_jsonl(child, dst_ds / child.name, kept_ids)
            else:
                shutil.copy(child, dst_ds / child.name)

    src_ev = src_dir / "events"
    dst_ev = dst_dir / "events"
    if src_ev.exists():
        dst_ev.mkdir(exist_ok=True)
        for child in src_ev.iterdir():
            if child.suffix == ".jsonl":
                _filter_jsonl(child, dst_ev / child.name, kept_ids)
            else:
                shutil.copy(child, dst_ev / child.name)

    for child in src_dir.iterdir():
        if child.is_dir() and child.name not in ("datasets", "events"):
            shutil.copytree(child, dst_dir / child.name, dirs_exist_ok=True)

    return dst_dir


# ── Hydrate everything we need from HF (one-time setup) ────────────────────


def hydrate_inputs() -> None:
    """Pull rollout dir + v7 fc_pair questionnaire dir from HF if missing."""
    pairs = [
        (ROLLOUT_HF_PATH, ROLLOUT_LOCAL),
        (V7_FC_HF_PATH, V7_FC_LOCAL),
    ]
    for hf_path, local in pairs:
        if not local.exists() or not any(local.rglob("*.jsonl")):
            print(f"hydrate {hf_path}")
            hydrate_dataset_subtree(
                repo_id=HF_REPO_ID,
                path_in_repo=hf_path,
                local_dir=local,
                required=True,
            )


# ── Sample selection ──────────────────────────────────────────────────────


def stratified_sample_ids(rollout_dir: Path, n: int, seed: int) -> list[str]:
    """Pick ``n`` sample_ids stratified by archetype; identical to unsup_4fac."""
    canon = rollout_dir / "datasets" / "canonical_samples.jsonl"
    rows: list[dict] = []
    with canon.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    print(f"[validate] rollout has {len(rows)} samples")

    rng = random.Random(seed)

    arch_path = rollout_dir / "archetype_assignments.json"
    if not arch_path.exists():
        ids = [r["sample_id"] for r in rows]
        return rng.sample(ids, n)

    arch_assign = json.loads(arch_path.read_text())
    by_arch: dict[str, list[str]] = {}
    for r in rows:
        sid = r["sample_id"]
        row_idx = r["source_info"]["row_index"]
        a = arch_assign.get(str(row_idx), "_unknown")
        by_arch.setdefault(a, []).append(sid)

    n_archs = len(by_arch)
    per_arch_floor = max(1, n // n_archs - 1)
    chosen: list[str] = []
    for a, ids in by_arch.items():
        take = min(len(ids), max(per_arch_floor, int(round(n * len(ids) / len(rows)))))
        chosen.extend(rng.sample(ids, take))
    if len(chosen) < n:
        rest = [r["sample_id"] for r in rows if r["sample_id"] not in set(chosen)]
        chosen.extend(rng.sample(rest, n - len(chosen)))
    elif len(chosen) > n:
        chosen = rng.sample(chosen, n)

    print(f"[validate] sampled {len(chosen)} personas (target={n}) "
          f"across {n_archs} archetypes")
    return chosen


# ── Questionnaire admin (LoRA-aware) ───────────────────────────────────────


async def admin_v7_fc_pair(
    *,
    rollout_dir: Path,
    adapter_path: str | None,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Administer the v7 fc_pair questionnaire on the subsample with
    optional LoRA. Returns ``(matrix, items, metadata)``.
    """
    items, column_defs = load_questionnaire(V7_FC_QUESTIONNAIRE, fa_blocks=("fc_pair",))

    cfg = QuestionnaireStageConfig(
        ctx=None,  # not used by run_questionnaire_inference_async
        questionnaire_path=V7_FC_QUESTIONNAIRE,
        questionnaire_version=QUESTIONNAIRE_VERSION,
        fa_blocks=("fc_pair",),
        use_logprobs=True,
        phrasing="direct",
        provider="vllm",
        model=QUESTIONNAIRE_MODEL,
        adapter_path=adapter_path,
        max_new_tokens=32,
        max_concurrent=32,
        timeout=120,
        vllm_personas_per_batch=8,
        vllm_gpu_memory_utilization=0.92,
        top_logprobs=20,
    )

    matrix, metadata = await run_questionnaire_inference_async(
        cfg,
        rollout_dir=rollout_dir,
        items=items,
        column_defs=column_defs,
        output_dir=output_dir,
        num_conversation_turns=NUM_CONVERSATION_TURNS,
    )
    return matrix, list(column_defs), metadata


# ── Match new responses to FA-fit columns ──────────────────────────────────


def align_to_fit_items(
    M_new: np.ndarray,
    items_new: list[dict],
    fit_items: list[dict],
) -> np.ndarray:
    """Slice ``M_new``'s columns to match ``fit_items`` order, by ``col_id``.

    Columns not present in ``fit_items`` are dropped. Columns missing from
    ``items_new`` raise — we can't score factors on partially-administered
    items.
    """
    new_index: dict[str, int] = {it["col_id"]: i for i, it in enumerate(items_new)}
    cols: list[int] = []
    missing: list[str] = []
    for it in fit_items:
        cid = it["col_id"]
        if cid not in new_index:
            missing.append(cid)
        else:
            cols.append(new_index[cid])
    if missing:
        raise RuntimeError(
            f"Subsample run is missing {len(missing)} items present in the "
            f"FA fit (e.g. {missing[:3]}). Cannot score factors."
        )
    print(f"[validate] aligned {len(cols)} columns to FA-fit items")
    return M_new[:, cols]


# ── Factor identity anchors ───────────────────────────────────────────────
#
# Anchor each canonical factor to a small set of high-loading items whose
# signed loadings we read off the k=4 v7_pf3 oblimin fit (see
# scratch/psychometric_fa.pf3-k4/.../fa_4_principal_oblimin.npz). After
# refit, find the unique permutation + sign-flip that maps refit factors
# to canonical ones, and abort if no clean mapping exists.
#
# Anchor format: {canonical_factor_index: [(col_id, expected_sign), ...]}.

EXPECTED_FACTOR_ANCHORS: dict[int, list[tuple[str, int]]] = {
    0: [  # F0_Initiative — proactive volunteering / position-taking
        ("v7fc_030", +1),  # handle edge case anyway          load=+0.79
        ("v7fc_032", +1),  # answer pre-emptively              load=+0.75
        ("v7fc_031", +1),  # volunteer info                    load=+0.71
        ("v7fc_048", -1),  # name what they actually need      load=-0.72 (reversed pole)
    ],
    1: [  # F1_Pedagogy — show working / formal / structured / protective
        ("v7fc_037", +1),  # show working on math/logic        load=+0.73
        ("v7fc_046", -1),  # mention efficient alternative     load=-0.72 (reversed pole)
        ("v7fc_015", +1),  # comparison as list/table          load=+0.66
        ("v7fc_012", -1),  # validate before disagreeing       load=-0.63 (reversed pole)
    ],
    2: [  # F2_Warmth — playful / register-mirroring / accommodating
        ("v7fc_052", +1),  # self-id as playful                load=+0.83
        ("v7fc_051", +1),  # value wit/playfulness             load=+0.83
        ("v7fc_008", +1),  # cushion feedback                  load=+0.77
        ("v7fc_024", -1),  # echo emoji/playful punctuation    load=-0.72 (reversed pole)
    ],
    3: [  # F3_Hedging — uncertainty-flagging / non-committal / yielding
        ("v7fc_060", +1),  # flag uncertainty inline           load=+0.78
        ("v7fc_020", +1),  # flag knowledge thin               load=+0.69
        ("v7fc_057", +1),  # narrate reasoning step-by-step    load=+0.62
        ("v7fc_061", -1),  # lay out considerations            load=-0.54 (reversed pole)
    ],
}
CANONICAL_FACTOR_NAMES = ["F0_Initiative", "F1_Pedagogy", "F2_Warmth", "F3_Hedging"]

# Map ``--target`` value (lower-case factor name) to the canonical factor
# index. Used by report_comparison to highlight the target row.
TARGET_FACTOR_INDEX: dict[str, int] = {
    "initiative": 0,
    "pedagogy":   1,
    "warmth":     2,
    "hedging":    3,
}


def _find_anchor_rows(items: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Locate each anchor in ``items`` by ``col_id``. Returns
    ``{factor: [(row_idx, sign), ...]}``.
    """
    out: dict[int, list[tuple[int, int]]] = {}
    cid_to_row = {it["col_id"]: i for i, it in enumerate(items)}
    for fi, anchors in EXPECTED_FACTOR_ANCHORS.items():
        rows: list[tuple[int, int]] = []
        for cid, sign in anchors:
            if cid not in cid_to_row:
                raise RuntimeError(
                    f"Factor F{fi} anchor col_id {cid!r} not found in items "
                    "(expected in the preprocessed FA fit). The questionnaire "
                    "format may have changed; update EXPECTED_FACTOR_ANCHORS."
                )
            rows.append((cid_to_row[cid], sign))
        out[fi] = rows
    return out


def align_factor_identity(
    loadings: np.ndarray,
    items: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Find the permutation + sign-flip mapping refit factors to canonical
    ones. Brute-forces all 4! = 24 permutations, picks the one that
    maximises agreement with the per-factor anchors, then verifies every
    anchor is on the right sign in the chosen mapping. Aborts with a
    descriptive error if any anchor is on the wrong sign.

    Returns ``(perm, signs)`` such that
    ``loadings[:, perm] * signs[None, :]`` is the canonical-order,
    canonical-sign loading matrix.
    """
    k = loadings.shape[1]
    assert k == 4, f"expected 4 factors, got {k}"

    anchor_rows = _find_anchor_rows(items)

    # Per (canonical_i, refit_j): mean over i's anchors of expected_sign·sign(load).
    # Loadings with magnitude < 0.1 contribute 0 (treated as ambiguous).
    score = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            agreements: list[float] = []
            for row, expected_sign in anchor_rows[i]:
                load = loadings[row, j]
                if abs(load) < 0.1:
                    agreements.append(0.0)
                else:
                    agreements.append(float(expected_sign * np.sign(load)))
            score[i, j] = float(np.mean(agreements))

    abs_score = np.abs(score)
    best_perm: tuple[int, ...] | None = None
    best_sum = -np.inf
    for perm in permutations(range(k)):
        s = float(sum(abs_score[i, perm[i]] for i in range(k)))
        if s > best_sum:
            best_sum = s
            best_perm = perm
    assert best_perm is not None
    perm_arr = np.array(best_perm)
    signs = np.array(
        [int(np.sign(score[i, perm_arr[i]])) or 1 for i in range(k)]
    )

    # Verify alignment against every anchor.
    failures: list[str] = []
    for i in range(k):
        for row, expected_sign in anchor_rows[i]:
            aligned_load = float(loadings[row, perm_arr[i]] * signs[i])
            if expected_sign * np.sign(aligned_load) <= 0:
                failures.append(
                    f"  F{i} ({CANONICAL_FACTOR_NAMES[i]}) anchor "
                    f"col_id={items[row]['col_id']}: "
                    f"aligned_loading={aligned_load:+.3f} but expected sign "
                    f"{expected_sign:+d}"
                )
    if failures:
        raise RuntimeError(
            "Factor identity check FAILED — refit FA does not match the "
            f"k=4 v7_pf3 oblimin canonical factors.\n"
            f"Best permutation: {perm_arr.tolist()}, signs: {signs.tolist()}\n"
            "Anchor mismatches:\n" + "\n".join(failures)
        )

    if not (perm_arr.tolist() == [0, 1, 2, 3] and signs.tolist() == [1, 1, 1, 1]):
        print(
            f"[validate] WARNING: refit factors required reordering. "
            f"permutation={perm_arr.tolist()} signs={signs.tolist()}. "
            "Results still valid after the alignment, but worth investigating."
        )
    else:
        print("[validate] factor identity check passed (canonical order preserved).")

    return perm_arr, signs


# ── FA refit + scoring ────────────────────────────────────────────────────


def refit_fa_for_scoring() -> tuple[
    FactorAnalyzer, np.ndarray, list[dict], list[str], np.ndarray, np.ndarray
]:
    """Refit the v7 fc_pair k=4 oblimin FA on the cached baseline matrix.

    Deterministic with seed=436. Returns the fitted ``FactorAnalyzer``, the
    preprocessed response matrix, the aligned column items, the per-row
    sample_ids, and the canonical-identity ``(perm, signs)`` for
    fa.transform outputs.
    """
    M, meta, items = load_pair_outputs(V7_FC_LOCAL)
    M_pp, meta_pp, items_pp, _ = preprocess_response_matrix(
        M,
        meta,
        items,
        min_item_variance=0.1,
        high_variance_persona_drop_pct=0.0,
        do_residualize=False,
    )
    fa = FactorAnalyzer(n_factors=4, method="principal", rotation="oblimin")
    fa.fit(M_pp)
    perm, signs = align_factor_identity(fa.loadings_, items_pp)
    return fa, M_pp, items_pp, [m["sample_id"] for m in meta_pp], perm, signs


def aligned_transform(
    fa: FactorAnalyzer, M: np.ndarray, perm: np.ndarray, signs: np.ndarray
) -> np.ndarray:
    """``fa.transform(M)`` reordered + sign-flipped to canonical (F0..F3)."""
    return fa.transform(M)[:, perm] * signs[None, :]


# ── Item-level shifts ─────────────────────────────────────────────────────


def report_item_level_shifts(
    *,
    M_lora: np.ndarray,
    M_baseline: np.ndarray,
    loadings_aligned: np.ndarray,
    items_pp: list[dict],
    threshold: float = 0.4,
) -> list[dict]:
    """Per-factor: pole-aligned shift on items where this factor is dominant.

    Identical methodology to unsup_4fac/validate_lora.py — see that script's
    docstring on this function for rationale.
    """
    delta_mean = M_lora.mean(0) - M_baseline.mean(0)
    dominant = np.abs(loadings_aligned).argmax(axis=1)
    rows: list[dict] = []
    for f in range(loadings_aligned.shape[1]):
        load_f = loadings_aligned[:, f]
        mask = (dominant == f) & (np.abs(load_f) >= threshold)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "factor": CANONICAL_FACTOR_NAMES[f],
                "n_dominant_items_loading_ge_thresh": 0,
                "threshold": threshold,
                "pole_aligned_mean_shift": None,
                "pole_aligned_median_shift": None,
                "n_items_moving_to_high_pole": None,
                "n_items_moving_to_low_pole": None,
            })
            continue
        signed = np.sign(load_f[mask]) * delta_mean[mask]
        rows.append({
            "factor": CANONICAL_FACTOR_NAMES[f],
            "n_dominant_items_loading_ge_thresh": n,
            "threshold": threshold,
            "pole_aligned_mean_shift": float(signed.mean()),
            "pole_aligned_median_shift": float(np.median(signed)),
            "pole_aligned_shift_std": float(signed.std(ddof=1)) if n > 1 else None,
            "n_items_moving_to_high_pole": int((signed > 0).sum()),
            "n_items_moving_to_low_pole": int((signed < 0).sum()),
        })
    return rows


# ── Comparison + reporting ────────────────────────────────────────────────


def report_comparison(
    *,
    sample_ids: list[str],
    lora_scores: np.ndarray,
    baseline_scores: np.ndarray,
    item_level_summary: list[dict],
    adapter: str,
    factor_identity: dict,
    label: str,
    output_dir: Path,
    target_factor_index: int | None = None,
) -> dict:
    """Per-factor paired comparison + JSON dump + violin plot."""
    diff = lora_scores - baseline_scores
    k = lora_scores.shape[1]
    factor_names = CANONICAL_FACTOR_NAMES[:k]
    summary: dict = {
        "label": label,
        "adapter": adapter,
        "n_personas": len(sample_ids),
        "factor_identity": factor_identity,
        "target_factor_index": target_factor_index,
        "target_factor": (
            factor_names[target_factor_index] if target_factor_index is not None else None
        ),
        "factor_summary": [],
        "item_level_summary": item_level_summary,
    }
    for i in range(k):
        d = diff[:, i]
        baseline_d = baseline_scores[:, i]
        lora_d = lora_scores[:, i]
        rng = np.random.default_rng(SEED)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        dz = float(d.mean() / max(d.std(ddof=1), 1e-9))
        summary["factor_summary"].append({
            "factor": factor_names[i],
            "mean_baseline": float(baseline_d.mean()),
            "mean_lora": float(lora_d.mean()),
            "mean_diff": float(d.mean()),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
            "cohen_dz": dz,
            "n_pos_diffs": int((d > 0).sum()),
            "n_neg_diffs": int((d < 0).sum()),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2))

    np.savez(
        output_dir / f"{label}_scores.npz",
        sample_ids=np.array(sample_ids),
        lora_scores=lora_scores,
        baseline_scores=baseline_scores,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.violinplot([diff[:, i] for i in range(k)], showmeans=True, showmedians=True)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xticks(range(1, k + 1))
        ax.set_xticklabels(factor_names, rotation=15)
        ax.set_ylabel("F̂(LoRA) − F̂(baseline) per persona")
        ax.set_title(f"{label}: paired factor-score change (n={len(sample_ids)})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_paired_diff.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"[validate] plot skipped: {e}")

    print()
    print(f"=== {label}  (n={len(sample_ids)})  adapter={adapter} ===")
    if target_factor_index is not None:
        print(f"target factor: {factor_names[target_factor_index]}  (arrow ▶ in tables below)")
    print("Factor-score Δ (fa.transform; σ-units inflated on OOD responses):")
    for i, row in enumerate(summary["factor_summary"]):
        marker = "▶ " if i == target_factor_index else "  "
        print(
            f"{marker}{row['factor']:18s}  "
            f"baseline={row['mean_baseline']:+.3f}  lora={row['mean_lora']:+.3f}  "
            f"Δ={row['mean_diff']:+.3f}  "
            f"95% CI [{row['ci_95_lo']:+.3f}, {row['ci_95_hi']:+.3f}]  "
            f"dz={row['cohen_dz']:+.2f}"
        )
    print(
        "Item-level pole-aligned shift on |loading|≥0.4 dominant items "
        "(raw response scale; positive ⇒ moved toward HIGH pole):"
    )
    for i, row in enumerate(summary["item_level_summary"]):
        marker = "▶ " if i == target_factor_index else "  "
        if row["n_dominant_items_loading_ge_thresh"] == 0:
            print(f"{marker}{row['factor']:18s}  (no dominant items at threshold)")
            continue
        print(
            f"{marker}{row['factor']:18s}  "
            f"n={row['n_dominant_items_loading_ge_thresh']:3d}  "
            f"mean_shift={row['pole_aligned_mean_shift']:+.3f}  "
            f"median={row['pole_aligned_median_shift']:+.3f}  "
            f"toward HIGH={row['n_items_moving_to_high_pole']}/"
            f"{row['n_items_moving_to_high_pole'] + row['n_items_moving_to_low_pole']}"
        )
    return summary


# ── Driver ────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    hydrate_inputs()

    # 1. Stratified subsample.
    sample_ids = stratified_sample_ids(ROLLOUT_LOCAL, n=args.n_personas, seed=SEED)
    sample_set = set(sample_ids)
    label = args.label

    # 2. Mirror the rollout dir filtered to those sample_ids.
    sub_rollout = VALIDATE_ROOT / label / "rollout_subsample"
    write_subsample_rollout(ROLLOUT_LOCAL, sub_rollout, sample_set)

    # 3. Run v7 fc_pair questionnaire admin with the LoRA.
    out_v7 = VALIDATE_ROOT / label / "questionnaire_v7_fc_pair"
    print("[validate] running v7 fc_pair questionnaire admin")
    M_lora, items_lora, meta_lora = await admin_v7_fc_pair(
        rollout_dir=sub_rollout,
        adapter_path=args.adapter,
        output_dir=out_v7,
    )

    # 4. Order LoRA matrix by sample_id, drop NaN-row personas.
    sid_lora = {m["sample_id"]: i for i, m in enumerate(meta_lora)}
    common = [s for s in sample_ids if s in sid_lora]
    print(f"[validate] {len(common)}/{len(sample_ids)} personas with v7 fc_pair complete")
    M_lora = M_lora[[sid_lora[s] for s in common]]

    keep = ~np.isnan(M_lora).any(axis=1)
    if not keep.all():
        print(f"[validate] dropping {(~keep).sum()} personas with NaNs in response matrix")
    M_lora = M_lora[keep]
    common = [s for s, k in zip(common, keep) if k]

    # 5. Refit FA on full baseline matrix; canonicalise factor order/sign.
    print("[validate] refitting FA on baseline matrix for scoring...")
    fa, M_baseline_full, fit_items, baseline_sids, perm, signs = refit_fa_for_scoring()
    factor_identity = {"permutation": perm.tolist(), "signs": signs.tolist()}
    loadings_aligned = fa.loadings_[:, perm] * signs[None, :]

    M_lora_aligned = align_to_fit_items(M_lora, items_lora, fit_items)

    # 6. Score baseline subset for the same personas (paired comparison).
    sid_to_baseline_row = {s: i for i, s in enumerate(baseline_sids)}
    rows = [sid_to_baseline_row[s] for s in common if s in sid_to_baseline_row]
    common_with_baseline = [s for s in common if s in sid_to_baseline_row]
    if len(rows) != len(common):
        print(
            f"[validate] {len(common) - len(rows)} subsample personas missing "
            "from baseline FA (preprocessing dropped them)"
        )
    baseline_scores = aligned_transform(fa, M_baseline_full[rows], perm, signs)

    lora_scores = aligned_transform(fa, M_lora_aligned, perm, signs)
    keep_idx = [i for i, s in enumerate(common) if s in sid_to_baseline_row]
    lora_scores = lora_scores[keep_idx]
    M_lora_paired = M_lora_aligned[keep_idx]
    M_baseline_paired = M_baseline_full[rows]

    # 7. Item-level shifts on raw response scale.
    item_level_summary = report_item_level_shifts(
        M_lora=M_lora_paired,
        M_baseline=M_baseline_paired,
        loadings_aligned=loadings_aligned,
        items_pp=fit_items,
        threshold=0.4,
    )

    # 8. Compare and report.
    out_dir = VALIDATE_ROOT / label
    report_comparison(
        sample_ids=common_with_baseline,
        lora_scores=lora_scores,
        baseline_scores=baseline_scores,
        item_level_summary=item_level_summary,
        adapter=args.adapter,
        factor_identity=factor_identity,
        label=label,
        output_dir=out_dir,
        target_factor_index=TARGET_FACTOR_INDEX[args.target],
    )

    # 9. Optional: push the eval folder to the monorepo.
    if args.upload_monorepo:
        from src_dev.utils import upload_folder_to_dataset_repo

        path_in_repo = (
            f"fine_tuning/llama-3.1-8b-it/unsupervised/{args.target}/"
            f"{args.direction}/v{args.monorepo_version}/evals/"
            f"factor_validate/{label}"
        )
        url = upload_folder_to_dataset_repo(
            local_dir=out_dir,
            repo_id="persona-shattering-lasr/monorepo",
            path_in_repo=path_in_repo,
            commit_message=(
                f"Add {label} factor-validate results for {args.target} "
                f"{args.direction} v{args.monorepo_version}."
            ),
            ignore_patterns=["rollout_subsample/**"],
        )
        print(f"[validate] uploaded results to {url}/tree/main/{path_in_repo}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--target",
        required=True,
        choices=sorted(TARGET_FACTOR_INDEX),
        help=(
            "Factor the LoRA was trained along — used to highlight the target "
            "row in the report. The script reports all four factors regardless."
        ),
    )
    ap.add_argument(
        "--adapter",
        required=True,
        help=(
            "LoRA adapter reference. Either a local path (or ``local://path``), "
            "an HF repo id, or ``hf_repo_id::subfolder``."
        ),
    )
    ap.add_argument("--n-personas", type=int, default=200,
                    help="Number of personas to sample from the rollout.")
    ap.add_argument("--label", required=True,
                    help="Output subdir name + label in summary JSON, e.g. initiative_amp.")
    ap.add_argument(
        "--upload-monorepo",
        action="store_true",
        help=(
            "Push the eval folder to the monorepo at "
            "fine_tuning/.../{trait}/{direction}/v{version}/evals/factor_validate/{label}/. "
            "Requires --direction and --monorepo-version."
        ),
    )
    ap.add_argument(
        "--direction",
        choices=["amplifier", "suppressor"],
        default=None,
        help="Adapter direction (required iff --upload-monorepo is set).",
    )
    ap.add_argument(
        "--monorepo-version",
        default="unsup_k4_v7_pf3_paired_dpo",
        help="Monorepo version under which to upload results (no leading 'v').",
    )
    args = ap.parse_args()
    if args.upload_monorepo and args.direction is None:
        ap.error("--upload-monorepo requires --direction")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

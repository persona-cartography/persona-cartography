"""Validate a trained unsup_4fac LoRA by re-administering the FA questionnaire.

Goal: take a representative subsample of the existing 2500-persona B rollout,
re-administer the v5 + trait_ocean_natural_v1 combined questionnaire on
Llama-3.1-8B-Instruct + LoRA, and compare per-persona factor scores against
the baseline (no-LoRA) scores from the paper FA fit.

The expectation is that the target factor's score moves in the direction the
LoRA was trained for (up for amplifier, down for suppressor), while the other
three factors shift much less (the constitution explicitly told the teacher
to keep them neutral). The script reports per-factor paired diffs for all
four factors regardless of ``--target``; ``--target`` only controls which
row is highlighted in the console output and the default label.

Usage::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py \\
        --target conviction \\
        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/amplifier/vunsup_4fac_paired_dpo/lora/<adapter-subfolder> \\
        --n-personas 200 \\
        --label conviction_amp

    # Or for warmth (the original use case):
    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py \\
        --target warmth \\
        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/warmth/amplifier/vunsup_4fac_paired_dpo/lora/<adapter-subfolder> \\
        --n-personas 200 \\
        --label warmth_amp

Pre-reqs:
    1. ``inspect_factor_loadings.py`` has been run (so ``scratch/factor_inspect/``
       contains the cached rollout dir + the saved FA fit).
    2. The patch to ``QuestionnaireStageConfig`` adding ``adapter_path`` is
       applied (it is, in this branch).
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
from src_dev.psychometric.hf_paths import hf_runs_path  # noqa: E402
from src_dev.psychometric.preprocessing import preprocess_response_matrix  # noqa: E402
from src_dev.psychometric.questionnaire_inference import (  # noqa: E402
    run_questionnaire_inference_async,
)
from src_dev.psychometric.questionnaire_io import load_questionnaire  # noqa: E402
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
ROLLOUT_HF_PATH = hf_runs_path(
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6"
)
ROLLOUT_LOCAL = Path("scratch/factor_inspect/hydrated") / Path(ROLLOUT_HF_PATH).name

V5_HF_PATH = hf_runs_path(
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_v5-likert-direct-lp20"
)
V5_LOCAL = Path("scratch/factor_inspect/hydrated") / Path(V5_HF_PATH).name

MCQ_HF_PATH = hf_runs_path(
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_trait_ocean_natural_v1-trait_mcq-aside-lp20-"
    "p2-pf2-tmv2"
)
MCQ_LOCAL = Path("scratch/factor_inspect/hydrated") / Path(MCQ_HF_PATH).name

V5_QUESTIONNAIRE = Path("datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json")
MCQ_QUESTIONNAIRE = Path("datasets/psychometric_questionnaires/trait_ocean_natural_v1.json")

FA_FIT_NPZ = Path("scratch/factor_inspect/fa_fit.npz")
FA_FIT_ITEMS = Path("scratch/factor_inspect/items.json")  # filtered 186-item list
QUESTIONNAIRE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

VALIDATE_ROOT = Path("scratch/factor_inspect/validate")
NUM_CONVERSATION_TURNS = 15  # match the B preset used in the paper


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
    """Write a clean rollout dir at ``dst_dir`` containing only ``kept_ids``.

    Mirrors the on-disk structure of a B rollout: top-level metadata files,
    ``datasets/{canonical_samples,sample_inputs,message_events}.jsonl``, and
    optional ``events/`` sub-dir. Filters every .jsonl that has a
    ``sample_id`` field; copies everything else verbatim.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Top-level files (manifest, run_info, archetype assignments, etc.)
    for child in src_dir.iterdir():
        if child.is_file():
            shutil.copy(child, dst_dir / child.name)

    # datasets/ — filter the row-major jsonls.
    src_ds = src_dir / "datasets"
    dst_ds = dst_dir / "datasets"
    if src_ds.exists():
        dst_ds.mkdir(exist_ok=True)
        for child in src_ds.iterdir():
            if child.suffix == ".jsonl":
                _filter_jsonl(child, dst_ds / child.name, kept_ids)
            else:
                shutil.copy(child, dst_ds / child.name)

    # events/ — same treatment (events have sample_id when relevant).
    src_ev = src_dir / "events"
    dst_ev = dst_dir / "events"
    if src_ev.exists():
        dst_ev.mkdir(exist_ok=True)
        for child in src_ev.iterdir():
            if child.suffix == ".jsonl":
                _filter_jsonl(child, dst_ev / child.name, kept_ids)
            else:
                shutil.copy(child, dst_ev / child.name)

    # Other directories — copy verbatim (small).
    for child in src_dir.iterdir():
        if child.is_dir() and child.name not in ("datasets", "events"):
            shutil.copytree(child, dst_dir / child.name, dirs_exist_ok=True)

    return dst_dir


# ── Hydrate everything we need from HF (one-time setup) ────────────────────


def hydrate_inputs() -> None:
    """Pull rollout dir + both per-questionnaire dirs from HF if missing."""
    pairs = [
        (ROLLOUT_HF_PATH, ROLLOUT_LOCAL),
        (V5_HF_PATH, V5_LOCAL),
        (MCQ_HF_PATH, MCQ_LOCAL),
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
    """Pick ``n`` sample_ids stratified by archetype × scenario when possible.

    Falls back to uniform random if the archetype assignments file is
    missing.
    """
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
    # Bucket rows by (archetype). Scenario stratification is too granular
    # (100 scenarios / 200 samples ⇒ <2 per cell); archetype is enough to
    # ensure each persona-type is represented.
    by_arch: dict[str, list[str]] = {}
    for r in rows:
        sid = r["sample_id"]
        row_idx = r["source_info"]["row_index"]
        a = arch_assign.get(str(row_idx), "_unknown")
        by_arch.setdefault(a, []).append(sid)

    # Proportional allocation per archetype, with at least 1 per archetype
    # if we have headroom.
    n_archs = len(by_arch)
    per_arch_floor = max(1, n // n_archs - 1)
    chosen: list[str] = []
    for a, ids in by_arch.items():
        take = min(len(ids), max(per_arch_floor, int(round(n * len(ids) / len(rows)))))
        chosen.extend(rng.sample(ids, take))
    # If under-quota due to rounding, top up with a uniform draw from the rest.
    if len(chosen) < n:
        rest = [r["sample_id"] for r in rows if r["sample_id"] not in set(chosen)]
        chosen.extend(rng.sample(rest, n - len(chosen)))
    elif len(chosen) > n:
        chosen = rng.sample(chosen, n)

    print(f"[validate] sampled {len(chosen)} personas (target={n}) "
          f"across {n_archs} archetypes")
    return chosen


# ── Questionnaire admin (LoRA-aware) ───────────────────────────────────────


async def admin_one_questionnaire(
    *,
    rollout_dir: Path,
    questionnaire_path: Path,
    fa_block: str,                     # "likert" or "trait_mcq"
    adapter_path: str | None,
    output_dir: Path,
    questionnaire_version: str,
    trait_mcq_topic_switch_prefix: bool,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Admin one questionnaire on the subsample with optional LoRA, return (M, items, meta)."""
    items, column_defs = load_questionnaire(questionnaire_path, fa_blocks=(fa_block,))

    cfg = QuestionnaireStageConfig(
        ctx=None,  # not used by run_questionnaire_inference_async
        questionnaire_path=questionnaire_path,
        questionnaire_version=questionnaire_version,
        fa_blocks=(fa_block,),
        use_logprobs=True,
        phrasing="aside" if fa_block == "trait_mcq" else "direct",
        trait_mcq_topic_switch_prefix=trait_mcq_topic_switch_prefix,
        provider="vllm",
        model=QUESTIONNAIRE_MODEL,
        adapter_path=adapter_path,
        max_new_tokens=32,
        max_concurrent=32,
        timeout=120,
        vllm_personas_per_batch=8,
        vllm_gpu_memory_utilization=0.92,
        top_logprobs=20,
        # The B preset's own questionnaire runs use these defaults.
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


# ── Combine + filter to FA-fit columns ─────────────────────────────────────


def align_to_fit_items(
    M_combined: np.ndarray,
    items_combined: list[dict],
    fit_items: list[dict],
) -> np.ndarray:
    """Slice combined matrix columns to match the FA fit's 186 items, in order.

    Match by ``item_id`` + block. Columns not present in fit_items are
    dropped; columns missing from the new run raise an error (we can't
    score factors on partially-administered items).
    """
    new_index: dict[tuple[str, str], int] = {
        (it["item_id"], it.get("block", "")): i for i, it in enumerate(items_combined)
    }
    cols: list[int] = []
    missing: list[tuple[str, str]] = []
    for it in fit_items:
        key = (it["item_id"], it.get("block", ""))
        if key not in new_index:
            missing.append(key)
        else:
            cols.append(new_index[key])
    if missing:
        raise RuntimeError(
            f"Subsample run is missing {len(missing)} items present in the "
            f"FA fit (e.g. {missing[:3]}). Cannot score factors."
        )
    print(f"[validate] aligned {len(cols)} columns to FA-fit items")
    return M_combined[:, cols]


# ── Factor identity anchors ───────────────────────────────────────────────
#
# The refit FA is supposed to be deterministic with seed=436, but
# ``factor_analyzer`` doesn't guarantee a stable factor ordering or sign
# convention across library versions / preprocessing tweaks. To catch silent
# drift, we anchor each canonical factor to a small set of high-loading items
# whose signed loadings we read off the paper FA fit (see
# ``paper/appendices/fa_factors.tex``). After every refit we find the unique
# permutation + sign-flip that maps refit factors to canonical ones, and
# abort if no clean mapping exists.
#
# Anchor format: {canonical_factor_index: [(text_substring, expected_sign), ...]}.
# The substring must uniquely identify one item in ``items_pp`` (case-
# insensitive substring match on ``item['text']``). The sign is the sign of
# that item's loading on the canonical factor in the paper fit.
EXPECTED_FACTOR_ANCHORS: dict[int, list[tuple[str, int]]] = {
    0: [  # F0_Conviction
        ("instinct is to verify their claim", +1),
        ("make that shift visible rather than just presenting", +1),
        ("over-correct by agreeing too quickly", -1),
    ],
    1: [  # F1_Exuberance (MCQ-only)
        ("planning the dinner party to make it a success", +1),
        ("make the dinner date special and memorable", +1),
    ],
    2: [  # F2_Warmth
        ("makes a joke in their message, i try to match", +1),
        ("adding wit or playfulness to a response", +1),
        ("matching someone's informal energy", -1),
    ],
    3: [  # F3_Didacticism
        ("explaining the underlying concept wastes their time", +1),
        ("match my response length to the complexity", +1),
        ("carry them out first and offer my perspective only if asked", -1),
    ],
}
CANONICAL_FACTOR_NAMES = ["F0_Conviction", "F1_Exuberance", "F2_Warmth", "F3_Didacticism"]

# Map ``--target`` value (lower-case factor name) to the canonical factor
# index in CANONICAL_FACTOR_NAMES / fa.scores_ columns. Used by report_comparison
# to highlight the target row and by parse_args to validate.
TARGET_FACTOR_INDEX: dict[str, int] = {
    "conviction":  0,
    "exuberance":  1,
    "warmth":      2,
    "didacticism": 3,
}


def _find_anchor_rows(items: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Locate each anchor in ``items``. Returns ``{factor: [(row_idx, sign), ...]}``."""
    out: dict[int, list[tuple[int, int]]] = {}
    for fi, anchors in EXPECTED_FACTOR_ANCHORS.items():
        rows: list[tuple[int, int]] = []
        for substr, sign in anchors:
            sub_lc = substr.lower()
            matches = [i for i, it in enumerate(items) if sub_lc in it.get("text", "").lower()]
            if not matches:
                raise RuntimeError(
                    f"Factor F{fi} anchor not found in items: {substr!r}. "
                    "The questionnaire format may have changed; update "
                    "EXPECTED_FACTOR_ANCHORS."
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Factor F{fi} anchor {substr!r} matched {len(matches)} items "
                    "(should be unique). Pick a more distinctive substring."
                )
            rows.append((matches[0], sign))
        out[fi] = rows
    return out


def align_factor_identity(
    loadings: np.ndarray,
    items: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Find the permutation + sign-flip mapping refit factors to canonical ones.

    For each (canonical_i, refit_j) pair, scores how well refit factor j
    represents canonical factor i by averaging ``expected_sign * sign(loading)``
    over the canonical factor's anchors. Then brute-forces all 4! = 24
    permutations to maximise total absolute alignment score, and reads off
    the per-factor sign from the assigned cell.

    Aborts with a descriptive error if the best alignment leaves any anchor
    on the wrong sign — this means the refit FA's structure has drifted
    from the paper fit and the validation should not be trusted.

    Args:
        loadings: ``(n_items, 4)`` from a fresh FA refit.
        items: aligned ``items_pp`` list (same length as loadings).

    Returns:
        ``(perm, signs)`` such that ``loadings[:, perm] * signs[None, :]``
        is the canonical-order, canonical-sign loading matrix. Use the same
        permutation+signs to canonicalise ``fa.transform(...)`` outputs and
        ``fa.scores_``.
    """
    k = loadings.shape[1]
    assert k == 4, f"expected 4 factors, got {k}"

    anchor_rows = _find_anchor_rows(items)

    # Per (canonical_i, refit_j): mean over i's anchors of expected_sign·sign(load).
    # Loadings with magnitude < 0.1 are treated as ambiguous (contribute 0).
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
                    f"row={row} text={items[row].get('text','')[:70]!r}: "
                    f"aligned_loading={aligned_load:+.3f} but expected sign "
                    f"{expected_sign:+d}"
                )
    if failures:
        raise RuntimeError(
            "Factor identity check FAILED — refit FA does not match the "
            f"paper's k=4 oblimin canonical factors.\n"
            f"Best permutation: {perm_arr.tolist()}, signs: {signs.tolist()}\n"
            "Anchor mismatches:\n" + "\n".join(failures)
        )

    if not (perm_arr.tolist() == [0, 1, 2, 3] and signs.tolist() == [1, 1, 1, 1]):
        print(
            f"[validate] WARNING: refit factors required reordering to canonical "
            f"identity. permutation={perm_arr.tolist()} signs={signs.tolist()}. "
            f"This usually means the FA refit drifted from the paper fit "
            f"(library version / preprocessing change?); results still valid "
            f"after the alignment, but worth investigating."
        )
    else:
        print("[validate] factor identity check passed (canonical order preserved).")

    return perm_arr, signs


# ── FA refit + scoring ────────────────────────────────────────────────────


def refit_fa_for_scoring() -> tuple[FactorAnalyzer, np.ndarray, list[dict], list[str], np.ndarray, np.ndarray]:
    """Refit the paper's FA on the cached baseline matrix and return the
    fitted ``FactorAnalyzer``, the preprocessed response matrix, the
    aligned column items, the per-row sample_ids, and the canonical-
    identity ``(perm, signs)`` for fa.transform outputs.

    Deterministic with seed=436. Used so we can call ``fa.transform()`` on
    both the baseline subsample and the LoRA-administered matrix, matching
    the library's internal centering/standardisation.
    """
    Mv, mv, items_v = load_pair_outputs(V5_LOCAL)
    Mm, mm, items_m = load_pair_outputs(MCQ_LOCAL)
    for it in items_v:
        it.setdefault("version", "v5")
    for it in items_m:
        it.setdefault("version", "trait_ocean_natural_v1")
    sid_v = {m["sample_id"]: i for i, m in enumerate(mv)}
    sid_m = {m["sample_id"]: i for i, m in enumerate(mm)}
    common = sorted(set(sid_v) & set(sid_m))
    M = np.hstack([Mv[[sid_v[s] for s in common]], Mm[[sid_m[s] for s in common]]])
    items = items_v + items_m
    M_pp, meta_pp, items_pp, _ = preprocess_response_matrix(
        M,
        [{"sample_id": s} for s in common],
        items,
        min_item_variance=0.1,
        high_variance_persona_drop_pct=0.0,
        do_residualize=False,
    )
    fa = FactorAnalyzer(n_factors=4, method="principal", rotation="oblimin")
    fa.fit(M_pp)
    perm, signs = align_factor_identity(fa.loadings_, items_pp)
    return fa, M_pp, items_pp, [m["sample_id"] for m in meta_pp], perm, signs


def aligned_transform(fa: FactorAnalyzer, M: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """``fa.transform(M)`` reordered + sign-flipped to canonical (F0..F3) identity."""
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

    For each factor F, restrict to items where ``argmax(|loadings|) == F`` AND
    ``|loadings[:, F]| >= threshold``. The pole-aligned shift per item is
    ``sign(loading) * (mean(M_lora) - mean(M_baseline))``: positive means the
    LoRA pushed the item toward F's HIGH pole, negative toward LOW pole.

    This is more honest than ``fa.transform`` deltas when the LoRA's response
    distribution is far from the FA's training distribution — ``fa.transform``
    can inflate σ-units on out-of-distribution responses, whereas item-level
    shifts are reported on the raw response scale.

    Args:
        M_lora: ``(n_personas, n_items)`` LoRA-administered responses, aligned
            to ``items_pp`` columns.
        M_baseline: ``(n_personas, n_items)`` baseline responses for the same
            personas, aligned to ``items_pp`` columns.
        loadings_aligned: ``(n_items, 4)`` canonical-identity loadings.
        items_pp: ``items_pp`` matching the column order of M_*.
        threshold: minimum |loading| for an item to be considered dominant
            on its factor (psychometric salience convention: 0.4).

    Returns:
        One dict per factor in canonical (F0..F3) order.
    """
    delta_mean = M_lora.mean(0) - M_baseline.mean(0)        # [n_items]
    dominant = np.abs(loadings_aligned).argmax(axis=1)      # [n_items]
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
    """Per-factor paired comparison + JSON dump + violin plot.

    ``target_factor_index``: the index of the factor that the LoRA was trained
    along (0..3). When provided, the console output marks that row with an
    arrow and the violin plot highlights it. The math runs over all factors
    regardless.
    """
    diff = lora_scores - baseline_scores                  # [N, k]
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
        # Bootstrap 95% CI on mean diff.
        rng = np.random.default_rng(SEED)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        # Effect size (Cohen's dz on paired diffs).
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

    # Save raw arrays so we can re-plot later.
    np.savez(
        output_dir / f"{label}_scores.npz",
        sample_ids=np.array(sample_ids),
        lora_scores=lora_scores,
        baseline_scores=baseline_scores,
    )

    # Violin plot — paired per-persona diffs per factor.
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

    # 3. Run questionnaire admin × 2 with the LoRA.
    out_v5  = VALIDATE_ROOT / label / "questionnaire_v5"
    out_mcq = VALIDATE_ROOT / label / "questionnaire_trait_ocean_natural_v1"

    print("[validate] running v5 likert questionnaire admin")
    M_v5, items_v5, meta_v5 = await admin_one_questionnaire(
        rollout_dir=sub_rollout,
        questionnaire_path=V5_QUESTIONNAIRE,
        fa_block="likert",
        adapter_path=args.adapter,
        output_dir=out_v5,
        questionnaire_version="v5",
        trait_mcq_topic_switch_prefix=False,
    )
    for it in items_v5:
        it["version"] = "v5"

    print("[validate] running trait_ocean_natural_v1 trait_mcq questionnaire admin")
    M_mcq, items_mcq, meta_mcq = await admin_one_questionnaire(
        rollout_dir=sub_rollout,
        questionnaire_path=MCQ_QUESTIONNAIRE,
        fa_block="trait_mcq",
        adapter_path=args.adapter,
        output_dir=out_mcq,
        questionnaire_version="trait_ocean_natural_v1",
        trait_mcq_topic_switch_prefix=True,
    )
    for it in items_mcq:
        it["version"] = "trait_ocean_natural_v1"

    # 4. Combine on intersection of sample_ids.
    sid_v5 = {m["sample_id"]: i for i, m in enumerate(meta_v5)}
    sid_mcq = {m["sample_id"]: i for i, m in enumerate(meta_mcq)}
    common = [s for s in sample_ids if s in sid_v5 and s in sid_mcq]
    print(f"[validate] {len(common)} personas with both questionnaires complete")
    M_v5_ord = M_v5[[sid_v5[s] for s in common]]
    M_mcq_ord = M_mcq[[sid_mcq[s] for s in common]]
    M_lora = np.hstack([M_v5_ord, M_mcq_ord])
    items_combined = list(items_v5) + list(items_mcq)

    # 5. Drop NaN-row personas (logprob parse failures, etc.).
    keep = ~np.isnan(M_lora).any(axis=1)
    if not keep.all():
        print(f"[validate] dropping {(~keep).sum()} personas with NaNs in response matrix")
    M_lora = M_lora[keep]
    common = [s for s, k in zip(common, keep) if k]

    # 6. Refit the FA on the full baseline matrix (deterministic, ~30s) to
    # get a FactorAnalyzer object whose .transform() reproduces the paper
    # scores exactly. Slice the LoRA matrix to the fit's column order, and
    # canonicalise factor order/sign via anchor-based identity check.
    print("[validate] refitting FA on baseline matrix for scoring...")
    fa, M_baseline_full, fit_items, baseline_sids, perm, signs = refit_fa_for_scoring()
    factor_identity = {"permutation": perm.tolist(), "signs": signs.tolist()}
    loadings_aligned = fa.loadings_[:, perm] * signs[None, :]

    M_lora_aligned = align_to_fit_items(M_lora, items_combined, fit_items)

    # 7. Score baseline subset for the same personas (paired comparison).
    sid_to_baseline_row = {s: i for i, s in enumerate(baseline_sids)}
    rows = [sid_to_baseline_row[s] for s in common if s in sid_to_baseline_row]
    common_with_baseline = [s for s in common if s in sid_to_baseline_row]
    if len(rows) != len(common):
        print(
            f"[validate] {len(common) - len(rows)} subsample personas missing "
            "from baseline FA (preprocessing dropped them)"
        )
    baseline_scores = aligned_transform(fa, M_baseline_full[rows], perm, signs)

    # Filter LoRA scores to the same personas (in case any were dropped above).
    lora_scores = aligned_transform(fa, M_lora_aligned, perm, signs)
    keep_idx = [i for i, s in enumerate(common) if s in sid_to_baseline_row]
    lora_scores = lora_scores[keep_idx]
    M_lora_paired = M_lora_aligned[keep_idx]
    M_baseline_paired = M_baseline_full[rows]

    # 8. Item-level shifts on raw response scale (matched personas).
    item_level_summary = report_item_level_shifts(
        M_lora=M_lora_paired,
        M_baseline=M_baseline_paired,
        loadings_aligned=loadings_aligned,
        items_pp=fit_items,
        threshold=0.4,
    )

    # 9. Compare and report.
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

    # 10. Optional: push the eval folder to the monorepo so the results are
    # visible alongside the trained adapter.
    if args.upload_monorepo:
        from src_dev.utils import upload_folder_to_dataset_repo

        path_in_repo = (
            f"fine_tuning/llama-3.1-8b-it/unsupervised/{args.target}/"
            f"{args.direction}/v{args.monorepo_version}/evals/"
            f"factor_validate/{label}"
        )
        # Skip the bulky rollout subsample — it's a filtered copy of the
        # original rollout and trivially reconstructable. Keep summary,
        # scores, plot, and the per-questionnaire outputs (response matrix,
        # raw responses, items, metadata) so anyone can re-run analysis.
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
                    help="Output subdir name + label in summary JSON, e.g. conviction_amp.")
    ap.add_argument(
        "--upload-monorepo",
        action="store_true",
        help=(
            "Push the eval folder to the monorepo at "
            "fine_tuning/.../{trait}/{direction}/v{version}/evals/factor_validate/{label}/. "
            "Requires --direction. Skips the rollout_subsample/ subfolder."
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
        default="unsup_4fac_paired_dpo",
        help=(
            "Version segment in the monorepo path (without leading 'v'). "
            "Default matches the unsup_4fac paired-DPO training runs."
        ),
    )
    args = ap.parse_args()
    if args.upload_monorepo and args.direction is None:
        ap.error("--upload-monorepo requires --direction {amplifier,suppressor}")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

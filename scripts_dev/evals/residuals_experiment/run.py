#!/usr/bin/env python3
"""LoRA additivity / interaction-residuals experiment runner.

Generates rollouts and judge scores for all 56 combinations of OCEAN adapters:

  1 baseline  S(W)
 10 singles   S(W + Δi)          for each i in 10 adapters
 45 pairs     S(W + Δi + Δj)     for each unordered pair i < j
 ──────────────────────────────
 56 total     each scored by all 5 OCEAN v2 judges → 280 scores

All adapters are applied at scale +1.  The per-cell 5-vectors are written to
``scratch/residuals_experiment/scores.json`` for the analysis step.

Design note — measurement nonlinearity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Large residuals can arise from three sources:
  (a) genuine adapter interaction in weight space,
  (b) trait entanglement (one adapter affects a trait the other adapter also
      targets),
  (c) judge-score saturation near ±4 (a purely measurement artefact).

At scale=1 all adapters are far from the saturation range where vrun4_paired_dpo
single-adapter sweeps flatten, so source (c) is likely small.  Sources (a) and
(b) are the scientifically interesting ones.

Usage::

    # Dry-run: print config and cost estimates, touch nothing
    uv run python -m scripts_dev.evals.residuals_experiment.run --dry-run

    # Full run on a single GPU
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.evals.residuals_experiment.run

    # Skip rollouts (reuse existing local rollouts) and re-run judge only
    uv run python -m scripts_dev.evals.residuals_experiment.run --skip-rollouts

    # Skip both rollouts and judge; just re-aggregate existing judge outputs
    uv run python -m scripts_dev.evals.residuals_experiment.run --skip-rollouts --skip-judge

    # Run without uploading to HuggingFace
    uv run python -m scripts_dev.evals.residuals_experiment.run --no-upload
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Set env vars before any CUDA/vLLM imports.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_V1", "1")
# vLLM v1 forks worker processes; if CUDA was touched in the parent the fork
# will crash.  Spawn avoids this.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Config — adjust MAX_SAMPLES and JUDGE_REPEATS to trade cost vs. precision
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
HF_REPO_ID = "persona-shattering-lasr/monorepo"

# Mixed OCEAN benchmark: 250 questions, 50 per trait.
# DESIGN NOTE: using a single balanced dataset (rather than per-trait datasets)
# keeps the fingerprint stable across all 56 cells and avoids needing 5 separate
# rollout passes.  If you want to reuse existing vrun4_paired_dpo single-
# adapter rollouts, you would need to match each single-trait dataset + params.
DATASET_PATH = "data/trait_benchmark_ocean250.jsonl"
MAX_SAMPLES = 250   # full production run (50 per trait)

# Max cells per vLLM pass.  Each baked pair-adapter can be ~1.2 GB on disk;
# baking all 55 at once requires ~55 GB which can exceed available scratch
# space on a 100 GB root volume.  Batching keeps peak disk usage low.
ROLLOUT_BATCH_SIZE = 10

SEED = 42
NUM_ROLLOUTS_PER_PROMPT = 1
ASSISTANT_TEMPERATURE = 1.0     # matches vrun4_paired_dpo
ASSISTANT_TOP_P = 1.0           # matches vrun4_paired_dpo
ASSISTANT_MAX_NEW_TOKENS = 2048  # matches vrun4_paired_dpo
ASSISTANT_BATCH_SIZE = 32
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

ADAPTER_SCALE = 1.0  # fixed for all adapters throughout

JUDGE_REPEATS = 1    # 1 repeat keeps cost manageable (280 cells × 5 metrics)
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000

# All 5 OCEAN v2 metrics judged for every cell.
OCEAN_TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
OCEAN_METRICS = [f"{t}_v2" for t in OCEAN_TRAIT_NAMES]

EVAL_NAME = "residuals-vrun4-paired-dpo"
HF_TARGET_PATH = f"evals/residuals_experiment/{EVAL_NAME}"

SCRATCH_ROOT = project_root / "scratch" / "residuals_experiment"
STAGING_ROOT = project_root / "scratch" / "residuals_staging"
BAKED_ROOT = project_root / "scratch" / "residuals_baked"

# ---------------------------------------------------------------------------
# Adapter registry — canonical OCEAN order (O C E A N), amp then sup, + ctrl
# ---------------------------------------------------------------------------
# Importing inside a function avoids heavy imports at module load time (so
# --dry-run stays fast), but the list is module-level for clarity.

# Control adapter: paired-DPO training on neutral OCEAN-defining conversations.
CONTROL_ADAPTER_PATH = (
    "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier"
    "/vrun4_paired_dpo_s1vs2/lora/ocean_def_control_full_vrun4-persona"
)


def _build_adapters() -> list[Any]:
    from src_dev.common.lora_catalogue import OCEAN_REGISTRY
    from src_dev.evals.cell_sweep.cell_identity import AdapterSpec

    ordered_keys = [
        "o_plus", "o_minus",
        "c_plus", "c_minus",
        "e_plus", "e_minus",
        "a_plus", "a_minus",
        "n_plus", "n_minus",
    ]
    adapters = [
        AdapterSpec.from_ref(f"{HF_REPO_ID}::{OCEAN_REGISTRY[k].adapter_path_in_repo}")
        for k in ordered_keys
    ]
    # Append the control adapter as the 11th entry.
    adapters.append(AdapterSpec.from_ref(f"{HF_REPO_ID}::{CONTROL_ADAPTER_PATH}"))
    return adapters


# Friendly short slug for each AdapterSpec.slug → OCEAN_REGISTRY key mapping.
# Used only in display / analysis output.
def _build_friendly_slug_map() -> dict[str, str]:
    from src_dev.common.lora_catalogue import OCEAN_REGISTRY
    from src_dev.evals.cell_sweep.cell_identity import AdapterSpec

    result: dict[str, str] = {}
    for key, td in OCEAN_REGISTRY.items():
        spec = AdapterSpec.from_ref(f"{HF_REPO_ID}::{td.adapter_path_in_repo}")
        result[spec.slug] = key
    ctrl_spec = AdapterSpec.from_ref(f"{HF_REPO_ID}::{CONTROL_ADAPTER_PATH}")
    result[ctrl_spec.slug] = "ctrl"
    return result


# ---------------------------------------------------------------------------
# Cell enumeration: 1 baseline + C(10,1)=10 singles + C(10,2)=45 pairs = 56
# ---------------------------------------------------------------------------


def enumerate_residual_cells(adapters: list[Any]) -> list[Any]:
    """Return the 56 cells for the residuals experiment (degree ≤ 2 only)."""
    from src_dev.evals.cell_sweep.cell_identity import CanonicalCell

    cells: list[Any] = []

    # Baseline: no adapters
    cells.append(CanonicalCell(entries=()))

    # Single-adapter cells
    for a in adapters:
        cells.append(CanonicalCell.from_scales([(a, ADAPTER_SCALE)]))

    # Pair cells (unordered, i < j)
    for i, a in enumerate(adapters):
        for b in adapters[i + 1:]:
            cells.append(CanonicalCell.from_scales([(a, ADAPTER_SCALE), (b, ADAPTER_SCALE)]))

    return cells


# ---------------------------------------------------------------------------
# Rollout generation (all 56 cells in a single vLLM pass)
# ---------------------------------------------------------------------------


def _has_rollouts(cell_dir: Path) -> bool:
    p = cell_dir / "rollouts" / "rollouts.jsonl"
    return p.exists() and p.stat().st_size > 0


def _cell_local_dir(cell: Any, fingerprint: str) -> Path:
    return SCRATCH_ROOT / "cells" / fingerprint / cell.variant_label()


def _generate_all_rollouts(
    cells: list[Any],
    cell_dirs: dict[Any, Path],
    sweep_id: str,
) -> None:
    """Run all cells through vLLM in one combined sweep, then scatter to cell dirs."""
    from src_dev.rollout_generation.model_providers import CellSpec, VLLMLoRaComboProvider
    from src_dev.sweep import (
        ExperimentConfig,
        OutputPathConfig,
        SweepConfig,
        run_sweep,
        single_turn_conditions,
    )

    staging_root = STAGING_ROOT / sweep_id
    staging_root.mkdir(parents=True, exist_ok=True)

    conditions = single_turn_conditions({"no_prompt": None})
    condition_name = conditions[0].name  # e.g. "1turn_astNoSProm___no_prompt"

    cell_specs = [
        CellSpec(
            label=c.variant_label(),
            # For baseline, entries=() → adapter_scales=() (runs bare base model)
            adapter_scales=tuple((spec.ref, sc) for spec, sc in c.entries),
        )
        for c in cells
    ]

    provider = VLLMLoRaComboProvider(
        base_model=BASE_MODEL,
        cells=cell_specs,
        baked_adapters_dir=BAKED_ROOT / sweep_id,
        temperature=ASSISTANT_TEMPERATURE,
        top_p=ASSISTANT_TOP_P,
        max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
    )

    # Use hf_repo=None so run_sweep doesn't upload staging outputs.
    output_config = OutputPathConfig(
        scratch_root=staging_root,
        hf_repo=None,
        base_model="_staging",
        category="_residuals",
        trait="_cells",
        direction="_combo",
        version=sweep_id,
        stage_dir="",
        eval_name="",
    )

    sweep_config = SweepConfig(
        provider=provider,
        conditions=conditions,
        evaluations=[],
        experiment=ExperimentConfig(
            assistant_model=BASE_MODEL,
            assistant_provider="vllm",
            assistant_temperature=ASSISTANT_TEMPERATURE,
            assistant_top_p=ASSISTANT_TOP_P,
            assistant_max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
            assistant_batch_size=ASSISTANT_BATCH_SIZE,
            user_model=USER_MODEL,
            user_provider=USER_PROVIDER,
            user_temperature=0.7,
            user_top_p=0.95,
            user_max_new_tokens=128,
            user_batch_size=32,
            user_max_concurrent=32,
            dataset_path=DATASET_PATH,
            max_samples=MAX_SAMPLES,
            dataset_seed=SEED,
            num_rollouts=NUM_ROLLOUTS_PER_PROMPT,
            turns_per_phase=[1],
        ),
        output=output_config,
        skip_completed=True,
        skip_evals=True,
        on_cell_error="warn",
        max_concurrent_conditions=1,
        plot=False,
        metadata={"sweep_id": sweep_id, "experiment": EVAL_NAME},
    )

    run_sweep(sweep_config)

    # Scatter staging outputs to per-cell canonical dirs.
    staging_output = output_config.scratch_dir
    for cell in cells:
        src = staging_output / cell.variant_label() / condition_name
        dst = cell_dirs[cell]
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"  [rollout] staging src missing for {cell.variant_label()}: {src}")
            continue
        for item in src.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _build_judge_dataset(cell: Any, cell_dir: Path, out_path: Path) -> int:
    """Flatten one cell's rollouts into an OceanJudgeRunConfig-compatible JSONL."""
    rollouts_path = cell_dir / "rollouts" / "rollouts.jsonl"
    if not rollouts_path.exists():
        return 0
    cell_tag = cell.variant_label()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out, rollouts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seed_id = str(record.get("seed_id", ""))
            seed_input = str(record.get("seed_input", ""))
            for rollout_idx_str, turn_list in record.get("messages", {}).items():
                rollout_idx = int(rollout_idx_str)
                assistant_msgs = [m for m in turn_list if m.get("role") == "assistant"]
                if not assistant_msgs:
                    continue
                response = assistant_msgs[-1]["content"]
                out.write(json.dumps({
                    "response_id": f"{cell_tag}:{seed_id}:{rollout_idx}",
                    "condition": f"no_prompt@{cell_tag}",
                    "cell_tag": cell_tag,
                    "condition_name": "no_prompt",
                    "seed_id": seed_id,
                    "sample_id": seed_id,
                    "input_group_id": seed_id,
                    "response_index": rollout_idx,
                    "prompt_row_index": -1,
                    "prompt_id": seed_id,
                    "question": seed_input,
                    "response": response,
                    "assistant_model": BASE_MODEL,
                    "assistant_provider": "local",
                    "system_prompt_ref": "",
                }) + "\n")
                n += 1
    return n


def _has_judge_output(cell_dir: Path, rater_id: str, metric_name: str) -> bool:
    p = cell_dir / "judge_runs" / rater_id / f"{metric_name}.jsonl"
    return p.exists() and p.stat().st_size > 0


def _run_judge_for_cell_metric(
    cell: Any,
    cell_dir: Path,
    metric_name: str,
    judge_raters: list[Any],
) -> None:
    """Run one judge metric for one cell; writes to cell_dir/judge_runs/{rater}/{metric}.jsonl."""
    from src_dev.persona_metrics.llm_judge_agreement import (
        OceanJudgeRunConfig,
        get_judge_run_dir,
        run_ocean_judge_run,
    )
    from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

    # Resolve the OceanTrait for this metric (needed by OceanJudgeRunConfig).
    metric_trait = OceanTrait(metric_name.removesuffix("_v2"))

    transient_dir = cell_dir / "_judge_scratch" / metric_name
    transient_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = transient_dir / "all_responses.jsonl"

    n = _build_judge_dataset(cell, cell_dir, dataset_path)
    if n == 0:
        print(f"  [judge] {cell.variant_label()} / {metric_name}: no rollouts, skipping")
        return

    raters = [r.model_copy(update={"metric_name": metric_name}) for r in judge_raters]
    judge_cfg = OceanJudgeRunConfig(
        trait=metric_trait,
        dataset_path=dataset_path,
        judge_raters=raters,
        judge_repeats=JUDGE_REPEATS,
        plot=False,
        hf_repo_id=HF_REPO_ID,
        upload=False,
    )
    run_ocean_judge_run(judge_cfg)

    judge_dir = get_judge_run_dir(judge_cfg)
    raw_dir = judge_dir / "judge_calls" / "raw"
    if not raw_dir.exists():
        print(f"  [judge] {cell.variant_label()} / {metric_name}: no raw output produced")
        return

    # Consolidate per-rater raw JSONL into cell_dir/judge_runs/{rater}/{metric}.jsonl
    for rater in judge_raters:
        out_file = cell_dir / "judge_runs" / rater.rater_id / f"{metric_name}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as out:
            for raw_path in sorted(raw_dir.glob(f"{rater.rater_id}*.jsonl")):
                with raw_path.open() as f:
                    for raw_line in f:
                        raw_line = raw_line.rstrip("\n")
                        if raw_line:
                            out.write(raw_line + "\n")


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def _cell_mean_score(cell_dir: Path, rater_ids: list[str], metric_name: str) -> float | None:
    """Mean judge score across all responses and raters for one cell × metric."""
    scores: list[float] = []
    for rater_id in rater_ids:
        path = cell_dir / "judge_runs" / rater_id / f"{metric_name}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") not in {"success", "parse_error"}:
                    continue
                score = record.get("score")
                if isinstance(score, (int, float)):
                    scores.append(float(score))
    return statistics.mean(scores) if scores else None


def extract_all_scores(
    cells: list[Any],
    cell_dirs: dict[Any, Path],
    judge_raters: list[Any],
) -> dict[str, dict[str, float | None]]:
    """Build cell_tag → {metric → mean_score} for all cells."""
    rater_ids = [r.rater_id for r in judge_raters]
    result: dict[str, dict[str, float | None]] = {}
    for cell in cells:
        tag = cell.variant_label()
        result[tag] = {
            metric: _cell_mean_score(cell_dirs[cell], rater_ids, metric)
            for metric in OCEAN_METRICS
        }
    return result


# ---------------------------------------------------------------------------
# CLI + dry-run
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA additivity / interaction-residuals experiment runner.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and cost estimates; touch nothing.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload.")
    parser.add_argument("--skip-rollouts", action="store_true",
                        help="Reuse existing local rollouts; skip vLLM generation.")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge scoring; only aggregate existing outputs.")
    return parser.parse_args()


def _print_dry_run(cells: list[Any], fingerprint: str) -> None:
    n_rollouts = len(cells) * MAX_SAMPLES * NUM_ROLLOUTS_PER_PROMPT
    n_judge_calls = n_rollouts * len(OCEAN_METRICS) * JUDGE_REPEATS
    baseline_cells = [c for c in cells if c.tier == "baseline"]
    single_cells = [c for c in cells if c.tier == "single_adapter"]
    pair_cells = [c for c in cells if c.tier == "combo"]

    print("DRY RUN: LoRA additivity / interaction-residuals experiment")
    print(f"  eval name        : {EVAL_NAME}")
    print(f"  base model       : {BASE_MODEL}")
    print(f"  dataset          : {DATASET_PATH}")
    print(f"  max_samples      : {MAX_SAMPLES}  ({MAX_SAMPLES // 5} per trait)")
    print(f"  seed             : {SEED}")
    print(f"  adapter_scale    : {ADAPTER_SCALE:+.1f}")
    print(f"  cells            : {len(cells)}"
          f"  (baseline={len(baseline_cells)}, single={len(single_cells)}, pairs={len(pair_cells)})")
    print(f"  rollouts total   : {n_rollouts}")
    print(f"  judge metrics    : {OCEAN_METRICS}")
    print(f"  judge repeats    : {JUDGE_REPEATS}")
    print(f"  judge calls est  : {n_judge_calls}")
    print(f"  fingerprint      : {fingerprint}")
    print(f"  local output     : {SCRATCH_ROOT}")
    print(f"  HF path          : {HF_REPO_ID}/{HF_TARGET_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    flags = _parse_flags()
    load_dotenv()

    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    from src_dev.persona_metrics.config import JudgeLLMConfig
    from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
    from src_dev.evals.llm_judge_sweep.cell_identity import rollout_fingerprint
    from src_dev.utils.hf_hub import login_from_env

    judge_raters = [
        JudgeRaterConfig(
            rater_id="qwen3_235b",
            judge=JudgeLLMConfig(
                provider="openrouter",
                model="qwen/qwen3-235b-a22b-2507",
                temperature=0.0,
                max_concurrent=32,
            ),
        ),
    ]

    adapters = _build_adapters()
    cells = enumerate_residual_cells(adapters)

    fingerprint = rollout_fingerprint(
        base_model=BASE_MODEL,
        dataset_path=DATASET_PATH,
        max_samples=MAX_SAMPLES,
        seed=SEED,
        num_rollouts_per_prompt=NUM_ROLLOUTS_PER_PROMPT,
        assistant_temperature=ASSISTANT_TEMPERATURE,
        assistant_top_p=ASSISTANT_TOP_P,
        assistant_max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
    )

    if flags.dry_run:
        _print_dry_run(cells, fingerprint)
        return

    upload = not flags.no_upload
    if upload:
        login_from_env()

    cell_dirs = {c: _cell_local_dir(c, fingerprint) for c in cells}
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: rollouts ──────────────────────────────────────────────────

    if not flags.skip_rollouts:
        cells_needing_rollouts = [c for c in cells if not _has_rollouts(cell_dirs[c])]
        if cells_needing_rollouts:
            n_done = len(cells) - len(cells_needing_rollouts)
            print(f"[rollout] {len(cells_needing_rollouts)} cells need rollouts "
                  f"({n_done} already done); batching {ROLLOUT_BATCH_SIZE} cells per vLLM pass")
            # Process in batches to cap peak baked-adapter disk usage.
            for batch_start in range(0, len(cells_needing_rollouts), ROLLOUT_BATCH_SIZE):
                batch = cells_needing_rollouts[batch_start: batch_start + ROLLOUT_BATCH_SIZE]
                batch_num = batch_start // ROLLOUT_BATCH_SIZE + 1
                n_batches = math.ceil(len(cells_needing_rollouts) / ROLLOUT_BATCH_SIZE)
                print(f"[rollout] batch {batch_num}/{n_batches}: {len(batch)} cells")
                sweep_id = f"{EVAL_NAME}_{fingerprint}_{uuid.uuid4().hex[:8]}"
                baked_dir = BAKED_ROOT / sweep_id
                baked_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _generate_all_rollouts(batch, cell_dirs, sweep_id)
                finally:
                    # Baked adapter dirs can be many GB; always clean up after each batch.
                    if baked_dir.exists():
                        shutil.rmtree(baked_dir, ignore_errors=True)
                        print(f"[cleanup] removed baked dir: {baked_dir}")
        else:
            print("[rollout] all cells already have rollouts")
    else:
        print("[rollout] --skip-rollouts set; skipping")

    # ── Stage 2: judge (per-cell, all metrics in parallel threads) ─────────

    if not flags.skip_judge:
        for cell in cells:
            cell_dir = cell_dirs[cell]
            pending_metrics = [
                m for m in OCEAN_METRICS
                if not all(
                    _has_judge_output(cell_dir, r.rater_id, m) for r in judge_raters
                )
            ]
            if not pending_metrics:
                continue
            print(f"[judge] {cell.variant_label()} / {pending_metrics}")
            # Run all pending metrics for this cell in parallel (I/O bound via API).
            with ThreadPoolExecutor(max_workers=len(pending_metrics)) as pool:
                futures = [
                    pool.submit(_run_judge_for_cell_metric, cell, cell_dir, m, judge_raters)
                    for m in pending_metrics
                ]
                for fut in as_completed(futures):
                    fut.result()
    else:
        print("[judge] --skip-judge set; skipping")

    # ── Stage 3: extract scores and save ──────────────────────────────────

    friendly_map = _build_friendly_slug_map()
    scores = extract_all_scores(cells, cell_dirs, judge_raters)

    # Annotate each cell's entry with its friendly adapter slugs.
    cell_metadata: dict[str, Any] = {}
    for cell in cells:
        tag = cell.variant_label()
        cell_metadata[tag] = {
            "tier": cell.tier,
            "adapters": [
                {"slug": spec.slug, "friendly": friendly_map.get(spec.slug, spec.slug), "scale": sc}
                for spec, sc in cell.entries
            ],
        }

    scores_path = SCRATCH_ROOT / "scores.json"
    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_name": EVAL_NAME,
                "fingerprint": fingerprint,
                "adapter_scale": ADAPTER_SCALE,
                "adapter_slugs": [a.slug for a in adapters],
                "friendly_slugs": {a.slug: friendly_map.get(a.slug, a.slug) for a in adapters},
                "ocean_metrics": OCEAN_METRICS,
                "cell_metadata": cell_metadata,
                "cell_scores": {
                    tag: {m: v for m, v in metric_scores.items()}
                    for tag, metric_scores in scores.items()
                },
            },
            f,
            indent=2,
        )
    print(f"[scores] wrote {scores_path}")

    # ── Stage 4: upload to HF ──────────────────────────────────────────────

    if upload:
        from src_dev.utils.hf_hub import upload_folder_to_dataset_repo

        upload_folder_to_dataset_repo(
            local_dir=SCRATCH_ROOT,
            repo_id=HF_REPO_ID,
            path_in_repo=HF_TARGET_PATH,
            commit_message=f"{EVAL_NAME}: upload residuals experiment results",
        )
        print(f"[upload] → {HF_REPO_ID}/{HF_TARGET_PATH}")

    print(f"\nDone.  Results at: {scores_path}")
    print("Next: run analyze.py to compute and plot interaction residuals.")


if __name__ == "__main__":
    main()

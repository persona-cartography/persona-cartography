#!/usr/bin/env python3
"""Cell-oriented LLM-judge sweep runner (single-adapter and multi-adapter combos).

Replaces the stage-chained flow of :mod:`runner` with cell-level content-
addressed caching on HuggingFace. Each *cell* (unique LoRA combination at
specific scales) is atomic, hydrated/uploaded as a unit, and lives at the
canonical path given by :class:`CanonicalCell.hf_dir`.

Key differences from the old runner:

- Baseline cell (no adapters) lives under ``combos/{model}/_baseline/...``,
  not under ``fine_tuning/``.
- Single-adapter cells (one non-zero scale) live under
  ``fine_tuning/{model}/{category}/{trait}/...`` regardless of whether
  they were produced by a single-adapter sweep or a multi-adapter combo
  sweep — so a ``neu × con`` combo sweep populates the same locations as
  a ``neu`` single-adapter sweep and vice versa.
- Multi-adapter cells (≥2 non-zero scales) live under
  ``combos/{model}/{combo_slug}/...``.
- Per-cell judge output is scoped per ``(rater_id, metric_name)`` so a
  missing judge metric can be recomputed without redoing rollouts.

Config compatibility:

- **Combo configs** supply ``ADAPTERS: list[AdapterSpec]`` and
  ``SCALES_PER_ADAPTER: dict[slug, list[float]]``.
- **Legacy single-adapter configs** (``ADAPTER_REF`` + ``SCALE_POINTS``) are
  auto-promoted to the combo shape on load.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from types import ModuleType
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_V1", "1")
# vLLM v1 forks worker processes; if CUDA was touched in the parent (e.g. by
# seed_all() or PEFT baking) the fork will crash with "Cannot re-initialize
# CUDA". Spawn is the safe alternative.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from dotenv import load_dotenv

# Ensure project root is on sys.path.
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.eval_stages import seed_all
from src_dev.evals.cell_sweep.runner import (
    enumerate_cells as _enumerate_cells_generic,
    load_config_module,
    parse_sweep_flags,
    upload_sweep_root as _upload_sweep_root_generic,
    write_cell_info,
)
from src_dev.evals.llm_judge_sweep.cell_cache import (
    ROLLOUTS_RELPATH,
    cell_status_on_disk,
    hydrate_cell,
    upload_cell,
)
from src_dev.evals.llm_judge_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
    format_scale,
    rollout_fingerprint,
    sweep_hf_root,
)
from src_dev.evals.llm_judge_sweep.defaults import (
    check_sweep_defaults,
    confirm_or_abort,
)
from src_dev.rollout_generation.model_providers import cleanup_baked_dir
from src_dev.utils.hf_hub import login_from_env

HF_REPO_ID = "persona-shattering-lasr/monorepo"
EVAL_NAME_DEFAULT = "llm_judge_lora_scale_sweep"
SCRATCH_ROOT = Path("scratch/monorepo")
STAGING_ROOT = Path("scratch/sweep_staging")
BAKED_ROOT = Path("scratch/baked_combo_adapters")

# Opt-in batched-upload mode. When set, the sweep writes cells locally
# throughout and does a SINGLE HF commit per sweep at the end instead of
# per-cell commits. Use this to stay under HF's commits-per-hour rate limit
# during long/wide runs. Does not affect single-invocation semantics —
# the HF layout is identical; just fewer commits.
_BATCH_UPLOAD = os.environ.get("LLM_JUDGE_SWEEP_BATCH_UPLOAD") == "1"
# Opt-in skip-upload mode. When set, every HF upload call is a no-op — the
# sweep runs purely locally. Use this when HF is rate-limited and you want
# compute/forging to complete without wasting time on upload retries; then
# run a consolidated upload pass later.
_SKIP_UPLOAD = os.environ.get("LLM_JUDGE_SWEEP_SKIP_UPLOAD") == "1"
if _SKIP_UPLOAD:
    print("[runner_cells] LLM_JUDGE_SWEEP_SKIP_UPLOAD=1 — all HF uploads are no-ops this run")
# Grace window for legacy baked dirs with no .pid marker (e.g. from before
# this cleanup was added, or concurrent sweeps yet to write their marker).
_ORPHAN_BAKED_GRACE_SEC = 3600  # 1 hour


def _prune_orphan_baked_dirs() -> None:
    """Prune baked-adapter dirs whose owning process is no longer alive.

    Complements the ``finally`` block at the end of the rollout stage, which
    only runs on clean exit (not SIGKILL, OOM-kill, or hard vLLM crashes).
    On startup we sweep ``BAKED_ROOT`` and remove any dir whose ``.pid``
    marker points at a dead PID. Dirs with no marker are only removed if
    older than ``_ORPHAN_BAKED_GRACE_SEC``, so we never race a concurrent
    sweep that hasn't written its marker yet.
    """
    import time
    if not BAKED_ROOT.exists():
        return
    for d in BAKED_ROOT.iterdir():
        if not d.is_dir():
            continue
        pid_file = d / ".pid"
        stale = False
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                stale = True
        else:
            try:
                if time.time() - d.stat().st_mtime > _ORPHAN_BAKED_GRACE_SEC:
                    stale = True
            except OSError:
                continue
        if stale:
            try:
                print(f"[cleanup] pruning orphan baked dir: {d}")
                cleanup_baked_dir(d)
            except Exception as exc:
                print(f"[cleanup] failed to prune {d}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    return parse_sweep_flags(
        "Cell-oriented LLM-judge sweep runner.",
        extras=[
            ("--skip-rollouts", {"action": "store_true"}),
            ("--skip-judge", {"action": "store_true"}),
        ],
    )


# ---------------------------------------------------------------------------
# Config normalisation: unify legacy single-adapter + new combo shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedConfig:
    adapters: tuple[AdapterSpec, ...]
    scales_per_adapter: dict[str, tuple[float, ...]]  # keyed by AdapterSpec.slug
    eval_name: str
    base_model: str
    base_model_slug: str
    dataset_path: str
    max_samples: int
    seed: int
    num_rollouts_per_prompt: int
    assistant_temperature: float
    assistant_top_p: float
    assistant_max_new_tokens: int
    assistant_batch_size: int
    assistant_max_model_len: int | None
    assistant_gpu_memory_utilization: float | None
    assistant_enforce_eager: bool
    user_model: str
    user_provider: str
    judge_repeats: int
    judge_raters: tuple[Any, ...]
    judge_metric_traits: tuple[str, ...]
    judge_metric_coherence: str | None
    trait: Any | None
    ci_confidence: float
    ci_bootstrap_resamples: int
    plot_title: str
    trait_color: str
    coherence_color: str
    x_axis_label: str
    # Activation-capping mode: when set, the rollout stage applies
    # per-layer activation-capping hooks at the cell's "scale" (interpreted
    # as a fraction along the persona's axis) instead of baking a LoRA. The
    # baseline cell still runs as the unmodified base model. The cell-level
    # HF layout remains identical to the LoRA case.
    activation_cap_axis_path: str | None
    activation_cap_per_layer_range_path: str | None
    activation_cap_capping_layers: tuple[int, ...] | None


def _normalise_config(cfg: ModuleType) -> NormalisedConfig:
    if hasattr(cfg, "ADAPTERS"):
        adapters = tuple(cfg.ADAPTERS)
        scales_per_adapter = {
            a.slug: tuple(float(s) for s in cfg.SCALES_PER_ADAPTER[a.slug])
            for a in adapters
        }
    else:
        spec = AdapterSpec.from_ref(cfg.ADAPTER_REF)
        adapters = (spec,)
        scales_per_adapter = {spec.slug: tuple(float(s) for s in cfg.SCALE_POINTS)}

    # Trait metric(s): combo configs over two trait axes can request multiple
    # trait judges via JUDGE_METRIC_TRAITS (list). Singular JUDGE_METRIC_TRAIT
    # and TRAIT.v2_metric_name remain supported for single-adapter configs.
    if hasattr(cfg, "JUDGE_METRIC_TRAITS"):
        trait_metrics = tuple(cfg.JUDGE_METRIC_TRAITS)
    elif hasattr(cfg, "JUDGE_METRIC_TRAIT"):
        trait_metrics = (cfg.JUDGE_METRIC_TRAIT,)
    elif hasattr(cfg, "TRAIT"):
        trait_metrics = (cfg.TRAIT.v2_metric_name,)
    else:
        raise ValueError(
            "Config must set JUDGE_METRIC_TRAITS, JUDGE_METRIC_TRAIT, or TRAIT."
        )

    cap_cfg = getattr(cfg, "ACTIVATION_CAP_CONFIG", None)
    if cap_cfg is not None:
        cap_axis_path = cap_cfg["axis_path"]
        cap_per_layer = cap_cfg["per_layer_range_path"]
        cap_layers = tuple(int(l) for l in cap_cfg["capping_layers"])
        default_x_axis = "Activation cap fraction"
    else:
        cap_axis_path = None
        cap_per_layer = None
        cap_layers = None
        default_x_axis = "LoRA scale"

    return NormalisedConfig(
        adapters=adapters,
        scales_per_adapter=scales_per_adapter,
        eval_name=getattr(cfg, "EVAL_NAME_CANONICAL", EVAL_NAME_DEFAULT),
        base_model=cfg.BASE_MODEL,
        base_model_slug=cfg.BASE_MODEL_SLUG,
        dataset_path=cfg.DATASET_PATH,
        max_samples=cfg.MAX_SAMPLES,
        seed=cfg.SEED,
        num_rollouts_per_prompt=cfg.NUM_ROLLOUTS_PER_PROMPT,
        assistant_temperature=cfg.ASSISTANT_TEMPERATURE,
        assistant_top_p=cfg.ASSISTANT_TOP_P,
        assistant_max_new_tokens=cfg.ASSISTANT_MAX_NEW_TOKENS,
        assistant_batch_size=getattr(cfg, "ASSISTANT_BATCH_SIZE", 32),
        assistant_max_model_len=getattr(cfg, "ASSISTANT_MAX_MODEL_LEN", None),
        assistant_gpu_memory_utilization=getattr(cfg, "ASSISTANT_GPU_MEMORY_UTILIZATION", None),
        assistant_enforce_eager=bool(getattr(cfg, "ASSISTANT_ENFORCE_EAGER", False)),
        user_model=getattr(cfg, "USER_MODEL", "z-ai/glm-4.5-air:free"),
        user_provider=getattr(cfg, "USER_PROVIDER", "openrouter"),
        judge_repeats=cfg.JUDGE_REPEATS,
        judge_raters=tuple(cfg.JUDGE_RATERS),
        judge_metric_traits=trait_metrics,
        judge_metric_coherence=getattr(cfg, "COHERENCE_METRIC", None) or None,
        trait=getattr(cfg, "TRAIT", None),
        ci_confidence=cfg.CI_CONFIDENCE,
        ci_bootstrap_resamples=cfg.CI_BOOTSTRAP_RESAMPLES,
        plot_title=getattr(cfg, "PLOT_TITLE", "LLM-judge LoRA sweep"),
        trait_color=getattr(cfg, "TRAIT_COLOR", "#4A76AA"),
        coherence_color=getattr(cfg, "COHERENCE_COLOR", "#757575"),
        x_axis_label=getattr(cfg, "X_AXIS_LABEL", default_x_axis),
        activation_cap_axis_path=cap_axis_path,
        activation_cap_per_layer_range_path=cap_per_layer,
        activation_cap_capping_layers=cap_layers,
    )


def _rollout_params(nc: NormalisedConfig) -> dict[str, Any]:
    return dict(
        base_model=nc.base_model,
        dataset_path=nc.dataset_path,
        max_samples=nc.max_samples,
        seed=nc.seed,
        num_rollouts_per_prompt=nc.num_rollouts_per_prompt,
        assistant_temperature=nc.assistant_temperature,
        assistant_top_p=nc.assistant_top_p,
        assistant_max_new_tokens=nc.assistant_max_new_tokens,
    )


def _judge_metrics(nc: NormalisedConfig) -> list[str]:
    metrics = list(nc.judge_metric_traits)
    if nc.judge_metric_coherence:
        metrics.append(nc.judge_metric_coherence)
    return metrics


def _resolve_trait_for_metric(nc: NormalisedConfig, metric_name: str) -> Any:
    """Return the OceanTrait that pairs with ``metric_name`` for OceanJudgeRunConfig.

    v2 trait metrics follow the ``<trait>_v2`` convention so we parse the
    prefix. Coherence (and anything that doesn't match) reuses ``nc.trait`` if
    set, else the first configured trait metric — the trait field in
    OceanJudgeRunConfig only affects fingerprinting/system-prompt-naming, not
    the actual judge call (driven per-rater by ``metric_name``).
    """
    from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

    if metric_name in nc.judge_metric_traits and metric_name.endswith("_v2"):
        prefix = metric_name[: -len("_v2")]
        try:
            return OceanTrait(prefix)
        except ValueError:
            pass
    if nc.trait is not None:
        return nc.trait
    if nc.judge_metric_traits and nc.judge_metric_traits[0].endswith("_v2"):
        prefix = nc.judge_metric_traits[0][: -len("_v2")]
        try:
            return OceanTrait(prefix)
        except ValueError:
            pass
    raise ValueError(
        f"Could not resolve OceanTrait for metric {metric_name!r} — set TRAIT in config."
    )


def _required_judge_pairs(nc: NormalisedConfig) -> list[tuple[str, str]]:
    return [(r.rater_id, m) for r in nc.judge_raters for m in _judge_metrics(nc)]


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


def _enumerate_cells(nc: NormalisedConfig) -> list[CanonicalCell]:
    return _enumerate_cells_generic(nc.adapters, nc.scales_per_adapter)


# ---------------------------------------------------------------------------
# Rollout generation via run_sweep + staging → per-cell move
# ---------------------------------------------------------------------------


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy every file under ``src`` into ``dst`` (overwrite on conflict)."""
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _generate_rollouts(
    nc: NormalisedConfig,
    cells_to_rollout: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    *,
    sweep_id: str,
) -> None:
    """Bake every cell needing rollouts and run one combined sweep.

    Uses the existing :func:`run_sweep` machinery with a *staging*
    OutputPathConfig (``hf_repo=None`` so no unintended uploads) and then
    copies each cell's staging output into its canonical local dir.
    """
    from src_dev.rollout_generation.model_providers import (
        ActivationCapProvider,
        CellSpec,
        VLLMLoRaComboProvider,
    )
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

    is_capping = nc.activation_cap_axis_path is not None
    if is_capping:
        # Build one provider variant per cell; baseline cell uses fraction 0.0
        # (cap threshold at the base end → effectively a no-op, matching the
        # LoRA baseline's "unmodified base model" semantics).
        fractions: list[float] = []
        variant_label_map: dict[str, str] = {}
        for cell in cells_to_rollout:
            scale = cell.entries[0][1] if cell.entries else 0.0
            fractions.append(float(scale))
            variant_label_map[str(float(scale))] = cell.variant_label()

        provider = ActivationCapProvider(
            base_model=nc.base_model,
            axis_path=nc.activation_cap_axis_path,
            per_layer_range_path=nc.activation_cap_per_layer_range_path,
            fractions=fractions,
            capping_layers=list(nc.activation_cap_capping_layers or []),
            variant_label_map=variant_label_map,
        )
        assistant_provider_name = "local"
    else:
        cell_specs = [
            CellSpec(
                label=c.variant_label(),
                adapter_scales=tuple((spec.ref, scale) for spec, scale in c.entries),
            )
            for c in cells_to_rollout
        ]
        provider = VLLMLoRaComboProvider(
            base_model=nc.base_model,
            cells=cell_specs,
            baked_adapters_dir=BAKED_ROOT / sweep_id,
            temperature=nc.assistant_temperature,
            top_p=nc.assistant_top_p,
            max_new_tokens=nc.assistant_max_new_tokens,
            max_model_len=nc.assistant_max_model_len,
            gpu_memory_utilization=nc.assistant_gpu_memory_utilization,
            enforce_eager=nc.assistant_enforce_eager,
        )
        assistant_provider_name = "vllm"

    output_config = OutputPathConfig(
        scratch_root=staging_root,
        hf_repo=None,
        base_model="_staging",
        category="_sweep",
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
            assistant_model=nc.base_model,
            assistant_provider=assistant_provider_name,
            assistant_temperature=nc.assistant_temperature,
            assistant_top_p=nc.assistant_top_p,
            assistant_max_new_tokens=nc.assistant_max_new_tokens,
            assistant_batch_size=nc.assistant_batch_size,
            user_model=nc.user_model,
            user_provider=nc.user_provider,
            user_temperature=0.7,
            user_top_p=0.95,
            user_max_new_tokens=128,
            user_batch_size=32,
            user_max_concurrent=32,
            dataset_path=nc.dataset_path,
            max_samples=nc.max_samples,
            dataset_seed=nc.seed,
            num_rollouts=nc.num_rollouts_per_prompt,
            turns_per_phase=[1],
        ),
        output=output_config,
        skip_completed=True,
        skip_evals=True,
        on_cell_error="warn",
        max_concurrent_conditions=1,
        plot=False,
        metadata={
            "sweep_id": sweep_id,
            "n_cells": len(cells_to_rollout),
        },
    )
    run_sweep(sweep_config)

    staging_output = output_config.scratch_dir
    for cell in cells_to_rollout:
        src = staging_output / cell.variant_label() / condition_name
        dst = cell_dirs[cell]
        if not src.exists():
            print(f"  [rollout] staging src missing for cell {cell.variant_label()}: {src}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        _copy_tree(src, dst)


# ---------------------------------------------------------------------------
# Per-cell judge
# ---------------------------------------------------------------------------


def _build_judge_dataset_for_cell(
    cell: CanonicalCell,
    cell_dir: Path,
    *,
    assistant_model: str,
    out_path: Path,
) -> int:
    """Flatten one cell's rollouts into a judge-compatible JSONL dataset.

    Row shape matches ``OceanJudgeRunConfig``'s expectations. ``condition``
    is set to ``no_prompt@<cell_tag>`` so the raw judge output carries the
    cell tag for later demux (though here each judge invocation is per-cell,
    so demux is trivial).
    """
    rollouts_path = cell_dir / ROLLOUTS_RELPATH
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
                assistant_msgs = [
                    m for m in turn_list if m.get("role") == "assistant"
                ]
                if not assistant_msgs:
                    continue
                response = assistant_msgs[-1]["content"]
                response_id = f"{cell_tag}:{seed_id}:{rollout_idx}"
                out.write(json.dumps({
                    "response_id": response_id,
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
                    "assistant_model": assistant_model,
                    "assistant_provider": "local",
                    "system_prompt_ref": "",
                }) + "\n")
                n += 1
    return n


def _run_judge_for_cell_metric(
    nc: NormalisedConfig,
    cell: CanonicalCell,
    cell_dir: Path,
    metric_name: str,
) -> None:
    """Run the LLM judge for one (cell, metric) and write per-cell raw output.

    Invokes :func:`run_ocean_judge_run` on a transient per-cell dataset, then
    copies the raw judge-call JSONL files into
    ``cell_dir/judge_runs/{rater_id}/{metric_name}.jsonl``.
    """
    from src_dev.persona_metrics.llm_judge_agreement import (
        OceanJudgeRunConfig,
        get_judge_run_dir,
        run_ocean_judge_run,
    )

    transient_dir = cell_dir / "_judge_scratch" / metric_name
    transient_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = transient_dir / "all_responses.jsonl"
    n_rows = _build_judge_dataset_for_cell(
        cell, cell_dir, assistant_model=nc.base_model, out_path=dataset_path,
    )
    if n_rows == 0:
        print(f"  [judge] {cell.variant_label()} / {metric_name}: no rollouts, skipping")
        return

    raters = [
        r.model_copy(update={"metric_name": metric_name}) for r in nc.judge_raters
    ]
    metric_trait = _resolve_trait_for_metric(nc, metric_name)
    judge_cfg = OceanJudgeRunConfig(
        trait=metric_trait,
        dataset_path=dataset_path,
        judge_raters=raters,
        judge_repeats=nc.judge_repeats,
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

    # Copy raw calls per-rater into cell_dir/judge_runs/{rater}/{metric}.jsonl.
    # Raw files from OceanJudgeRunConfig are per-rater; filenames vary by
    # repeat index. Concatenate all repeats for each rater into one JSONL.
    for rater in nc.judge_raters:
        out_file = cell_dir / "judge_runs" / rater.rater_id / f"{metric_name}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as out:
            for raw_path in sorted(raw_dir.glob(f"{rater.rater_id}*.jsonl")):
                with raw_path.open() as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if line:
                            out.write(line + "\n")


# ---------------------------------------------------------------------------
# Aggregation (scale_summary for 1D, grid_summary for combo)
# ---------------------------------------------------------------------------


def _cell_scores(
    cell_dir: Path, rater_ids: list[str], metric_name: str
) -> list[float]:
    """Median per-response score across repeats within this cell/metric.

    Mirrors the per-response median in the old ``_scale_scores`` — one value
    per response per rater after collapsing repeats.
    """
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
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
                if not isinstance(score, int):
                    continue
                key = (rater_id, str(record.get("response_id", "")))
                grouped[key].append(score)
    return [float(statistics.median(v)) for v in grouped.values() if v]


def _ci95_from_bootstrap(
    values: list[float], seed: int, n_resamples: int, confidence: float
) -> tuple[float, float]:
    if len(values) <= 1:
        m = values[0] if values else math.nan
        return m, m
    import numpy as np
    from scipy import stats as scipy_stats

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    try:
        r = scipy_stats.bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence / 100,
            random_state=rng,
            method="BCa",
        )
        low = float(r.confidence_interval.low)
        high = float(r.confidence_interval.high)
    except Exception as exc:
        print(
            f"[warn] bootstrap CI failed ({type(exc).__name__}: {exc}); "
            "returning degenerate interval (mean, mean)",
            flush=True,
        )
        m = float(arr.mean())
        return m, m
    if not (math.isfinite(low) and math.isfinite(high)):
        m = float(arr.mean())
        return m, m
    return low, high


def _summary_row(
    nc: NormalisedConfig, cell: CanonicalCell, metric_name: str, values: list[float]
) -> dict[str, Any]:
    ci_method = f"ci{nc.ci_confidence:g}_from_bootstrap_{nc.ci_bootstrap_resamples}"
    base: dict[str, Any] = {
        "metric": metric_name,
        "cell_tag": cell.variant_label(),
        "cell_entries": [
            {"slug": s.slug, "scale": sc} for s, sc in cell.entries
        ],
        "tier": cell.tier,
    }
    if not values:
        return {
            **base,
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "ci_method": ci_method,
        }
    lo, hi = _ci95_from_bootstrap(
        values, nc.seed, nc.ci_bootstrap_resamples, nc.ci_confidence
    )
    return {
        **base,
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_method": ci_method,
    }


def _aggregate(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> Path:
    """Walk all cells' judge outputs; write aggregated summary JSONL."""
    rater_ids = [r.rater_id for r in nc.judge_raters]
    metrics = _judge_metrics(nc)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for metric in metrics:
            values = _cell_scores(cell_dirs[cell], rater_ids, metric)
            rows.append(_summary_row(nc, cell, metric, values))
    out_path = sweep_root / "analysis" / "grid_summary.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote summary: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plotting (1D for single adapter, 2D heatmaps for 2-adapter combo)
# ---------------------------------------------------------------------------


def _plot_1d(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping.")
        return

    rater_ids = [r.rater_id for r in nc.judge_raters]
    coherence_metric = nc.judge_metric_coherence

    adapter = nc.adapters[0]
    scales = sorted(nc.scales_per_adapter[adapter.slug])
    # Map scale -> cell
    cell_by_scale: dict[float, CanonicalCell] = {}
    for cell in cells:
        if not cell.entries:
            cell_by_scale[0.0] = cell
        else:
            cell_by_scale[cell.entries[0][1]] = cell

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for trait_metric in nc.judge_metric_traits:
        plot_metrics = [trait_metric] + ([coherence_metric] if coherence_metric else [])
        metric_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in plot_metrics}
        for metric in plot_metrics:
            for scale in scales:
                cell = cell_by_scale.get(scale) or CanonicalCell.from_scales([(adapter, scale)])
                values = _cell_scores(cell_dirs[cell], rater_ids, metric) \
                    if cell in cell_dirs else []
                metric_rows[metric].append({"scale": scale, **_summary_row(nc, cell, metric, values)})

        _render_1d_figure(nc, trait_metric, coherence_metric, metric_rows, plots_dir, plt)


def _render_1d_figure(
    nc: NormalisedConfig,
    trait_metric: str,
    coherence_metric: str | None,
    metric_rows: dict[str, list[dict[str, Any]]],
    plots_dir: Path,
    plt: Any,
) -> None:
    fig, left = plt.subplots(figsize=(7.0, 3.5))
    metric_axes: dict[str, tuple[Any, str, str]] = {
        trait_metric: (left, nc.trait_color, "Trait"),
    }
    if coherence_metric:
        right = left.twinx()
        metric_axes[coherence_metric] = (right, nc.coherence_color, "Coherence")
    lines = []
    for metric in metric_axes:
        rows = metric_rows[metric]
        ax, color, label = metric_axes[metric]
        xs = [r["scale"] for r in rows]
        ys = [r["mean"] for r in rows]
        (ln,) = ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
        yerr = [
            [max(0.0, r["mean"] - r["ci_lower"]) for r in rows],
            [max(0.0, r["ci_upper"] - r["mean"]) for r in rows],
        ]
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", color=color,
                    capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.75, zorder=5)
        ax.set_ylabel(f"{label} mean judge score", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        if metric == trait_metric:
            ax.set_ylim(-4, 4)
        else:
            ax.set_ylim(0, 10)
        lines.append(ln)
    left.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    left.set_title(f"{nc.plot_title} — {trait_metric}")
    left.set_xlabel(nc.x_axis_label)
    left.grid(alpha=0.25)
    if lines:
        left.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()

    suffix = "" if len(nc.judge_metric_traits) == 1 else f"_{trait_metric}"
    out = plots_dir / f"llm_judge_scale_sweep{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out}")


def _plot_2d(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib/numpy not available; skipping 2D plots.")
        return

    rater_ids = [r.rater_id for r in nc.judge_raters]
    a, b = nc.adapters[0], nc.adapters[1]
    xs = sorted(nc.scales_per_adapter[a.slug])
    ys = sorted(nc.scales_per_adapter[b.slug])

    # Index canonical cells by (scale_a, scale_b), with zero-dropping aware lookup.
    def find_cell(sa: float, sb: float) -> CanonicalCell:
        return CanonicalCell.from_scales([(a, sa), (b, sb)])

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in _judge_metrics(nc):
        mat = np.full((len(ys), len(xs)), math.nan, dtype=float)
        for i, sa in enumerate(xs):
            for j, sb in enumerate(ys):
                cell = find_cell(sa, sb)
                values = (
                    _cell_scores(cell_dirs[cell], rater_ids, metric)
                    if cell in cell_dirs
                    else []
                )
                if values:
                    mat[j, i] = statistics.mean(values)

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=(xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5),
            vmin=-4.0,
            vmax=4.0,
        )
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.set_xlabel(f"{a.slug} scale")
        ax.set_ylabel(f"{b.slug} scale")
        ax.set_title(f"{metric} — {nc.plot_title}")
        fig.colorbar(im, ax=ax)
        # Overlay numeric values
        for i, sa in enumerate(xs):
            for j, sb in enumerate(ys):
                if not math.isnan(mat[j, i]):
                    ax.text(sa, sb, f"{mat[j, i]:.1f}",
                            ha="center", va="center", color="white", fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"heatmap_{metric}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {out}")


def _make_plots(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    sweep_root: Path,
) -> None:
    n = len(nc.adapters)
    if n == 1:
        _plot_1d(nc, cells, cell_dirs, sweep_root)
    elif n == 2:
        _plot_2d(nc, cells, cell_dirs, sweep_root)
    else:
        print(f"[plot] N={n} adapters — JSON grid_summary only, no plot.")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


_UPLOAD_RETRY_ATTEMPTS = 4
_UPLOAD_RETRY_BASE_DELAY_SEC = 10.0
_UPLOAD_TRANSIENT_MARKERS = (
    "500 Server Error",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "504 Gateway Timeout",
    "Internal Error",
    "Gateway Time-out",
    "429",
    "Too Many Requests",
    "Connection reset",
    "Connection aborted",
    "Connection refused",
    "Read timed out",
    "Temporary failure",
)


def _is_transient_upload_error(exc: BaseException) -> bool:
    """Return True if an HF upload exception looks retryable (5xx, 429, network)."""
    msg = str(exc)
    type_name = type(exc).__name__
    if any(marker in msg for marker in _UPLOAD_TRANSIENT_MARKERS):
        return True
    if "Timeout" in type_name or "Connection" in type_name:
        return True
    return False


def _with_upload_retry(description: str, fn):
    """Retry an HF upload with exponential backoff on transient server errors.

    Per-sweep uploads can occasionally hit 500/503 hiccups from the HF hub;
    letting those fail the whole sweep wastes hours of compute. We retry up
    to ``_UPLOAD_RETRY_ATTEMPTS`` times, doubling the delay each time.
    Non-transient errors (4xx auth, malformed request, etc.) surface immediately.
    """
    if _SKIP_UPLOAD:
        return None
    for attempt in range(1, _UPLOAD_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except BaseException as exc:
            if not _is_transient_upload_error(exc) or attempt >= _UPLOAD_RETRY_ATTEMPTS:
                raise
            delay = _UPLOAD_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(
                f"[upload-retry] {description}: attempt {attempt}/{_UPLOAD_RETRY_ATTEMPTS} "
                f"failed with {type(exc).__name__}: {str(exc)[:200]}; retrying in {delay:.0f}s"
            )
            time.sleep(delay)


def _upload_cells(
    nc: NormalisedConfig,
    cells: list[CanonicalCell],
    cell_dirs: dict[CanonicalCell, Path],
    fingerprint: str,
) -> None:
    for cell in cells:
        write_cell_info(cell, cell_dirs[cell], fingerprint)
        _with_upload_retry(
            f"upload_cell {cell.variant_label()}",
            lambda cell=cell: upload_cell(
                cell,
                local_dir=cell_dirs[cell],
                model_slug=nc.base_model_slug,
                eval_name=nc.eval_name,
                fingerprint=fingerprint,
                repo_id=HF_REPO_ID,
                commit_message=f"{nc.eval_name}: upload cell {cell.variant_label()}",
                allow_patterns=[
                    "rollouts/**",
                    "judge_runs/**",
                    "cell_info.json",
                ],
            ),
        )


def _upload_sweep_root(
    nc: NormalisedConfig,
    sweep_root: Path,
    fingerprint: str,
) -> None:
    hf_path = sweep_hf_root(
        list(nc.adapters),
        model_slug=nc.base_model_slug,
        eval_name=nc.eval_name,
        fingerprint=fingerprint,
    )
    _with_upload_retry(
        f"upload_sweep_root {hf_path}",
        lambda: _upload_sweep_root_generic(
            sweep_root,
            hf_path=hf_path,
            repo_id=HF_REPO_ID,
            commit_message=f"{nc.eval_name}: upload sweep analysis + plots",
            allow_patterns=["plots/**", "analysis/**", "sweep.log", "sweep_config.json"],
        ),
    )
    print(f"  [upload] sweep root → {HF_REPO_ID}/{hf_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _print_dry_run(nc: NormalisedConfig, cells: list[CanonicalCell], fingerprint: str) -> None:
    print("DRY RUN: Cell-oriented LLM-judge sweep")
    print(f"  eval name       : {nc.eval_name}")
    print(f"  base model      : {nc.base_model} ({nc.base_model_slug})")
    print(f"  adapters        : {[a.slug for a in nc.adapters]}")
    for a in nc.adapters:
        print(f"    {a.slug}: scales={list(nc.scales_per_adapter[a.slug])}")
    print(f"  cells           : {len(cells)} canonical cells")
    for cell in cells:
        print(f"    [{cell.tier}] {cell.variant_label()}")
    print(f"  fingerprint     : {fingerprint}")
    print(f"  judge raters    : {[r.rater_id for r in nc.judge_raters]}")
    print(f"  judge metrics   : {_judge_metrics(nc)}")
    print(f"  judge repeats   : {nc.judge_repeats}")
    sweep_hf = sweep_hf_root(
        list(nc.adapters), model_slug=nc.base_model_slug,
        eval_name=nc.eval_name, fingerprint=fingerprint,
    )
    print(f"  sweep HF root   : {sweep_hf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    flags = _parse_flags()
    cfg = load_config_module(flags.config)

    diffs = check_sweep_defaults(cfg)
    confirm_or_abort(diffs, allow_custom=flags.allow_custom_fingerprint)

    nc = _normalise_config(cfg)
    seed_all(nc.seed)
    load_dotenv()
    upload = not flags.no_upload
    if upload:
        login_from_env()

    fingerprint = rollout_fingerprint(**_rollout_params(nc))
    cells = _enumerate_cells(nc)
    required_pairs = _required_judge_pairs(nc)

    if flags.dry_run:
        _print_dry_run(nc, cells, fingerprint)
        return

    _prune_orphan_baked_dirs()

    # Stage 1: hydrate every cell from HF (a no-op for brand-new cells).
    cell_dirs: dict[CanonicalCell, Path] = {}
    cell_status: dict[CanonicalCell, Any] = {}
    for cell in cells:
        local_dir, status = hydrate_cell(
            cell,
            scratch_root=SCRATCH_ROOT,
            model_slug=nc.base_model_slug,
            eval_name=nc.eval_name,
            fingerprint=fingerprint,
            repo_id=HF_REPO_ID,
            required_judge_metrics=required_pairs,
            skip_download=not upload,
        )
        cell_dirs[cell] = local_dir
        cell_status[cell] = status
    n_with_rollouts = sum(1 for s in cell_status.values() if s.has_rollouts)
    print(f"[hydrate] {n_with_rollouts}/{len(cells)} cells already have rollouts on HF")

    # Stages 2+3 (concurrent): rollouts on GPU, judges via remote API.
    #
    # The rollout loop runs in a background thread, generating one cell at a
    # time inside ``_generate_rollouts``. A judge-worker thread consumes a
    # queue of cells. The main thread polls each in-flight cell's local dir
    # and enqueues it for judging the moment its ``rollouts.jsonl`` lands —
    # so API-bound judging overlaps with GPU-bound rollout generation of
    # later cells. Within a single cell, trait + coherence metrics run in
    # parallel threads (disjoint output files; cache fingerprint unaffected).
    # Each cell uploads to HF right after its metrics finish, so a mid-run
    # crash preserves all already-judged cells on HF.
    cells_to_rollout = [
        c for c in cells if not cell_status[c].has_rollouts
    ]
    # Baseline cells have no adapters → rollouts still need generation, but
    # VLLMLoRaComboProvider supports empty adapter_scales cells.

    def _judge_and_upload(cell: CanonicalCell) -> None:
        """Judge pending metrics for a cell (in parallel threads), then upload."""
        cell_dir = cell_dirs[cell]
        status = cell_status_on_disk(cell_dir, required_judge_metrics=required_pairs)
        pending = [
            metric for metric in _judge_metrics(nc)
            if not all(
                (rater.rater_id, metric) in status.present_judge_metrics
                for rater in nc.judge_raters
            )
        ]
        if pending:
            print(f"[judge] {cell.variant_label()} / {pending} (parallel)")
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = [
                    pool.submit(
                        _run_judge_for_cell_metric,
                        nc, cell, cell_dir, metric,
                    )
                    for metric in pending
                ]
                for fut in as_completed(futures):
                    fut.result()
        cell_status[cell] = cell_status_on_disk(
            cell_dir, required_judge_metrics=required_pairs,
        )
        if upload and pending:
            write_cell_info(cell, cell_dir, fingerprint)
            if _BATCH_UPLOAD:
                # Skip per-cell upload — Stage 6 batches the whole sweep into
                # a single commit so we stay under HF's commits-per-hour cap.
                pass
            else:
                print(f"[upload] {cell.variant_label()}")
                _with_upload_retry(
                    f"upload_cell {cell.variant_label()}",
                    lambda: upload_cell(
                        cell,
                        local_dir=cell_dir,
                        model_slug=nc.base_model_slug,
                        eval_name=nc.eval_name,
                        fingerprint=fingerprint,
                        repo_id=HF_REPO_ID,
                        commit_message=f"{nc.eval_name}: upload cell {cell.variant_label()}",
                        allow_patterns=[
                            "rollouts/**",
                            "judge_runs/**",
                            "cell_info.json",
                        ],
                    ),
                )

    _JUDGE_SENTINEL = object()
    judge_queue: Queue = Queue()
    judge_errors: list[tuple[CanonicalCell, BaseException]] = []

    def _judge_worker() -> None:
        while True:
            item = judge_queue.get()
            if item is _JUDGE_SENTINEL:
                return
            try:
                _judge_and_upload(item)
            except BaseException as exc:
                print(
                    f"[judge] ERROR on {item.variant_label()}: "
                    f"{type(exc).__name__}: {exc}"
                )
                judge_errors.append((item, exc))

    judge_thread: threading.Thread | None = None
    if not flags.skip_judge:
        judge_thread = threading.Thread(
            target=_judge_worker, name="judge-worker", daemon=False,
        )
        judge_thread.start()

    # Enqueue cells already hydrated from HF (no rollout needed).
    for cell in cells:
        if cell in cells_to_rollout:
            continue
        if cell_status[cell].has_rollouts and judge_thread is not None:
            judge_queue.put(cell)

    rollout_error: list[BaseException] = []
    rollout_thread: threading.Thread | None = None
    if cells_to_rollout and not flags.skip_rollouts:
        sweep_id = f"{nc.eval_name}_{fingerprint}_{uuid.uuid4().hex[:8]}"
        baked_dir = BAKED_ROOT / sweep_id
        baked_dir.mkdir(parents=True, exist_ok=True)
        (baked_dir / ".pid").write_text(str(os.getpid()))
        print(f"[rollout] generating rollouts for {len(cells_to_rollout)} cell(s); sweep_id={sweep_id}")

        def _rollout_worker() -> None:
            try:
                _generate_rollouts(nc, cells_to_rollout, cell_dirs, sweep_id=sweep_id)
            except BaseException as exc:
                rollout_error.append(exc)
            finally:
                # Combo bakes are uuid-suffixed and never reused across runs;
                # they can each be tens of GB, so remove regardless of outcome.
                try:
                    cleanup_baked_dir(baked_dir)
                    print(f"[cleanup] removed baked combo dir: {baked_dir}")
                except Exception as exc:
                    print(f"[cleanup] failed to remove {baked_dir}: {exc}")

        rollout_thread = threading.Thread(
            target=_rollout_worker, name="rollout-worker", daemon=False,
        )
        rollout_thread.start()

        # Main thread: poll for per-cell rollout completion, enqueue judge work.
        POLL_INTERVAL_SEC = 10.0
        remaining = set(cells_to_rollout)
        while remaining:
            done_this_round = []
            for cell in remaining:
                status = cell_status_on_disk(
                    cell_dirs[cell], required_judge_metrics=required_pairs,
                )
                if status.has_rollouts:
                    cell_status[cell] = status
                    if judge_thread is not None:
                        judge_queue.put(cell)
                        print(f"[judge] enqueued {cell.variant_label()}")
                    done_this_round.append(cell)
            for cell in done_this_round:
                remaining.discard(cell)
            if not remaining:
                break
            if not rollout_thread.is_alive():
                # Rollout thread exited but some cells didn't finish (likely error).
                break
            time.sleep(POLL_INTERVAL_SEC)

        rollout_thread.join()
    elif cells_to_rollout:
        print(f"[rollout] --skip-rollouts set; {len(cells_to_rollout)} cell(s) will be skipped")

    # Drain judge queue; signal worker to exit.
    if judge_thread is not None:
        judge_queue.put(_JUDGE_SENTINEL)
        judge_thread.join()

    # Surface errors after both threads have cleanly stopped.
    if rollout_error:
        raise rollout_error[0]
    if judge_errors:
        errs = "\n".join(
            f"  {c.variant_label()}: {type(e).__name__}: {e}"
            for c, e in judge_errors
        )
        raise RuntimeError(
            f"Judge stage had {len(judge_errors)} failure(s):\n{errs}"
        )

    # Stage 4: aggregate.
    sweep_root = SCRATCH_ROOT / sweep_hf_root(
        list(nc.adapters),
        model_slug=nc.base_model_slug,
        eval_name=nc.eval_name,
        fingerprint=fingerprint,
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    _aggregate(nc, cells, cell_dirs, sweep_root)

    # Stage 5: plots.
    _make_plots(nc, cells, cell_dirs, sweep_root)

    # Stage 6: upload.
    if upload:
        if _BATCH_UPLOAD:
            # Single upload per sweep — big allow_patterns covers all scale
            # cells + plots + analysis in one HF commit. Baseline is still
            # its own commit (different parent path), handled separately
            # below. Keeps us well under the commits-per-hour rate limit
            # for long runs.
            sweep_hf_path = sweep_hf_root(
                list(nc.adapters),
                model_slug=nc.base_model_slug,
                eval_name=nc.eval_name,
                fingerprint=fingerprint,
            )
            print(f"[upload-batch] sweep → {HF_REPO_ID}/{sweep_hf_path}")
            _with_upload_retry(
                f"upload_sweep_batch {sweep_hf_path}",
                lambda: _upload_sweep_root_generic(
                    sweep_root,
                    hf_path=sweep_hf_path,
                    repo_id=HF_REPO_ID,
                    commit_message=f"{nc.eval_name}: batched sweep upload",
                    allow_patterns=[
                        "scale_*/rollouts/**",
                        "scale_*/judge_runs/**",
                        "scale_*/cell_info.json",
                        "cell_*/rollouts/**",
                        "cell_*/judge_runs/**",
                        "cell_*/cell_info.json",
                        "plots/**",
                        "analysis/**",
                        "sweep.log",
                        "sweep_config.json",
                    ],
                ),
            )
            # Baseline (if written this sweep) lives at a different parent
            # path. Upload it with _upload_cells over just the baseline cells.
            baseline_cells = [c for c in cells if c.tier == "baseline"]
            if baseline_cells:
                _upload_cells(nc, baseline_cells, cell_dirs, fingerprint)
        else:
            _upload_cells(nc, cells, cell_dirs, fingerprint)
            _upload_sweep_root(nc, sweep_root, fingerprint)

    print(f"Done. sweep_root={sweep_root}")


if __name__ == "__main__":
    main()

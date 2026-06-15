"""Stage 1 — rollout generation.

Generates diverse multi-turn conversations to produce a population of LLM
personas. This stage handles the cache / HF-sync / upload plumbing and then
delegates the actual generation to
:func:`src_dev.rollout_generation.run.run_rollout_generation`.

Because the user-simulator mode (``scenarios`` / ``archetypes`` / ``legacy``)
drives a lot of script-specific preamble — loading scenarios, writing a
synthetic seed JSONL, registering user-simulator templates, composing per-
sample templates — this stage takes a **callable** that builds the
:class:`RolloutGenerationConfig` after a cache miss. The experiment script
provides that callable; the stage calls it only when generation is required.

Signature of the callable::

    build_rollout_config(run_dir, retry_terminal_sample_ids) -> RolloutGenerationConfig

That keeps scenario / archetype / user-simulator-prompt specifics in the
script (next to the modules they depend on) and leaves the generic
cache/HF/upload flow here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src_dev.psychometric.config import (
    RolloutsStageConfig,
    RolloutStageResult,
)
from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.rollout_generation.config import RolloutGenerationConfig
from src_dev.rollout_generation.run import run_rollout_generation
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


BuildRolloutConfigFn = Callable[[Path, list[str]], RolloutGenerationConfig]


def run_stage_rollouts(
    cfg: RolloutsStageConfig,
    *,
    build_rollout_config: BuildRolloutConfigFn,
) -> RolloutStageResult:
    """Generate rollouts (or return the cached run directory).

    Args:
        cfg: Rollouts stage config. ``cfg.retry_terminal_sample_ids`` — when
            non-empty — bypasses the cache short-circuit and regenerates
            just those sample IDs.
        build_rollout_config: Script-provided callable that returns the full
            :class:`RolloutGenerationConfig` given ``(run_dir, retry_terminal
            _sample_ids)``. Only called on a cache miss.

    Returns:
        :class:`RolloutStageResult` with the run directory and whether it
        was hydrated / generated.
    """
    retry_terminal_sample_ids = list(cfg.retry_terminal_sample_ids or [])
    run_dir = cfg.ctx.rollout_dir
    run_id = cfg.ctx.rollout_run_id
    hf_repo_id = cfg.ctx.hf_repo_id

    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")
    hf_path = hf_runs_path(run_id, model_slug=cfg.ctx.model_slug or None)

    # Check local cache. Require both the export AND manifest.json — a dir
    # missing manifest.json is a partially-hydrated cache (older code paths
    # only pulled exports), and downstream consumers (e.g.
    # find_consecutive_assistant_turn_sample_ids → load_samples) need the
    # manifest. Falling through here lets the HF branch below re-hydrate.
    rollout_export = run_dir / "exports" / "conversation_training.jsonl"
    manifest_path = run_dir / "manifest.json"
    if rollout_export.exists() and manifest_path.exists() and not retry_terminal_sample_ids:
        if not check_exists_in_dataset_repo(
            repo_id=hf_repo_id,
            path_in_repo=hf_path + "/exports/conversation_training.jsonl",
        ):
            print(f"[Stage 1] Local rollouts found but not on HF — uploading now")
            try:
                upload_folder_to_dataset_repo(
                    local_dir=run_dir,
                    repo_id=hf_repo_id,
                    path_in_repo=hf_path,
                    commit_message=f"Rollouts: {run_id}",
                )
                print(f"[Stage 1] Uploaded to HF: {hf_path}")
            except Exception as e:
                logger.warning("Failed to upload rollouts to HF: %s", e)
        print(f"[Stage 1] Rollouts already exist locally: {run_dir}")
        return RolloutStageResult(rollout_dir=run_dir)

    # Check HF cache
    if (
        check_exists_in_dataset_repo(repo_id=hf_repo_id, path_in_repo=hf_path)
        and not retry_terminal_sample_ids
    ):
        print(f"[Stage 1] Hydrating rollouts from HF: {run_id}")
        hydrate_dataset_subtree(
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            local_dir=run_dir,
        )
        if rollout_export.exists():
            print(f"[Stage 1] Hydrated rollouts from HF: {run_dir}")
            return RolloutStageResult(rollout_dir=run_dir, hydrated_from_hf=True)
        print("[Stage 1] HF hydration incomplete, regenerating...")

    # Cache miss — build the generation config via the caller's callback.
    rollout_config = build_rollout_config(run_dir, retry_terminal_sample_ids)

    print(
        f"[Stage 1] Generating {cfg.max_prompts} rollouts with "
        f"{cfg.num_conversation_turns} turns each..."
    )
    if retry_terminal_sample_ids:
        print(
            f"[Stage 1] Retry-terminal mode enabled for {len(retry_terminal_sample_ids)} sample(s)"
        )
    _dataset, result = run_rollout_generation(rollout_config)
    print(
        f"[Stage 1] Complete: {result.num_completed}/{result.num_conversations} rollouts, "
        f"{result.num_failed} failed"
    )

    # Upload to HF
    try:
        upload_folder_to_dataset_repo(
            local_dir=run_dir,
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            commit_message=f"Rollouts: {run_id}",
        )
        print(f"[Stage 1] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload rollouts to HF: %s", e)

    export_path = run_dir / "exports" / "conversation_training.jsonl"
    if export_path.exists():
        print(f"[Stage 1] View rollouts:")
        print(
            f"  uv run python -m src_dev.jsonl_tui.cli {export_path} "
            f"--conversation-field messages"
        )

    return RolloutStageResult(
        rollout_dir=run_dir,
        num_samples=int(getattr(result, "num_completed", 0)) or None,
        generated=True,
    )

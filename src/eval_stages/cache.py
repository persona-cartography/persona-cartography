"""Stage-level caching with local filesystem and HuggingFace Hub hydration.

Implements the cache-lookup flow proven in the Bloom eval script:

    1. Check local cache directory for a completion marker
    2. Attempt to download from HuggingFace Hub
    3. Run the stage if neither cache hit
    4. Write completion marker with full config for reproducibility
    5. Upload results to HuggingFace Hub
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _git_commit_hash() -> str | None:
    """Return the current short git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class StageCacheConfig:
    """Configuration for stage-level caching and HF hydration.

    Args:
        cache_root: Local root directory for cached stage outputs.
        hf_repo: HuggingFace dataset repo ID (e.g. ``persona-cartography/monorepo``).
            Set to ``None`` to disable HF sync entirely.
        hf_base_path: Prefix path within the HF repo for this eval's artifacts
            (e.g. ``evals/llm-judge-sweep/conscientiousness-suppressor``).
        no_remote: If True, disable all HF interaction (both download and upload).
            Equivalent to local-only mode.
    """

    cache_root: Path
    hf_repo: str | None = None
    hf_base_path: str = ""
    no_remote: bool = False


class StageCache:
    """Per-stage caching with local → HF → run fallback.

    Each stage is identified by a ``(stage_name, run_id)`` pair.  Results are
    stored under ``{cache_root}/{hf_base_path}/{stage}/{run_id}/``.

    Usage::

        cache = StageCache(StageCacheConfig(
            cache_root=Path("scratch/eval-cache"),
            hf_repo="persona-cartography/monorepo",
            hf_base_path="evals/bloom",
        ))

        def do_rollout():
            out = cache.stage_dir("rollout", rollout_id)
            out.mkdir(parents=True, exist_ok=True)
            # ... write results into out/ ...

        result_dir = cache.run_or_hydrate(
            "rollout", rollout_id, do_rollout,
            config={"model": "llama", "seed": 42},
        )
    """

    def __init__(self, config: StageCacheConfig) -> None:
        self._cfg = config

    @property
    def hf_repo(self) -> str | None:
        """HF dataset repo id (``None`` iff remote interaction is disabled)."""
        if self._cfg.no_remote:
            return None
        return self._cfg.hf_repo

    @property
    def hf_base_path(self) -> str:
        """HF base path prefix for this cache's artifacts."""
        return self._cfg.hf_base_path

    @property
    def has_remote(self) -> bool:
        """True iff HF interactions (download/upload) are enabled."""
        return bool(self._cfg.hf_repo) and not self._cfg.no_remote

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def stage_dir(self, stage: str, run_id: str) -> Path:
        """Local directory for a stage's cached output."""
        if self._cfg.hf_base_path:
            return self._cfg.cache_root / self._cfg.hf_base_path / stage / run_id
        return self._cfg.cache_root / stage / run_id

    def hf_path(self, stage: str, run_id: str) -> str:
        """Remote path within the HF repo for a stage's output."""
        if self._cfg.hf_base_path:
            return f"{self._cfg.hf_base_path}/{stage}/{run_id}"
        return f"{stage}/{run_id}"

    # ------------------------------------------------------------------
    # Cache checks
    # ------------------------------------------------------------------

    def is_complete(self, stage: str, run_id: str, marker: str = "done.json") -> bool:
        """Check if a stage has a completion marker in the local cache."""
        return (self.stage_dir(stage, run_id) / marker).exists()

    def try_hydrate(self, stage: str, run_id: str, marker: str = "done.json") -> bool:
        """Try to obtain stage results from local cache or HuggingFace.

        Returns True if the stage is available locally after this call.
        """
        # 1. Local cache hit
        if self.is_complete(stage, run_id, marker):
            return True

        # 2. HF download attempt
        if self._cfg.no_remote or not self._cfg.hf_repo:
            return False

        return self._fetch_from_hf(stage, run_id, marker)

    # ------------------------------------------------------------------
    # Write & upload
    # ------------------------------------------------------------------

    def mark_complete(
        self,
        stage: str,
        run_id: str,
        *,
        config: dict[str, Any],
        parent_run_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        marker: str = "done.json",
    ) -> Path:
        """Write a completion marker with full provenance metadata.

        The marker file records everything needed to reproduce or audit
        the stage: the config that produced it, the run ID chain, the
        git commit, and a timestamp.

        Returns:
            Path to the written marker file.
        """
        marker_path = self.stage_dir(stage, run_id) / marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "stage": stage,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_hash(),
            "config": config,
        }
        if parent_run_id is not None:
            payload["parent_run_id"] = parent_run_id
        if extra_metadata:
            payload["extra"] = extra_metadata

        marker_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return marker_path

    def upload(self, stage: str, run_id: str, commit_message: str) -> None:
        """Upload a stage directory to the HuggingFace monorepo."""
        if self._cfg.no_remote or not self._cfg.hf_repo:
            return

        from src.utils.hf_hub import upload_folder_to_dataset_repo

        local = self.stage_dir(stage, run_id)
        if not local.exists():
            logger.warning("Stage dir does not exist, skipping upload: %s", local)
            return

        upload_folder_to_dataset_repo(
            local_dir=local,
            repo_id=self._cfg.hf_repo,
            path_in_repo=self.hf_path(stage, run_id),
            commit_message=commit_message,
        )

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------

    def run_or_hydrate(
        self,
        stage: str,
        run_id: str,
        run_fn: Callable[[], None],
        *,
        config: dict[str, Any],
        parent_run_id: str | None = None,
        marker: str = "done.json",
        commit_message: str | None = None,
    ) -> Path:
        """Run a stage, or reuse cached results if available.

        Flow:
            1. Check local cache / download from HF
            2. If cache miss: call ``run_fn()`` (which should write output
               into ``self.stage_dir(stage, run_id)``)
            3. Write completion marker with config provenance
            4. Upload to HF (unless ``no_remote``)

        Args:
            stage: Stage name (used for directory structure and logging).
            run_id: Deterministic content-addressed run ID.
            run_fn: Zero-argument callable that executes the stage.  It should
                write its output into ``self.stage_dir(stage, run_id)``.
            config: Config dict stored in the completion marker for
                reproducibility.  Should match the dict used to compute
                ``run_id``.
            parent_run_id: Upstream stage's run ID, recorded in the marker
                for dependency tracing.
            marker: Filename of the completion marker (default ``done.json``).
            commit_message: HF upload commit message.  Auto-generated if None.

        Returns:
            Path to the stage output directory.
        """
        sdir = self.stage_dir(stage, run_id)
        tag = f"{stage} (run_id={run_id})"

        print(f"-- {tag} --")

        if self.try_hydrate(stage, run_id, marker):
            print(f"  Cache hit -> reusing {sdir}")
            return sdir

        print("  Cache miss -> running")
        sdir.mkdir(parents=True, exist_ok=True)
        run_fn()

        self.mark_complete(
            stage,
            run_id,
            config=config,
            parent_run_id=parent_run_id,
            marker=marker,
        )

        msg = commit_message or f"{stage} {run_id}"
        self.upload(stage, run_id, msg)

        return sdir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_hf(
        self, stage: str, run_id: str, marker: str = "done.json"
    ) -> bool:
        """Attempt to download a stage from HuggingFace Hub.

        Returns True if the download succeeded and the marker file is present.
        """
        if not self._cfg.hf_repo:
            return False

        from src.utils.hf_hub import (
            check_exists_in_dataset_repo,
            download_path_to_dir,
        )

        remote = self.hf_path(stage, run_id)

        # Quick existence check before downloading
        try:
            exists = check_exists_in_dataset_repo(
                repo_id=self._cfg.hf_repo,
                path_in_repo=remote,
            )
        except Exception:
            logger.debug("HF existence check failed for %s", remote, exc_info=True)
            return False

        if not exists:
            return False

        # Download into local cache
        try:
            download_path_to_dir(
                repo_id=self._cfg.hf_repo,
                path_in_repo=remote,
                target_dir=self.stage_dir(stage, run_id),
            )
        except Exception:
            logger.debug("HF download failed for %s", remote, exc_info=True)
            return False

        return self.is_complete(stage, run_id, marker)

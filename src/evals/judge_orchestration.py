"""Judge execution orchestration for Inspect custom evals."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai import score as inspect_score
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import Model
from inspect_ai.scorer import Scorer

from src.evals.config import JudgeExecutionConfig


def configure_inspect_paths(native_dir: Path) -> None:
    """Route Inspect data/cache writes into workspace-accessible directories."""
    data_home = native_dir / "inspect_data"
    cache_home = native_dir / "inspect_cache"
    data_home.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_DATA_HOME"] = str(data_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_home)


def _resolve_batch_setting(judge_exec: JudgeExecutionConfig) -> bool | int | dict[str, Any] | None:
    if judge_exec.inspect_batch is not None:
        return judge_exec.inspect_batch
    if judge_exec.prefer_batch:
        return True
    return None


def _wait_for_path(
    *,
    path: Path,
    label: str,
    poll_interval_seconds: int,
    timeout_seconds: int | None,
) -> None:
    deadline = (
        time.monotonic() + timeout_seconds
        if timeout_seconds is not None
        else None
    )
    interval = max(1, poll_interval_seconds)

    while not path.exists():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}: {path}")
        time.sleep(interval)


def run_task_with_mode(
    *,
    task: Any,
    model_uri: str | Model,
    native_log_dir: Path,
    mode: str,
    limit: int | None,
    judge_exec: JudgeExecutionConfig,
    inspect_model_args: dict[str, Any] | None = None,
    temperature: float = 0.0,
    hf_log_dir: str | None = None,
) -> EvalLog:
    """Run an Inspect task in blocking or submit mode.

    When ``hf_log_dir`` is provided (e.g. ``hf://datasets/org/repo``), Inspect
    writes logs directly to HF Hub under that path.  The local ``native_log_dir``
    is still created for sibling data/cache directories.
    """
    native_log_dir.mkdir(parents=True, exist_ok=True)
    # Route Inspect data/cache writes to a sibling of inspect_logs so that
    # the log directory contains only actual log files.  Placing them inside
    # inspect_logs would pollute the directory with non-log JSON files (e.g.
    # hf dataset_info.json), which confuses `inspect view start`.
    configure_inspect_paths(native_log_dir.parent)

    effective_log_dir = hf_log_dir if hf_log_dir is not None else str(native_log_dir)

    kwargs: dict[str, Any] = {}
    batch = _resolve_batch_setting(judge_exec)
    if batch is not None:
        kwargs["batch"] = batch
    if judge_exec.timeout_seconds is not None:
        kwargs["time_limit"] = judge_exec.timeout_seconds

    if temperature > 0:
        kwargs["temperature"] = temperature

    logs = inspect_eval(
        task,
        model=model_uri,
        model_args=inspect_model_args or {},
        limit=limit,
        score=(mode == "blocking"),
        log_dir=effective_log_dir,
        log_format="json",
        log_samples=True,
        display="plain",
        **kwargs,
    )
    if not logs:
        raise RuntimeError("Inspect returned no logs for task run")
    return logs[0]


def write_jobs_manifest(
    *,
    run_dir: Path,
    log_path: str,
    scorer_names: list[str],
    eval_name: str,
) -> Path:
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "pending",
        "eval_name": eval_name,
        "log_path": log_path,
        "scorers": scorer_names,
    }
    path = jobs_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def resume_from_manifest(
    *,
    manifest_path: Path,
    scorers: list[Scorer],
    judge_exec: JudgeExecutionConfig | None = None,
) -> EvalLog:
    judge_exec = judge_exec or JudgeExecutionConfig()
    _wait_for_path(
        path=manifest_path,
        label="jobs manifest",
        poll_interval_seconds=judge_exec.poll_interval_seconds,
        timeout_seconds=judge_exec.timeout_seconds,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = manifest.get("log_path")
    if not log_path:
        raise RuntimeError(f"Manifest missing log_path: {manifest_path}")
    log_file = Path(log_path)

    _wait_for_path(
        path=log_file,
        label="inspect log",
        poll_interval_seconds=judge_exec.poll_interval_seconds,
        timeout_seconds=judge_exec.timeout_seconds,
    )

    if manifest.get("status") == "completed":
        return read_eval_log(log_path, format="json")

    log = read_eval_log(log_path, format="json")
    scored = inspect_score(log, scorers=scorers)
    write_eval_log(scored, location=log_path, format="json")

    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return scored

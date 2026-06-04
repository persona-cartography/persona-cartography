"""Inspect backend runner for benchmark and custom eval specifications.

The thin seam between a suite's eval *spec* and Inspect's task-execution
machinery. Each function builds the appropriate Inspect ``Task`` (benchmark or
custom), runs it via :func:`src.evals.judge_orchestration.run_task_with_mode`,
and wraps the outcome in :class:`InspectRunResult` (catching exceptions into a
``status="failed"`` result rather than propagating). Custom evals additionally
support the *submit* / *resume* judge-execution modes: ``run_custom_eval`` in
*submit* mode writes a jobs manifest and returns ``status="pending"``, and
:func:`score_custom_eval_from_log` later resumes scoring from that manifest.

Called by :func:`src.evals.suite.run_eval_suite`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import EvalLog
from inspect_ai.model import Model

from src.evals.config import (
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
)
from src.evals.inspect_benchmarks import build_benchmark_task
from src.evals.inspect_custom import build_custom_scorer, build_custom_task
from src.evals.judge_orchestration import (
    resume_from_manifest,
    run_task_with_mode,
    write_jobs_manifest,
)


@dataclass
class InspectRunResult:
    """Outcome of one Inspect eval execution (status + optional log/error)."""

    status: str
    log: EvalLog | None = None
    error: str | None = None
    manifest_path: Path | None = None


def run_benchmark_eval(
    *,
    spec: InspectBenchmarkSpec,
    model_uri: str | Model,
    run_dir: Path,
    inspect_model_args: dict | None = None,
    temperature: float = 0.0,
    hf_log_dir: str | None = None,
    task: "Task | None" = None,
) -> InspectRunResult:
    """Run a benchmark eval.

    Args:
        task: Optional pre-built Inspect Task.  When provided, skips
            ``build_benchmark_task(spec)`` — useful for reusing a cached
            task across scale points in a sweep.
    """
    native_log_dir = run_dir / "native" / "inspect_logs"

    try:
        if task is None:
            task = build_benchmark_task(spec)
        log = run_task_with_mode(
            task=task,
            model_uri=model_uri,
            native_log_dir=native_log_dir,
            mode="blocking",
            limit=spec.limit,
            # judge_exec=JudgeExecutionConfig(mode="blocking", prefer_batch=True),
            judge_exec=JudgeExecutionConfig(mode="blocking", prefer_batch=False),
            inspect_model_args=inspect_model_args,
            temperature=temperature,
            hf_log_dir=hf_log_dir,
        )
        return InspectRunResult(status="ok", log=log)
    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc))


def run_custom_eval(
    *,
    spec: InspectCustomEvalSpec,
    model_uri: str | Model,
    run_dir: Path,
    judge_exec: JudgeExecutionConfig,
    inspect_model_args: dict | None = None,
    hf_log_dir: str | None = None,
) -> InspectRunResult:
    """Run a custom (prompt-built) eval, honoring the judge execution mode.

    In *submit* mode, writes a jobs manifest and returns ``status="pending"``
    instead of a scored log; otherwise returns the scored log with
    ``status="ok"``.
    """
    native_log_dir = run_dir / "native" / "inspect_logs"

    try:
        task, scorer_name = build_custom_task(spec)
        log = run_task_with_mode(
            task=task,
            model_uri=model_uri,
            native_log_dir=native_log_dir,
            mode=judge_exec.mode,
            limit=spec.dataset.max_samples,
            judge_exec=judge_exec,
            inspect_model_args=inspect_model_args,
            hf_log_dir=hf_log_dir,
        )

        if judge_exec.mode == "submit":
            manifest_path = write_jobs_manifest(
                run_dir=run_dir,
                log_path=log.location,
                scorer_names=[scorer_name],
                eval_name=spec.name,
            )
            return InspectRunResult(
                status="pending",
                log=log,
                manifest_path=manifest_path,
            )

        return InspectRunResult(status="ok", log=log)

    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc))


def score_custom_eval_from_log(
    *,
    spec: InspectCustomEvalSpec,
    run_dir: Path,
    judge_exec: JudgeExecutionConfig | None = None,
) -> InspectRunResult:
    """Resume scoring a previously-submitted custom eval from its jobs manifest."""
    manifest_path = run_dir / "jobs" / "manifest.json"

    try:
        scorer_obj, _ = build_custom_scorer(spec)
        scored_log = resume_from_manifest(
            manifest_path=manifest_path,
            scorers=[scorer_obj],
            judge_exec=judge_exec,
        )
        return InspectRunResult(status="ok", log=scored_log, manifest_path=manifest_path)
    except Exception as exc:
        return InspectRunResult(status="failed", error=str(exc), manifest_path=manifest_path)

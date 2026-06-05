"""Runtime engine for rollout sweeps.

This is the top of the :mod:`src.sweep` stack: the phased-rollout primitives,
per-message evaluation, the single-experiment runners, the file-logging tee,
and the variant×condition sweep driver (:func:`run_sweep`). It depends on
:mod:`src.sweep.config` (the config types + builders) and
:mod:`src.sweep.export` (the export / provenance / HF helpers); nothing in the
package depends back on it.

``run_sweep`` loop structure::

    with provider:
        for variant in variants:
            with provider.activate(variant) as (model, tokenizer):
                asyncio.run(concurrent conditions sharing one GPU executor)
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.inference.providers import get_provider
from src.persona_metrics.conversation_eval import (
    ConversationMetricsConfig,
    MessageSelector,
    run_conversation_metrics_async,
)
from src.rollout_generation.config import FailurePolicyConfig, RolloutGenerationConfig
from src.rollout_generation.gpu_executor import GpuBatchExecutor
from src.rollout_generation.run import (
    run_rollout_generation,
    run_rollout_generation_async,
)

from .config import (
    ExperimentConfig,
    SweepConfig,
    build_assistant_inference,
    build_dataset,
    build_user_simulator,
)
from .export import (
    _batch_upload_variant_cells_to_hf,
    _git_commit_hash,
    _load_completed_cells_from_hf,
    _write_eval_info,
    _write_rollout_info,
    _write_run_info,
    export_evaluated_rollouts,
    export_rollouts,
    save_experiment_metadata,
    upload_evals_to_hf,
)

if TYPE_CHECKING:
    from src.persona_metrics.config import PersonaMetricSpec
    from src.persona_metrics.conversation_eval import ConversationMetricsResult
    from src.rollout_generation.config import UserSimulatorConfig

    from .config import Phase


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


# ── Phased rollout ────────────────────────────────────────────────────────────


def run_phased_rollout(
    config: ExperimentConfig,
    phases: list[Phase],
    run_dir: Path,
    *,
    user_sim: UserSimulatorConfig | None = None,
    preloaded_model: tuple | None = None,
    prompt_template_per_sample: dict[str, str] | None = None,
    user_sim_generates_opening: bool = False,
) -> None:
    """Execute a sequence of rollout phases on the same run_dir.

    Args:
        config: Experiment configuration.
        phases: List of Phase objects defining each phase.
        run_dir: Shared run directory for all phases.
        user_sim: Default user simulator config (used when phase.user_simulator is None).
        preloaded_model: Optional ``(model, tokenizer)`` tuple to skip model loading.
        prompt_template_per_sample: Per-sample user-sim template mapping.
        user_sim_generates_opening: If True, user sim generates the opening turn.
    """
    if user_sim is None:
        user_sim = build_user_simulator(config)
    dataset = build_dataset(config)
    assistant = build_assistant_inference(config)
    if preloaded_model is not None:
        assistant = assistant.model_copy(
            update={
                "local": assistant.local.model_copy(
                    update={"preloaded_model": preloaded_model}
                )
            }
        )

    cumulative_turns = 0
    for phase_idx, phase in enumerate(phases):
        cumulative_turns += phase.num_turns
        phase_user_sim = phase.user_simulator or user_sim

        print(
            f"\n  Phase {phase_idx + 1}/{len(phases)}: "
            f"{phase.num_turns} turns, "
            f"system_prompt={'yes' if phase.assistant_system_prompt else 'no'}, "
            f"user_template={phase_user_sim.prompt_template}"
        )

        is_last_phase = phase_idx == len(phases) - 1
        rollout_config = RolloutGenerationConfig(
            dataset=dataset,
            run_dir=run_dir,
            num_assistant_turns=cumulative_turns,
            num_rollouts_per_prompt=config.num_rollouts,
            system_prompt=phase.assistant_system_prompt,
            assistant_inference=assistant,
            user_simulator=phase_user_sim,
            failure_policy=FailurePolicyConfig(
                assistant_max_attempts_per_turn=3,
                user_max_attempts_per_turn=3,
            ),
            skip_final_user_turn=is_last_phase,
            resume=True,
            prompt_template_per_sample=prompt_template_per_sample or {},
            user_sim_generates_opening=user_sim_generates_opening,
        )
        _, result = run_rollout_generation(rollout_config)
        print(
            f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations"
        )

        if result.num_failed > 0:
            raise RuntimeError(
                f"Phase {phase_idx + 1}/{len(phases)} had {result.num_failed}/{result.num_conversations} "
                f"failed conversations (target: {cumulative_turns} assistant turns)"
            )


async def run_phased_rollout_async(
    config: ExperimentConfig,
    phases: list[Phase],
    run_dir: Path,
    *,
    user_sim: UserSimulatorConfig | None = None,
    preloaded_model: tuple | None = None,
    gpu_executor: GpuBatchExecutor | None = None,
    prompt_template_per_sample: dict[str, str] | None = None,
    user_sim_generates_opening: bool = False,
) -> None:
    """Async version of :func:`run_phased_rollout`.

    When ``gpu_executor`` is provided, all phases feed into the shared
    executor instead of each creating its own.
    """
    if user_sim is None:
        user_sim = build_user_simulator(config)
    dataset = build_dataset(config)
    assistant = build_assistant_inference(config)
    if preloaded_model is not None:
        assistant = assistant.model_copy(
            update={
                "local": assistant.local.model_copy(
                    update={"preloaded_model": preloaded_model}
                )
            }
        )

    cumulative_turns = 0
    for phase_idx, phase in enumerate(phases):
        cumulative_turns += phase.num_turns
        phase_user_sim = phase.user_simulator or user_sim

        print(
            f"\n  Phase {phase_idx + 1}/{len(phases)}: "
            f"{phase.num_turns} turns, "
            f"system_prompt={'yes' if phase.assistant_system_prompt else 'no'}, "
            f"user_template={phase_user_sim.prompt_template}"
        )

        is_last_phase = phase_idx == len(phases) - 1
        rollout_config = RolloutGenerationConfig(
            dataset=dataset,
            run_dir=run_dir,
            num_assistant_turns=cumulative_turns,
            num_rollouts_per_prompt=config.num_rollouts,
            system_prompt=phase.assistant_system_prompt,
            assistant_inference=assistant,
            user_simulator=phase_user_sim,
            failure_policy=FailurePolicyConfig(
                assistant_max_attempts_per_turn=3,
                user_max_attempts_per_turn=3,
            ),
            skip_final_user_turn=is_last_phase,
            resume=True,
            prompt_template_per_sample=prompt_template_per_sample or {},
            user_sim_generates_opening=user_sim_generates_opening,
        )
        _, result = await run_rollout_generation_async(
            rollout_config, gpu_executor=gpu_executor
        )
        print(
            f"  -> Completed: {result.num_completed}/{result.num_conversations} conversations"
        )

        if result.num_failed > 0:
            raise RuntimeError(
                f"Phase {phase_idx + 1}/{len(phases)} had {result.num_failed}/{result.num_conversations} "
                f"failed conversations (target: {cumulative_turns} assistant turns)"
            )


# ── Evaluation ────────────────────────────────────────────────────────────────


async def evaluate_messages(
    run_dir: Path,
    evaluations: list[str | PersonaMetricSpec],
    *,
    message_selector: MessageSelector | None = None,
) -> ConversationMetricsResult:
    """Run per-message evaluation on a completed rollout."""
    if message_selector is None:
        message_selector = MessageSelector(exclude_seed=True)

    print(f"\n  Evaluating messages with {evaluations}...")
    eval_config = ConversationMetricsConfig(
        evaluations=evaluations,
        run_dir=run_dir,
        message_selector=message_selector,
        output_path=run_dir / "per_message_metrics.jsonl",
    )
    result = await run_conversation_metrics_async(eval_config)

    print(
        f"  -> Evaluated {result.num_messages_evaluated} messages "
        f"across {result.num_conversations} conversations"
    )
    if result.aggregates:
        for key, val in sorted(result.aggregates.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")
            elif not isinstance(val, dict):
                print(f"     {key}: {val}")

    grouped = result.aggregates.get("by_prompt_and_role", {})
    if grouped:
        print("\n  Per-prompt/role breakdown:")
        for key, val in sorted(grouped.items()):
            if isinstance(val, float):
                print(f"     {key}: {val:.4f}")

    return result


# ── Single experiment runner ──────────────────────────────────────────────────


def run_experiment(
    config: ExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str | PersonaMetricSpec],
    run_dir: Path,
    *,
    user_sim: UserSimulatorConfig | None = None,
    preloaded_model: tuple | None = None,
    skip_rollouts: bool = False,
    skip_evals: bool = False,
    eval_roles: list[str] | None = None,
    prompt_template_per_sample: dict[str, str] | None = None,
    user_sim_generates_opening: bool = False,
) -> ConversationMetricsResult | None:
    """Run a complete experiment: phased rollout, evaluation, metadata, export.

    Args:
        config: Experiment configuration.
        name: Experiment name.
        phases: List of Phase objects defining the rollout.
        evaluations: Persona metrics to run on each message.
        run_dir: Output directory for this experiment.
        user_sim: Optional default user simulator override.
        preloaded_model: Optional ``(model, tokenizer)`` tuple.
        skip_rollouts: If True, skip rollout generation (use
            existing rollouts) and only run evaluation.
        eval_roles: Which message roles to evaluate (e.g. ``["assistant"]``).
            ``None`` evaluates all roles.

    Returns:
        ConversationMetricsResult with per-message scores and aggregates.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    # Phase 1: Rollout generation
    if not skip_rollouts:
        rollout_t0 = time.perf_counter()
        run_phased_rollout(
            config,
            phases,
            run_dir,
            user_sim=user_sim,
            preloaded_model=preloaded_model,
            prompt_template_per_sample=prompt_template_per_sample,
            user_sim_generates_opening=user_sim_generates_opening,
        )
        export_rollouts(run_dir)
        save_experiment_metadata(config, run_dir, name, phases)
        _write_rollout_info(run_dir, time.perf_counter() - rollout_t0)
    else:
        print("  Skipping rollout generation (using existing)")

    # Phase 2: Evaluation
    if skip_evals or not evaluations:
        print("  Skipping evaluation")
        return None

    message_selector = MessageSelector(exclude_seed=True, roles=eval_roles)
    eval_t0 = time.perf_counter()
    result = asyncio.run(evaluate_messages(run_dir, evaluations, message_selector=message_selector))
    export_evaluated_rollouts(run_dir, result)
    _write_eval_info(run_dir, evaluations, time.perf_counter() - eval_t0)

    return result


# ── Sweep runner ──────────────────────────────────────────────────────────────


async def _run_experiment_async(
    config: ExperimentConfig,
    name: str,
    phases: list[Phase],
    evaluations: list[str | PersonaMetricSpec],
    run_dir: Path,
    *,
    user_sim: UserSimulatorConfig | None = None,
    preloaded_model: tuple | None = None,
    skip_rollouts: bool = False,
    skip_evals: bool = False,
    gpu_executor: GpuBatchExecutor | None = None,
    eval_roles: list[str] | None = None,
    prompt_template_per_sample: dict[str, str] | None = None,
    user_sim_generates_opening: bool = False,
) -> ConversationMetricsResult | None:
    """Async version of :func:`run_experiment`.

    Accepts ``gpu_executor`` so that multiple experiments can share a single
    GPU batch queue when run concurrently.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Run dir: {run_dir}")
    print(f"{'=' * 60}")

    if not skip_rollouts:
        rollout_t0 = time.perf_counter()
        await run_phased_rollout_async(
            config,
            phases,
            run_dir,
            user_sim=user_sim,
            preloaded_model=preloaded_model,
            gpu_executor=gpu_executor,
            prompt_template_per_sample=prompt_template_per_sample,
            user_sim_generates_opening=user_sim_generates_opening,
        )
        export_rollouts(run_dir)
        save_experiment_metadata(config, run_dir, name, phases)
        _write_rollout_info(run_dir, time.perf_counter() - rollout_t0)
    else:
        print("  Skipping rollout generation (using existing)")

    if skip_evals or not evaluations:
        print("  Skipping evaluation")
        return None

    message_selector = MessageSelector(exclude_seed=True, roles=eval_roles)
    eval_t0 = time.perf_counter()
    result = await evaluate_messages(run_dir, evaluations, message_selector=message_selector)
    export_evaluated_rollouts(run_dir, result)
    _write_eval_info(run_dir, evaluations, time.perf_counter() - eval_t0)

    return result


# ── File logging ──────────────────────────────────────────────────────────────


class _TeeStream(io.TextIOBase):
    """Write to both the original stream and a log file."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, s: str) -> int:
        self._original.write(s)
        self._log_file.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()


def _setup_file_logging(
    output_dir: Path,
) -> tuple[io.TextIOBase, logging.Handler]:
    """Tee stdout/stderr to a log file and add a file handler for the logger.

    Returns (log_file_handle, logger_file_handler) for cleanup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sweep.log"
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    # Tee stdout and stderr to the log file.
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    # Also capture Python logging (used by rollout_generation, gpu_executor, etc.)
    # Call setup_logging() first so it installs the StreamHandler to stderr
    # (now tee'd). The file handler is added on top so logs go to both terminal
    # and sweep.log. Without this, setup_logging()'s "if not logger.handlers:"
    # guard fires too late and the StreamHandler is never added.
    from src.utils import setup_logging as _setup_logging
    _setup_logging()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger("persona_shattering").addHandler(file_handler)

    return log_file, file_handler


def _teardown_file_logging(
    log_file: io.TextIOBase,
    file_handler: logging.Handler,
) -> None:
    """Restore stdout/stderr and remove the file handler."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    logging.getLogger("persona_shattering").removeHandler(file_handler)
    file_handler.close()
    log_file.close()


def _print_timing_summary(
    timings: list[tuple[str, str, str, float]],
    suite_elapsed: float,
) -> None:
    if not timings:
        print(
            f"\n=== Sweep done in {_fmt_duration(suite_elapsed)} (no cells ran) ===\n",
            flush=True,
        )
        return

    col_variant = max(max(len(v) for v, _, _, _ in timings), 7)
    col_cond = max(max(len(c) for _, c, _, _ in timings), 9)

    header = (
        f"  {'Variant':<{col_variant}}  {'Condition':<{col_cond}}  {'Status':<7}  Time"
    )
    sep = "  " + "-" * (col_variant + col_cond + 22)
    print("\n=== Timing summary ===", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for variant_label, condition, status, elapsed in timings:
        t = _fmt_duration(elapsed) if elapsed > 0 else "-"
        print(
            f"  {variant_label:<{col_variant}}  {condition:<{col_cond}}  {status:<7}  {t}",
            flush=True,
        )
    print(sep, flush=True)
    print(f"  Total: {_fmt_duration(suite_elapsed)}", flush=True)
    print("======================\n", flush=True)


async def _run_variant_conditions_async(
    conditions: list,
    config: SweepConfig,
    model,
    tokenizer,
    output_root: Path,
    vlabel: str,
    rollouts_completed_on_hf: set[str],
) -> list[tuple[str, str, str, float]]:
    """Run all conditions concurrently for a single variant.

    Creates a shared :class:`GpuBatchExecutor` so that conversations from
    all conditions feed into the same GPU batch queue, improving utilisation.

    Returns:
        List of ``(vlabel, condition_name, status, elapsed)`` timing tuples.
    """
    # Build the assistant provider.
    # If the model is already an InferenceProvider (e.g. from VLLMLoRaScaleProvider),
    # use it directly.  Otherwise wrap the preloaded HF model in a LocalProvider.
    from src.inference.providers.base import InferenceProvider as _InferenceProvider

    if isinstance(model, _InferenceProvider):
        assistant_provider = model
        batch_size = max(1, config.experiment.assistant_batch_size)
    else:
        assistant_config = build_assistant_inference(config.experiment)
        assistant_config = assistant_config.model_copy(
            update={
                "local": assistant_config.local.model_copy(
                    update={"preloaded_model": (model, tokenizer)}
                )
            }
        )
        assistant_provider = get_provider(assistant_config.provider, assistant_config)
        batch_size = max(1, assistant_config.generation.batch_size)
    executor = GpuBatchExecutor(assistant_provider, batch_size=batch_size)
    executor_task = asyncio.create_task(executor.run())

    timings: list[tuple[str, str, str, float]] = []
    cells_to_upload: list[tuple[Path, str, str]] = []
    successful_cells: set[tuple[str, str]] = set()
    semaphore = asyncio.Semaphore(config.max_concurrent_conditions)

    async def _run_one_condition(condition) -> None:
        async with semaphore:
            cell_dir = output_root / vlabel / condition.name
            cell_label = f"{vlabel}/{condition.name}"

            cell_skip_rollouts = (
                config.skip_completed and cell_label in rollouts_completed_on_hf
            )
            if cell_skip_rollouts:
                print(
                    f"    running   {cell_label}  (rollouts on HF, evals only)",
                    flush=True,
                )
            else:
                print(
                    f"    running   {cell_label} ...",
                    flush=True,
                )

            # Remove any FAILED.md left by a previous failed attempt so it
            # is not re-uploaded if this run succeeds.
            if cell_dir.is_dir():
                (cell_dir / "FAILED.md").unlink(missing_ok=True)

            cell_t0 = time.perf_counter()
            cell_experiment = ExperimentConfig(**{**vars(config.experiment)})

            try:
                result = await _run_experiment_async(
                    cell_experiment,
                    name=condition.name,
                    phases=condition.phases,
                    evaluations=config.evaluations,
                    run_dir=cell_dir,
                    user_sim=condition.user_sim,
                    preloaded_model=(model, tokenizer),
                    skip_rollouts=cell_skip_rollouts,
                    skip_evals=config.skip_evals,
                    gpu_executor=executor,
                    eval_roles=condition.eval_roles,
                    prompt_template_per_sample=condition.prompt_template_per_sample,
                    user_sim_generates_opening=condition.user_sim_generates_opening,
                )
                elapsed = time.perf_counter() - cell_t0
                aggregates = result.aggregates if result else None
                _write_run_info(
                    cell_dir,
                    vlabel,
                    condition.name,
                    "ok",
                    aggregates,
                    None,
                    elapsed,
                )
                successful_cells.add((vlabel, condition.name))
                cells_to_upload.append((cell_dir, vlabel, condition.name))
                timings.append((vlabel, condition.name, "ok", elapsed))
                print(
                    f"    done      {cell_label}  ({_fmt_duration(elapsed)})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - cell_t0
                if config.on_cell_error == "raise":
                    raise
                print(
                    f"    FAILED    {cell_label}  ({_fmt_duration(elapsed)}): {exc}",
                    flush=True,
                )
                cell_dir.mkdir(parents=True, exist_ok=True)
                (cell_dir / "FAILED.md").write_text(
                    f"# Cell failed: {cell_label}\n\n"
                    f"**Elapsed:** {_fmt_duration(elapsed)}\n\n"
                    f"**Error:**\n```\n{exc}\n```\n",
                    encoding="utf-8",
                )
                _write_run_info(
                    cell_dir,
                    vlabel,
                    condition.name,
                    "failed",
                    None,
                    str(exc),
                    elapsed,
                )
                if config.on_cell_error == "continue":
                    cells_to_upload.append((cell_dir, vlabel, condition.name))
                timings.append((vlabel, condition.name, "failed", elapsed))

    try:
        await asyncio.gather(
            *[_run_one_condition(c) for c in conditions],
        )
    finally:
        executor.stop()
        await executor_task
        # Free intermediate tensors and CUDA cache.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return timings, cells_to_upload, successful_cells


def run_sweep(config: SweepConfig) -> Path:
    """Execute a full model-variant sweep and return the output directory.

    Loop structure::

        with provider:
            for variant in variants:
                with provider.activate(variant) as (model, tokenizer):
                    asyncio.run(concurrent conditions sharing GPU executor)

    Args:
        config: Full sweep configuration.

    Returns:
        Path to the run directory containing all results.
    """
    output_root = config.output.scratch_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # Tee all output (print + logging) to sweep.log in the output directory.
    log_file, log_handler = _setup_file_logging(output_root)

    variants = config.provider.variant_names()
    n_variants = len(variants)
    n_conditions = len(config.conditions)

    print(
        f"\n=== Sweep: {config.output.eval_name} "
        f"| {n_variants} variant(s) x {n_conditions} condition(s) ===",
        flush=True,
    )

    # Write sweep config for reproducibility.
    (output_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "provider_type": type(config.provider).__name__,
                "variants": [config.provider.variant_label(v) for v in variants],
                "conditions": [c.name for c in config.conditions],
                "evaluations": [
                    e if isinstance(e, str) else e.name for e in config.evaluations
                ],
                "experiment": dataclasses.asdict(config.experiment),
                "output": dataclasses.asdict(config.output),
                "git_commit_hash": _git_commit_hash(),
                **config.metadata,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    suite_t0 = time.perf_counter()
    timings: list[tuple[str, str, str, float]] = []

    # Pre-fetch cells with completed rollouts from HF.
    rollouts_completed_on_hf: set[str] = set()
    if config.skip_completed and config.output.hf_repo:
        print(
            "  Checking HF for completed rollouts...",
            flush=True,
        )
        rollouts_completed_on_hf = _load_completed_cells_from_hf(config.output)
        print(
            f"  Found {len(rollouts_completed_on_hf)} cell(s) "
            f"with completed rollouts on HF.",
            flush=True,
        )

    with config.provider:
        for variant_idx, variant in enumerate(variants, 1):
            vlabel = config.provider.variant_label(variant)
            print(
                f"\n  [{variant_idx}/{n_variants}] variant={vlabel}",
                flush=True,
            )

            with config.provider.activate(variant) as (model, tokenizer):
                variant_timings, cells_to_upload, successful_cells = asyncio.run(
                    _run_variant_conditions_async(
                        conditions=config.conditions,
                        config=config,
                        model=model,
                        tokenizer=tokenizer,
                        output_root=output_root,
                        vlabel=vlabel,
                        rollouts_completed_on_hf=rollouts_completed_on_hf,
                    )
                )
                timings.extend(variant_timings)
                if config.output.hf_repo and cells_to_upload:
                    _batch_upload_variant_cells_to_hf(
                        config.output, cells_to_upload, successful_cells
                    )
                    evals_dirs = [
                        cell_dir / "evals"
                        for cell_dir, _, _ in cells_to_upload
                        if (cell_dir / "evals" / "rollouts_evaluated.jsonl").exists()
                    ]
                    if evals_dirs:
                        upload_evals_to_hf(config.output, output_root, evals_dirs=evals_dirs)

    _print_timing_summary(timings, time.perf_counter() - suite_t0)

    _teardown_file_logging(log_file, log_handler)
    return output_root

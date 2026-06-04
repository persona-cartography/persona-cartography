"""Generalized rollout sweep infrastructure.

Provides model-agnostic sweep orchestration over multi-phase rollout
experiments. Supports any ``ModelProvider`` (LoRA scale, activation capping,
plain HF model) with reusable condition templates.

Usage::

    from src.sweep import (
        ExperimentConfig, Phase, SweepCondition, SweepConfig,
        OutputPathConfig, run_sweep, single_turn_conditions,
    )
    from src.rollout_generation.model_providers import LoRaScaleProvider

    provider = LoRaScaleProvider(...)
    conditions = single_turn_conditions({"baseline": None, "t_avoiding": "..."})
    config = SweepConfig(
        provider=provider,
        conditions=conditions,
        evaluations=["count_t"],
        experiment=ExperimentConfig(...),
        output=OutputPathConfig(...),
    )
    run_sweep(config)
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, model_validator

from src.common.config import DatasetConfig, GenerationConfig
from src.datasets import get_run_paths, load_samples, materialize_canonical_samples
from src.datasets.io import read_jsonl_tolerant
from src.inference.config import InferenceConfig, LocalProviderConfig, RetryConfig
from src.inference.providers import get_provider
from src.persona_metrics.config import PersonaMetricSpec
from src.persona_metrics.conversation_eval import (
    ConversationMetricsConfig,
    ConversationMetricsResult,
    MessageSelector,
    run_conversation_metrics,
    run_conversation_metrics_async,
)
from src.rollout_generation.config import (
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from src.rollout_generation.gpu_executor import GpuBatchExecutor
from src.rollout_generation.model_providers import ModelProvider
from src.rollout_generation.prompts import (
    get_user_simulator_instruction,
    register_user_simulator_template,
)
from src.rollout_generation.run import (
    run_rollout_generation,
    run_rollout_generation_async,
)
from src.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

# ── Helpers ───────────────────────────────────────────────────────────────────


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip() or None


def _read_calling_script() -> str | None:
    """Read the content of the calling script (sys.argv[0]) for provenance."""
    try:
        return Path(sys.argv[0]).resolve().read_text()
    except (OSError, ValueError):
        return None


def _nest_scores(flat_scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert flat metric scores to nested-by-evaluator format.

    Example::

        {"count_t.count": 82, "count_t.density": 13.29}
        → {"count_t": {"count": 82, "density": 13.29}}
    """
    nested: dict[str, dict[str, Any]] = {}
    for key, value in flat_scores.items():
        if "." in key:
            evaluator, subkey = key.split(".", 1)
            nested.setdefault(evaluator, {})[subkey] = value
        else:
            nested.setdefault(key, {})["score"] = value
    return nested


def _write_rollout_info(run_dir: Path, elapsed: float) -> Path:
    """Write rollouts/rollout_info.json with metadata about the generation run."""
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "status": "ok",
        "datetime": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed, 2),
    }
    path = rollouts_dir / "rollout_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def _write_eval_info(
    run_dir: Path,
    evaluations: list[str | PersonaMetricSpec],
    elapsed: float,
) -> Path:
    """Write eval_info.json and script copy into a timestamped eval_runs/ subdir."""
    evals_dir = run_dir / "evals"
    eval_names = [
        e if isinstance(e, str) else e.name if hasattr(e, "name") else str(e)
        for e in evaluations
    ]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{'__'.join(eval_names)}__{ts}"
    eval_run_dir = evals_dir / "eval_runs" / run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "status": "ok",
        "datetime": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed, 2),
        "evaluators": [
            e
            if isinstance(e, str)
            else dataclasses.asdict(e)
            if dataclasses.is_dataclass(e)
            else str(e)
            for e in evaluations
        ],
    }
    path = eval_run_dir / "eval_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, eval_run_dir / script_path.name)
    return path


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """All knobs for a rollout experiment. Instantiate once at top of script.

    No defaults — all fields must be specified explicitly.

    Suggested values::

        ExperimentConfig(
            assistant_model="meta-llama/Llama-3.1-8B-Instruct",
            assistant_provider="local",  # or "openrouter"
            assistant_temperature=0.7,
            assistant_top_p=0.95,
            assistant_max_new_tokens=256,
            assistant_batch_size=32,
            user_model="gpt-4.1-nano-2025-04-14",
            user_provider="openrouter",
            user_temperature=0.7,
            user_top_p=0.95,
            user_max_new_tokens=256,
            user_batch_size=16,
            user_max_concurrent=64,
            dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
            max_samples=10,
            num_rollouts=3,
            turns_per_phase=[3, 1],
        )
    """

    # Assistant model
    assistant_model: str
    assistant_provider: str
    assistant_temperature: float
    assistant_top_p: float
    assistant_max_new_tokens: int
    assistant_batch_size: int

    # User simulator
    user_model: str
    user_provider: str
    user_temperature: float
    user_top_p: float
    user_max_new_tokens: int
    user_batch_size: int
    user_max_concurrent: int

    # Dataset
    dataset_path: str
    max_samples: int
    dataset_seed: int | None = None
    num_rollouts: int = 1

    # Experiment
    turns_per_phase: list[int] = field(default_factory=lambda: [5, 5])
    system_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class Phase:
    """One phase of a multi-phase rollout experiment."""

    num_turns: int
    assistant_system_prompt: str | None = None
    user_simulator: UserSimulatorConfig | None = None


@dataclass
class SweepCondition:
    """One condition (system-prompt / user-sim variant) to run at every model variant.

    Args:
        name: Short identifier used in directory names.
        phases: Phase list passed to ``run_experiment``.
        user_sim: Optional user simulator override for this condition.
        eval_roles: Which message roles to evaluate.  ``["assistant"]`` for
            AU conditions (skip user messages), ``None`` for AA conditions
            (evaluate all messages).  Defaults to ``["assistant"]``.
        prompt_template_per_sample: Optional mapping from ``sample_id`` to
            user-simulator template name, threaded through to
            ``RolloutGenerationConfig``. Use with ``register_user_simulator_template``
            to set up per-sample user-sim prompts (e.g. one template per scenario).
            Empty dict = default template for all samples.
        user_sim_generates_opening: If True, the user simulator generates the
            opening user turn instead of using the dataset prompt verbatim.
            Useful for scenario-based rollouts where the scenario context
            should drive the first message. Defaults to False.
    """

    name: str
    phases: list[Phase]
    user_sim: UserSimulatorConfig | None = None
    eval_roles: list[str] | None = field(default_factory=lambda: ["assistant"])
    prompt_template_per_sample: dict[str, str] = field(default_factory=dict)
    user_sim_generates_opening: bool = False


@dataclass
class OutputPathConfig:
    """Structured output path configuration.

    Two layouts are supported:

    Legacy (``training_run`` only)::

        {scratch_root}/fine_tuning/{base_model}/{category}/{trait}/{training_run}/{stage_dir}/{eval_name}/

    OCT layout (when ``direction`` and ``version`` are set)::

        {scratch_root}/fine_tuning/{base_model}/{category}/{trait}/{direction}/{version}/{stage_dir}/{eval_name}/

    HF upload mirrors the same structure (minus scratch_root).

    Args:
        scratch_root: Root directory for scratch output.
        hf_repo: HuggingFace dataset repo ID for upload. None to skip upload.
        base_model: Model identifier (e.g. ``"llama-3.1-8B-Instruct"``).
        category: Trait category (e.g. ``"toy"``, ``"OCEAN"``).
        trait: Specific trait (e.g. ``"t character"``, ``"Openness/O+"``).
        training_run: Training run identifier (legacy layout). Ignored when
            both ``direction`` and ``version`` are set.
        direction: OCT direction slot (e.g. ``"amplifier"``, ``"suppressor"``).
            When set together with ``version``, triggers the OCT-style path.
        version: OCT version slot (e.g. ``"v1"``, ``"vanton2"``). When set
            together with ``direction``, triggers the OCT-style path.
        eval_name: Evaluation name (e.g. ``"rollout_sweep_lora_scale"``).
        stage_dir: Top-level artifact directory under the training run /
            version slot (e.g. ``"rollouts"``, ``"evals"``).
    """

    scratch_root: Path  # e.g. Path("scratch/monorepo")
    base_model: str  # e.g. "llama-3.1-8B-Instruct"
    category: str  # e.g. "toy", "OCEAN"
    trait: str  # e.g. "t_character", "Openness/O+"
    eval_name: str  # e.g. "rollout_sweep_lora_scale"
    training_run: str | None = None  # legacy; unused when direction+version set
    direction: str | None = None  # OCT: "amplifier", "suppressor"
    version: str | None = None  # OCT: "v1", "vanton2"
    hf_repo: str | None = None  # e.g. "persona-shattering-lasr/monorepo"
    stage_dir: str = "rollouts"

    def __post_init__(self) -> None:
        oct_set = self.direction is not None and self.version is not None
        legacy_set = self.training_run is not None
        if not oct_set and not legacy_set:
            raise ValueError(
                "OutputPathConfig requires either training_run (legacy) "
                "or both direction and version (OCT layout)."
            )

    @property
    def relative_path(self) -> Path:
        """Path relative to scratch_root (also used as HF repo path)."""
        base = Path("fine_tuning") / self.base_model / self.category / self.trait
        if self.direction is not None and self.version is not None:
            base = base / self.direction / self.version
        else:
            assert self.training_run is not None  # checked in __post_init__
            base = base / self.training_run
        return base / self.stage_dir / self.eval_name

    @property
    def scratch_dir(self) -> Path:
        """Full local scratch path."""
        return self.scratch_root / self.relative_path

    @property
    def hf_path(self) -> str:
        """Path within the HF repo."""
        return str(self.relative_path)


class SweepConfig(BaseModel):
    """Full configuration for a model-variant sweep.

    Args:
        provider: Model provider that defines the sweep dimension.
        conditions: List of conditions to run at every model variant.
        evaluations: Persona metrics to run on each message.
        experiment: Experiment configuration (generation settings, dataset, etc.).
        output: Output path configuration.
        skip_completed: Skip cells that already have ``run_info.json`` with
            ``status == "ok"``.
        plot: Generate a sweep plot after all cells complete.
        plot_metric: Aggregate key to plot. Auto-derived from first evaluation
            if ``None``.
        metadata: Arbitrary extra fields written into ``sweep_config.json``.
    """

    model_config = {"arbitrary_types_allowed": True}

    provider: ModelProvider
    conditions: list[SweepCondition]
    evaluations: list[str | PersonaMetricSpec]
    experiment: ExperimentConfig
    output: OutputPathConfig
    skip_completed: bool = True
    on_cell_error: Literal["raise", "warn", "continue"] = "continue"
    """How to handle a cell that raises an exception.

    - ``"raise"``: re-raise immediately (stops the script). Failed cell is
      **not** uploaded.
    - ``"warn"``: print a warning and continue to the next cell. Failed cell
      is **not** uploaded.
    - ``"continue"``: print a warning, write ``run_info.json``, and upload
      the failed cell (current default behaviour).
    """
    skip_evals: bool = False
    max_concurrent_conditions: int = 2
    """Maximum number of conditions to run concurrently within a variant.

    Higher values improve GPU utilisation (more conversations batched together)
    but increase peak memory.  Set to ``1`` for sequential (old) behaviour.
    """
    plot: bool = True
    plot_metric: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_plot_metric(self) -> "SweepConfig":
        if self.plot_metric is None and self.evaluations:
            from src.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric
            from src.persona_metrics.registry import PERSONA_METRIC_REGISTRY

            first_eval = self.evaluations[0]
            name = first_eval if isinstance(first_eval, str) else first_eval.name
            metric_cls = PERSONA_METRIC_REGISTRY.get(name)
            sub_key = (
                "score"
                if metric_cls is not None and issubclass(metric_cls, LLMJudgeMetric)
                else "density"
            )
            self.plot_metric = f"overall/{name}.{sub_key}/mean"
        return self


# ── Config builders ───────────────────────────────────────────────────────────


def build_assistant_inference(config: ExperimentConfig) -> InferenceConfig:
    """Build InferenceConfig for the assistant model."""
    return InferenceConfig(
        model=config.assistant_model,
        provider=config.assistant_provider,
        local=LocalProviderConfig(prompt_format="chat", truncate_inputs=False),
        generation=GenerationConfig(
            max_new_tokens=config.assistant_max_new_tokens,
            temperature=config.assistant_temperature,
            top_p=config.assistant_top_p,
            do_sample=True,
            batch_size=config.assistant_batch_size,
        ),
    )


def build_user_simulator(
    config: ExperimentConfig,
    prompt_template: str = "typical_user",
    prompt_format: str = "single_turn_text",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> UserSimulatorConfig:
    """Build UserSimulatorConfig, optionally overriding provider/model."""
    return UserSimulatorConfig(
        provider=provider or config.user_provider,
        model=model or config.user_model,
        prompt_template=prompt_template,
        prompt_format=prompt_format,
        generation=GenerationConfig(
            max_new_tokens=config.user_max_new_tokens,
            temperature=config.user_temperature,
            top_p=config.user_top_p,
            do_sample=True,
            batch_size=config.user_batch_size,
        ),
        max_concurrent=config.user_max_concurrent,
        retry=RetryConfig(),
    )


def build_dataset(config: ExperimentConfig) -> DatasetConfig:
    """Build DatasetConfig from experiment config."""
    return DatasetConfig(
        source="local",
        path=config.dataset_path,
        max_samples=config.max_samples,
        seed=config.dataset_seed,
    )


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


# ── Export ────────────────────────────────────────────────────────────────────


def _message_list_for_sample(
    sample: Any,
    scores_by_msg: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Extract seed_input and list of message dicts from a sample."""
    seed_input = None
    messages = []
    for msg in sample.messages:
        meta = msg.message_metadata or {}
        source = meta.get("source_stage", "")
        if source == "seed":
            seed_input = msg.content
            continue
        msg_entry: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
            "turn_index": meta.get("turn_index"),
            "system_prompt_hash": meta.get("system_prompt_hash")
            or meta.get("active_system_prompt"),
            "source": source,
        }
        if scores_by_msg:
            msg_scores = scores_by_msg.get(msg.message_id)
            if msg_scores:
                msg_entry["scores"] = msg_scores
        messages.append(msg_entry)
    return seed_input, messages


def _group_samples_by_seed(samples: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for s in samples:
        seed_id = s.input_group_id or s.sample_id
        groups.setdefault(seed_id, []).append(s)
    for group in groups.values():
        group.sort(key=lambda s: s.response_index)
    return groups


def _collect_failure_events(run_dir: Path, seed_ids: set[str]) -> list[dict[str, Any]]:
    """Read stage_events.jsonl and return failure/attempt events for given seeds.

    Stage events use per-rollout sample IDs (e.g. ``sample_abc123_r0_def456``)
    while rollout entries group by seed ID (``sample_abc123``).  We match by
    checking whether the event's sample_id starts with any of the seed IDs.
    """
    events_path = get_run_paths(run_dir)["stage_events"]
    rows, _ = read_jsonl_tolerant(events_path)
    # Build prefix tuple for fast startswith matching.
    prefixes = tuple(seed_ids)
    results = []
    for row in rows:
        sid = row.get("sample_id", "")
        if not any(sid.startswith(p) for p in prefixes):
            continue
        evt = row.get("event_type", "")
        if evt in {"terminal_failure", "invalid_state"}:
            results.append(row)
        elif evt in {"assistant_attempt", "user_attempt"}:
            payload = row.get("payload", {})
            if payload.get("status") == "failed":
                results.append(row)
    return results


def export_rollouts(run_dir: Path) -> Path:
    """Write rollouts.jsonl: one line per seed with messages dict keyed by rollout index."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    groups = _group_samples_by_seed(samples)

    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    # Validate: flag any rollout that produced zero messages.
    empty = [
        (e["seed_id"], idx)
        for e in entries
        for idx, msgs in e["messages"].items()
        if len(msgs) == 0
    ]
    if empty:
        # Collect failure details from stage_events.jsonl for the affected samples.
        empty_sample_ids = {seed_id for seed_id, _ in empty}
        failure_details = _collect_failure_events(run_dir, empty_sample_ids)

        # Write errors log so the user can inspect after the fact.
        errors_path = run_dir / "errors.jsonl"
        errors_path.write_text(
            "\n".join(json.dumps(e, default=str) for e in failure_details) + "\n"
        )

        summary_lines = [
            f"{len(empty)} rollout(s) have empty messages (no assistant output). "
            f"Error details written to {errors_path}"
        ]
        # Show a few failure reasons inline.
        reasons = [
            f"  {d['sample_id']}: {d['event_type']} — {d.get('payload', {}).get('reason') or d.get('payload', {}).get('error', '?')}"
            for d in failure_details[:10]
        ]
        if reasons:
            summary_lines.append("First failures:")
            summary_lines.extend(reasons)

        raise RuntimeError("\n".join(summary_lines))

    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    out_path = rollouts_dir / "rollouts.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    n_prompts = len(entries)
    n_responses = sum(len(e["messages"]) for e in entries)
    print(
        f"\n  Wrote {n_prompts} prompt(s) / {n_responses} response(s) "
        f"to {out_path}"
    )
    return out_path


def export_evaluated_rollouts(
    run_dir: Path,
    eval_result: ConversationMetricsResult,
) -> Path:
    """Write rollouts_evaluated.jsonl with scores merged by message_id."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    scores_by_msg: dict[str, dict[str, Any]] = {}
    for item in eval_result.per_message_scores:
        flat = item.get("scores", {})
        scores_by_msg[item["message_id"]] = _nest_scores(flat)

    groups = _group_samples_by_seed(samples)
    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample, scores_by_msg=scores_by_msg)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    out_path = evals_dir / "rollouts_evaluated.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    print(f"  Wrote {len(entries)} evaluated rollouts to {out_path}")
    return out_path


# ── Provenance ────────────────────────────────────────────────────────────────


def _build_system_prompts_map(
    config: ExperimentConfig,
    phases: list[Phase],
) -> dict[str, str]:
    """Build a hash->text map for all system prompts used in this experiment."""
    prompts: dict[str, str] = {}
    for text in config.system_prompts.values():
        prompts[_prompt_hash(text)] = text
    for phase in phases:
        if phase.assistant_system_prompt:
            prompts[_prompt_hash(phase.assistant_system_prompt)] = (
                phase.assistant_system_prompt
            )
        if phase.user_simulator:
            try:
                text = get_user_simulator_instruction(
                    phase.user_simulator.prompt_template
                )
                prompts[_prompt_hash(text)] = text
            except ValueError:
                pass
    default_user_sim = get_user_simulator_instruction("typical_user")
    prompts[_prompt_hash(default_user_sim)] = default_user_sim
    return prompts


def save_experiment_metadata(
    config: ExperimentConfig,
    run_dir: Path,
    experiment_name: str,
    phases: list[Phase],
) -> None:
    """Write experiment_metadata.json and a copy of the calling script to run_dir."""
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(config),
        "system_prompts": _build_system_prompts_map(config, phases),
    }
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    (rollouts_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, rollouts_dir / script_path.name)


def upload_to_hf(
    output_config: OutputPathConfig,
    run_dir: Path,
) -> None:
    """Upload run_dir to HuggingFace, mirroring the structured path.

    Uploads rollout artifacts, excluding evals, datasets, events, and exports.
    """
    if not output_config.hf_repo:
        return
    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    path_in_repo = output_config.hf_path
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=output_config.hf_repo,
        path_in_repo=path_in_repo,
        commit_message=f"Upload eval: {output_config.eval_name}{hash_suffix}",
        ignore_patterns=[
            "datasets/*",
            "events/*",
            "exports/*",
            "evals/*",
            "per_message_metrics.jsonl",
        ],
    )
    print(f"  Uploaded to {url}")


def download_rollouts_from_hf(output_config: OutputPathConfig) -> None:
    """Download rollouts and existing eval files from HF to local scratch.

    Fetches only ``rollouts/rollouts.jsonl`` and ``evals/rollouts_evaluated.jsonl``
    files under the configured HF path.
    """
    if not output_config.hf_repo:
        return

    from src.utils.hf_hub import download_from_dataset_repo

    login_from_env()

    print(
        f"Downloading rollouts from HF: {output_config.hf_repo}/{output_config.hf_path}"
    )
    download_from_dataset_repo(
        repo_id=output_config.hf_repo,
        path_in_repo=output_config.hf_path,
        local_dir=output_config.scratch_root,
        allow_patterns=[
            "*/rollouts/rollouts.jsonl",
            "*/evals/rollouts_evaluated.jsonl",
        ],
    )


def upload_evals_to_hf(
    output_config: OutputPathConfig,
    root_dir: Path,
    evals_dirs: list[Path] | None = None,
) -> None:
    """Upload evals/ subtrees to HuggingFace.

    Args:
        output_config: Output path configuration.
        root_dir: Local root directory for the sweep.
        evals_dirs: Specific evals/ directories to upload. If None, discovers
            all evals/ dirs under root_dir (slower for large trees).
    """
    if not output_config.hf_repo:
        return

    login_from_env()

    if evals_dirs is None:
        evals_dirs = [
            p.parent for p in sorted(root_dir.rglob("evals/rollouts_evaluated.jsonl"))
        ]

    if not evals_dirs:
        print("No evaluated rollouts to upload.")
        return

    print(f"Uploading evals from {len(evals_dirs)} cell(s) to HF...")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = []
    for evals_dir in evals_dirs:
        cell_dir = evals_dir.parent
        rel = cell_dir.relative_to(root_dir)
        for fpath in sorted(evals_dir.rglob("*")):
            if fpath.is_file():
                path_in_repo = f"{output_config.hf_path}/{rel}/evals/{fpath.relative_to(evals_dir)}"
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=str(fpath),
                    )
                )

    if not operations:
        print("No eval files to upload.")
        return

    api.create_commit(
        repo_id=output_config.hf_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload evals: {output_config.eval_name} ({len(evals_dirs)} cells)",
    )
    print(f"  Uploaded {len(operations)} file(s) from {len(evals_dirs)} cell(s)")


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


def _upload_plots_to_hf(
    output_config: OutputPathConfig,
    output_root: Path,
) -> None:
    """Upload plot files from the output root to HuggingFace."""
    plot_files = list(output_root.glob("*.png")) + list(output_root.glob("*.svg"))
    if not plot_files:
        return
    login_from_env()
    from huggingface_hub import HfApi

    api = HfApi()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    for plot_file in plot_files:
        path_in_repo = f"{output_config.hf_path}/{plot_file.name}"
        api.upload_file(
            path_or_fileobj=str(plot_file),
            path_in_repo=path_in_repo,
            repo_id=output_config.hf_repo,
            repo_type="dataset",
            commit_message=f"Upload plot: {plot_file.name}{hash_suffix}",
        )
        print(f"  Uploaded plot {plot_file.name}", flush=True)


def _load_completed_cells_from_hf(
    output_config: OutputPathConfig,
) -> set[str]:
    """Fetch cells with completed rollouts from HF.

    Returns a set of ``"variant/condition"`` strings that have
    ``rollouts/rollout_info.json`` with ``status == "ok"``.
    """
    if not output_config.hf_repo:
        return set()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = output_config.hf_path
    completed: set[str] = set()

    try:
        all_files = api.list_repo_tree(
            repo_id=output_config.hf_repo,
            path_in_repo=prefix,
            repo_type="dataset",
            recursive=True,
        )
        info_paths = [
            f.rfilename
            for f in all_files
            if hasattr(f, "rfilename")
            and f.rfilename.endswith("/rollouts/rollout_info.json")
        ]
    except Exception:  # noqa: BLE001
        return set()

    for rpath in info_paths:
        try:
            local_path = hf_hub_download(
                repo_id=output_config.hf_repo,
                filename=rpath,
                repo_type="dataset",
            )
            info = json.loads(Path(local_path).read_text())
            if info.get("status") == "ok":
                # Extract "variant/condition" from
                # "prefix/variant/condition/rollouts/rollout_info.json"
                rel = rpath[len(prefix) :].strip("/")
                cell_key = rel.rsplit("/rollouts/rollout_info.json", 1)[0]
                completed.add(cell_key)
        except Exception:  # noqa: BLE001
            continue

    return completed


def _upload_cell_to_hf(
    output_config: OutputPathConfig,
    cell_dir: Path,
    variant: str,
    condition: str,
) -> None:
    """Upload a single cell directory to HuggingFace."""
    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    path_in_repo = f"{output_config.hf_path}/{variant}/{condition}"
    url = upload_folder_to_dataset_repo(
        local_dir=cell_dir,
        repo_id=output_config.hf_repo,
        path_in_repo=path_in_repo,
        commit_message=f"Upload cell: {variant}/{condition}{hash_suffix}",
        ignore_patterns=[
            "datasets/*",
            "events/*",
            "exports/*",
            "evals/*",
            "per_message_metrics.jsonl",
        ],
        # If FAILED.md isn't being uploaded (successful re-run), delete
        # any stale copy left on HF from a previous failed attempt.
        delete_patterns=["FAILED.md"],
    )
    print(f"    Uploaded {variant}/{condition} to {url}", flush=True)


# Directories/files excluded from cell uploads (mirrors _upload_cell_to_hf ignore_patterns).
_CELL_UPLOAD_IGNORE = {
    "datasets",
    "events",
    "exports",
    "evals",
    "per_message_metrics.jsonl",
}


def _batch_upload_variant_cells_to_hf(
    output_config: OutputPathConfig,
    cells: list[tuple[Path, str, str]],
    successful: set[tuple[str, str]],
) -> None:
    """Upload all cells for a variant in a single HF commit.

    Args:
        output_config: Output path configuration.
        cells: List of ``(cell_dir, variant, condition)`` tuples to upload.
        successful: Unused — kept for call-site compatibility.
    """
    if not cells:
        return
    login_from_env()
    from huggingface_hub import CommitOperationAdd, HfApi

    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    api = HfApi()
    operations: list = []

    for cell_dir, variant, condition in cells:
        path_prefix = f"{output_config.hf_path}/{variant}/{condition}"
        for fpath in sorted(cell_dir.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(cell_dir)
            if rel.parts[0] in _CELL_UPLOAD_IGNORE or rel.name in _CELL_UPLOAD_IGNORE:
                continue
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{path_prefix}/{rel}",
                    path_or_fileobj=str(fpath),
                )
            )

    if not operations:
        return

    vlabel = next(v for _, v, _ in cells)
    api.create_commit(
        repo_id=output_config.hf_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload variant: {vlabel} ({len(cells)} cell(s)){hash_suffix}",
        create_pr=False,
    )
    print(
        f"    Uploaded {len(cells)} cell(s) for {vlabel} to HF in 1 commit",
        flush=True,
    )


def _write_run_info(
    run_dir: Path,
    variant: str,
    condition: str,
    status: str,
    aggregates: dict[str, Any] | None,
    error: str | None,
    elapsed: float | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_info.json"
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "condition": condition,
                "status": status,
                "aggregates": aggregates,
                "error": error,
                "elapsed_seconds": round(elapsed, 2) if elapsed is not None else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


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
) -> tuple[io.TextIOBase, io.TextIOBase, logging.Handler]:
    """Tee stdout/stderr to a log file and add a file handler for the logger.

    Returns (original_stdout, log_file_handle, logger_file_handler) for cleanup.
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

    if (
        config.evaluations
        and not config.skip_evals
        and config.plot
        and config.plot_metric
    ):
        try:
            from src.visualisations.plot_rollout_sweep import plot_sweep

            plot_sweep(output_root, metric_key=config.plot_metric)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: plot generation failed: {exc}", flush=True)

        # Upload plots to HF (cell data is uploaded per-variant in run_sweep).
        if config.output.hf_repo:
            _upload_plots_to_hf(config.output, output_root)

    _teardown_file_logging(log_file, log_handler)
    return output_root


# ── Condition naming ──────────────────────────────────────────────────────────


def _phase_label(
    num_turns: int,
    ast_prompted: bool,
    usr_prompted: bool | None = None,
    is_aa: bool = False,
) -> str:
    """Build a single phase label.

    Args:
        num_turns: Number of back-and-forth turns.
        ast_prompted: Whether the assistant has a system prompt.
        usr_prompted: Whether the user/2nd-assistant is prompted.
            ``None`` means no user role (single-turn).
        is_aa: If True, the user role is a second assistant.

    Returns:
        e.g. ``"3turn_astSProm_usrNoSProm"``
    """
    ast = "astSProm" if ast_prompted else "astNoSProm"
    parts = [f"{num_turns}turn", ast]
    if usr_prompted is not None:
        prefix = "ast2" if is_aa else "usr"
        tag = "SProm" if usr_prompted else "NoSProm"
        parts.append(f"{prefix}{tag}")
    return "_".join(parts)


def _build_condition_name(
    phase_specs: list[tuple[int, bool, bool | None]],
    trait: str,
    is_aa: bool = False,
) -> str:
    """Build a full condition name from phase specs and trait.

    Args:
        phase_specs: List of ``(num_turns, ast_prompted, usr_prompted)``
            tuples. ``usr_prompted=None`` means no user role.
        trait: Trait name (e.g. ``"t_avoiding"``).
        is_aa: Whether the user role is a second assistant.

    Returns:
        e.g. ``"3turn_astSProm_usrNoSProm___1turn_astNoSProm___t_avoiding"``
    """
    phase_labels = [
        _phase_label(turns, ast_p, usr_p, is_aa=is_aa)
        for turns, ast_p, usr_p in phase_specs
    ]
    return "___".join(phase_labels) + "___" + trait


# ── Condition template factories ──────────────────────────────────────────────


def single_turn_conditions(
    behavior_prompts: dict[str, str | None],
) -> list[SweepCondition]:
    """Create single-turn conditions from a behavior prompt dict.

    Condition names are generated using phase notation, e.g.
    ``1turn_astSProm___t_avoiding``.

    Args:
        behavior_prompts: Mapping of trait name to system prompt text.
            Use ``None`` for no system prompt (baseline).

    Returns:
        One ``SweepCondition`` per entry.

    Example::

        single_turn_conditions({
            "baseline": None,
            "t_avoiding": "You are a helpful assistant. ...",
        })
    """
    conditions = []
    for trait, prompt in behavior_prompts.items():
        ast_prompted = prompt is not None
        cond_name = _build_condition_name([(1, ast_prompted, None)], trait)
        conditions.append(
            SweepCondition(
                name=cond_name,
                phases=[
                    Phase(
                        num_turns=1,
                        assistant_system_prompt=prompt,
                    )
                ],
            )
        )
    return conditions


def multi_turn_au_conditions(
    config: ExperimentConfig,
    behavior_prompts: dict[str, str | None],
    user_behavior_templates: dict[str, str],
    turns_per_phase: tuple[int, int] = (3, 1),
) -> list[SweepCondition]:
    """Create multi-turn assistant-user conditions.

    For each non-None behavior prompt, creates:
    - ``assistant_{name}``: assistant prompted in phase 1, unprompted in phase 2
    - ``user_{name}``: user simulator prompted in phase 1, unprompted in phase 2

    For None-valued entries, creates a baseline condition with no prompting.

    User simulator templates are registered automatically via
    ``register_user_simulator_template``.

    Args:
        config: Experiment config (for building user simulator configs).
        behavior_prompts: Mapping of condition name to assistant system prompt.
            None = no system prompt (baseline).
        user_behavior_templates: Mapping of condition name to user simulator
            template text. Keys should match behavior_prompts keys.
        turns_per_phase: ``(phase1_turns, phase2_turns)``.

    Returns:
        List of SweepConditions.
    """
    p1, p2 = turns_per_phase
    default_user_sim = build_user_simulator(config, "typical_user")
    conditions = []

    for trait, prompt in behavior_prompts.items():
        if prompt is None:
            # Baseline: no prompting in either phase.
            cond_name = _build_condition_name(
                [(p1, False, False), (p2, False, None)],
                trait,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            user_simulator=default_user_sim,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=default_user_sim,
                        ),
                    ],
                )
            )
        else:
            # Assistant-prompted condition.
            cond_name = _build_condition_name(
                [(p1, True, False), (p2, False, None)],
                trait,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            assistant_system_prompt=prompt,
                            user_simulator=default_user_sim,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=default_user_sim,
                        ),
                    ],
                )
            )

            # User-prompted condition (if template provided).
            if trait in user_behavior_templates:
                user_template_name = f"{trait}_user"
                register_user_simulator_template(
                    user_template_name,
                    user_behavior_templates[trait],
                )
                user_sim_prompted = build_user_simulator(config, user_template_name)
                cond_name = _build_condition_name(
                    [(p1, False, True), (p2, False, None)],
                    trait,
                )
                conditions.append(
                    SweepCondition(
                        name=cond_name,
                        phases=[
                            Phase(
                                num_turns=p1,
                                user_simulator=user_sim_prompted,
                            ),
                            Phase(
                                num_turns=p2,
                                user_simulator=default_user_sim,
                            ),
                        ],
                    )
                )

    return conditions


def multi_turn_aa_conditions(
    config: ExperimentConfig,
    behavior_prompts: dict[str, str | None],
    aa_templates: dict[str, str],
    turns_per_phase: tuple[int, int] = (3, 1),
) -> list[SweepCondition]:
    """Create assistant-assistant conditions (both sides are LLMs).

    For each non-None behavior prompt, creates an ``aa_{name}`` condition where
    both the assistant and the "user" (second assistant) are prompted in phase 1,
    then unprompted in phase 2.

    For None-valued entries, creates an ``aa_baseline`` with no prompting.

    Args:
        config: Experiment config (for building user simulator configs).
        behavior_prompts: Mapping of condition name to assistant system prompt.
            None = no system prompt (baseline).
        aa_templates: Mapping of condition name to second-assistant (user-side)
            template text. Must also include a baseline key (e.g. ``"baseline"``).
        turns_per_phase: ``(phase1_turns, phase2_turns)``.

    Returns:
        List of SweepConditions.
    """
    p1, p2 = turns_per_phase
    conditions = []

    # Register AA templates and build user simulators.
    for template_name, template_text in aa_templates.items():
        register_user_simulator_template(f"aa_{template_name}", template_text)

    for trait, prompt in behavior_prompts.items():
        # AA user sim: uses the assistant model/provider, chat_messages format.
        aa_user_base = build_user_simulator(
            config,
            f"aa_{trait}",
            "chat_messages",
            provider=config.assistant_provider,
            model=config.assistant_model,
        )

        if prompt is None:
            # AA baseline: no behavioral prompting.
            cond_name = _build_condition_name(
                [
                    (p1, False, False),
                    (p2, False, False),
                ],
                trait,
                is_aa=True,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            user_simulator=aa_user_base,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=aa_user_base,
                        ),
                    ],
                    eval_roles=None,
                )
            )
        else:
            # AA prompted: both sides prompted in phase 1,
            # unprompted in phase 2.
            aa_user_prompted = build_user_simulator(
                config,
                f"aa_{trait}",
                "chat_messages",
                provider=config.assistant_provider,
                model=config.assistant_model,
            )
            aa_user_unprompted = build_user_simulator(
                config,
                "aa_baseline",
                "chat_messages",
                provider=config.assistant_provider,
                model=config.assistant_model,
            )
            cond_name = _build_condition_name(
                [
                    (p1, True, True),
                    (p2, False, False),
                ],
                trait,
                is_aa=True,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            assistant_system_prompt=prompt,
                            user_simulator=aa_user_prompted,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=aa_user_unprompted,
                        ),
                    ],
                    eval_roles=None,
                )
            )

    return conditions

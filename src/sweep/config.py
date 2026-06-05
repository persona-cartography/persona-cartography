"""Configuration types and config builders for rollout sweeps.

This is the leaf module of the :mod:`src.sweep` package — it defines the
dataclasses / pydantic models that describe a sweep (:class:`ExperimentConfig`,
:class:`Phase`, :class:`SweepCondition`, :class:`OutputPathConfig`,
:class:`SweepConfig`) plus the small constructors that turn an
:class:`ExperimentConfig` into the lower-level inference / dataset / user-sim
configs the rollout engine consumes. It imports nothing else from
``src.sweep``, so every other module in the package can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.common.config import DatasetConfig, GenerationConfig
from src.inference.config import InferenceConfig, LocalProviderConfig, RetryConfig
from src.persona_metrics.config import PersonaMetricSpec
from src.rollout_generation.config import UserSimulatorConfig
from src.rollout_generation.model_providers import ModelProvider

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

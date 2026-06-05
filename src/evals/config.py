"""Configuration models for the all-Inspect evals module.

Pydantic schema for everything a suite run needs:

- **What to run on** — :class:`ModelSpec` (a concrete model: base + adapters +
  dtype), or one of the sweep shorthands on :class:`SuiteConfig`
  (:class:`ScaleSweep` for LoRA scale sweeps, :class:`ActivationCapSweep` for
  fraction-based activation-capping sweeps) that auto-expand into one
  ``ModelSpec`` per point via :meth:`SuiteConfig.expand_models`.
- **What to run** — :class:`InspectBenchmarkSpec` (Inspect-native benchmark) or
  :class:`InspectCustomEvalSpec` (prompt-built custom eval scored by persona
  metrics / a custom scorer), unified as :data:`EvalSpec`.
- **How to score custom evals** — :class:`JudgeExecutionConfig`
  (blocking / submit / resume).
- **Results bookkeeping** — :class:`RunSummaryRow`, :class:`SuiteResult`.

These are pure data models with validators; all execution lives in
:mod:`src.evals.suite`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.common.config import DatasetConfig, GenerationConfig
from src.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec


class AdapterConfig(BaseModel):
    """A single LoRA adapter with its scaling factor."""

    path: str
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"scale must be finite, got {value}")
        return value


class ScaleSweep(BaseModel):
    """LoRA scale sweep parameters.

    Defines a linear grid of adapter scaling factors from *min* to *max*
    (inclusive) at *step* intervals.  The base model (scale=0, no adapter)
    is always included automatically.

    Alternatively, pass an explicit *points* list to use an arbitrary set of
    scale values (e.g. non-uniform grids).  When *points* is set, *min*,
    *max*, and *step* are ignored.

    The sweep is defined once at suite level and can be overridden
    per-eval via ``InspectBenchmarkSpec.sweep``.
    """

    min: float = -2.0
    max: float = 2.0
    step: float = 0.25
    points: list[float] | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> "ScaleSweep":
        if self.points is not None:
            return self
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self

    def scale_points(self) -> list[float]:
        """Return the sorted list of scale values, excluding 0.0 (which is the base model)."""
        if self.points is not None:
            return sorted(s for s in self.points if s != 0.0)
        n_steps = round((self.max - self.min) / self.step)
        return [
            s
            for s in (round(self.min + i * self.step, 10) for i in range(n_steps + 1))
            if s != 0.0
        ]


class ActivationCapSweep(BaseModel):
    """Activation capping sweep: fraction-based sweep along a pre-computed axis.

    Each fraction interpolates (or extrapolates) between the base model's
    typical projection onto the axis (fraction=0.0) and the LoRA model's
    typical projection (fraction=1.0).

    Positive fractions use floor capping (push activations toward the trait
    direction). Negative fractions use ceiling capping (suppress the trait).
    The base model (fraction=0.0) is always included automatically.

    Args:
        fractions: Explicit list of fraction values to sweep (0.0 excluded;
            the base model is always added as the "base" spec).
        axis_path: Path to a .pt file containing ``{"axis": Tensor, "metadata": {...}}``.
        per_layer_range_path: Path to a .pt file containing
            ``{"per_layer_range": {layer_idx: (min_proj, max_proj)}}``.
        capping_layers: Which layers to apply capping to.  When None, reads
            ``recommended_capping_layers`` from the axis metadata.
    """

    fractions: list[float]
    axis_path: str
    per_layer_range_path: str
    capping_layers: list[int] | None = None
    ceiling_from_hi: bool = True


class ModelSpec(BaseModel):
    """Model configuration for suite runs."""

    name: str
    base_model: str
    model_uri: str | None = None
    adapters: list[AdapterConfig] = Field(default_factory=list)
    dtype: str = "bfloat16"
    device_map: str = "auto"
    inspect_model_args: dict[str, Any] = Field(default_factory=dict)
    # Scale stored for downstream analysis; None for the base model.
    scale: float | None = None


class InspectBenchmarkSpec(BaseModel):
    """Inspect benchmark evaluation specification."""

    kind: Literal["benchmark"] = "benchmark"
    name: str
    benchmark: str
    benchmark_args: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = None
    # Number of independent runs per (model, eval) pair for noise estimation.
    n_runs: int = 1
    # Per-eval sweep override. When set, this eval uses its own scale grid
    # instead of the suite-level sweep (e.g. coarser steps for MMLU).
    sweep: ScaleSweep | None = None


class InspectCustomEvalSpec(BaseModel):
    """Inspect custom evaluation specification."""

    kind: Literal["custom"] = "custom"
    name: str
    dataset: DatasetConfig
    input_builder: str
    target_builder: str | None = None
    evaluations: list[str | PersonaMetricSpec] = Field(default_factory=list)
    scorer_builder: str | None = None
    scorer_builder_kwargs: dict[str, Any] = Field(default_factory=dict)
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    metrics_key: str = "persona_metrics"

    @model_validator(mode="after")
    def _validate_scoring_configuration(self) -> "InspectCustomEvalSpec":
        if not self.evaluations and not self.scorer_builder:
            raise ValueError(
                "custom eval must define at least one of: evaluations or scorer_builder"
            )
        return self


EvalSpec = InspectBenchmarkSpec | InspectCustomEvalSpec
JudgeExecutionMode = Literal["blocking", "submit", "resume"]


class JudgeExecutionConfig(BaseModel):
    """Judge execution behavior for custom evals."""

    mode: JudgeExecutionMode = "blocking"
    prefer_batch: bool = True
    poll_interval_seconds: int = 30
    timeout_seconds: int | None = None
    inspect_batch: bool | int | dict[str, Any] | None = None


class SuiteConfig(BaseModel):
    """Top-level suite configuration.

    Sweep mode
    ----------
    When *sweep* and *adapter* are both set, the suite automatically expands
    into one ModelSpec per scale point (plus a base model at scale=0).

    When *activation_cap* is set (with *base_model*), the suite expands into
    one ModelSpec per fraction (plus a base model at fraction=0).  The
    ``scale`` field of each ModelSpec stores the fraction value for downstream
    analysis compatibility.

    The explicit *models* list is used instead when neither *sweep* nor
    *activation_cap* is set, preserving full manual control.

    Each eval can override the suite-level sweep via
    ``InspectBenchmarkSpec.sweep`` (e.g. for a coarser MMLU scale grid).
    """

    # --- Sweep shorthand (mutually exclusive with explicit models list) ---
    base_model: str | None = None
    adapter: str | None = None
    sweep: ScaleSweep | None = None
    # Optional adapters merged into base weights before the sweep adapter is
    # loaded.  Useful for sweeping a trait LoRA on top of a fixed persona LoRA.
    fixed_adapters: list[AdapterConfig] = Field(default_factory=list)
    # --- Activation capping sweep (mutually exclusive with sweep / models) ---
    activation_cap: ActivationCapSweep | None = None

    # --- Explicit model list (used when sweep / activation_cap is not set) ---
    models: list[ModelSpec] = Field(default_factory=list)

    evals: list[EvalSpec]
    output_root: Path
    run_name: str | None = None
    skip_completed: bool = True
    # Generation temperature forwarded to Inspect for all benchmark evals.
    temperature: float = 0.0
    # Batch size for model generation. When None, Inspect uses its own default.
    batch_size: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional HF Hub path for Inspect to write logs directly during the run
    # (e.g. "hf://datasets/org/repo"). When None, logs are written locally only.
    hf_log_dir: str | None = None
    # Optional: upload the run directory to a HuggingFace dataset repo after completion.
    # upload_repo_id: e.g. "persona-shattering-lasr/monorepo"
    # upload_path_in_repo: destination prefix, e.g.
    #   "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/evals/mcq"
    # Supports {eval_name} template variable to split uploads per eval.
    upload_repo_id: str | None = None
    upload_path_in_repo: str | None = None
    # Optional: run analyze_results.generate_plots() after the sweep completes.
    # analyze_kwargs passes through to generate_plots() (e.g. random_baseline, spread, title_suffix).
    auto_analyze: bool = False
    analyze_kwargs: dict[str, Any] = Field(default_factory=dict)

    # --- Performance toggles ---
    # Cache Inspect tasks across scale points so dataset loading / choice
    # shuffling happens once instead of once per scale point.
    cache_tasks: bool = True
    # Cache per-sample tokenisation across scale points.  The prompts are
    # identical for every scale point in a sweep, so the token IDs only need
    # to be computed once.
    cache_tokenization: bool = True
    # Apply torch.compile to the model before the sweep.  Can give 10-30 %
    # faster forward passes depending on the model architecture.  Off by
    # default because it adds a one-time compilation latency and may not be
    # compatible with all model / PEFT combinations.
    torch_compile: bool = False

    @model_validator(mode="after")
    def _validate_model_source(self) -> "SuiteConfig":
        has_sweep = self.sweep is not None
        has_cap = self.activation_cap is not None
        has_models = bool(self.models)
        n_sources = sum([has_sweep, has_cap, has_models])
        if n_sources > 1:
            raise ValueError(
                "Provide exactly one of: 'sweep' + 'base_model', "
                "'activation_cap' + 'base_model', or explicit 'models'."
            )
        if n_sources == 0:
            raise ValueError(
                "Provide one of: 'sweep' + 'base_model', "
                "'activation_cap' + 'base_model', or an explicit 'models' list."
            )
        if (has_sweep or has_cap) and not self.base_model:
            raise ValueError(
                "'base_model' is required when 'sweep' or 'activation_cap' is set."
            )
        return self

    @field_validator("evals")
    @classmethod
    def _non_empty_evals(cls, value: list[EvalSpec]) -> list[EvalSpec]:
        if not value:
            raise ValueError("evals must not be empty")
        return value

    def expand_models(self) -> list[ModelSpec]:
        """Return the full ModelSpec list, expanding the sweep if configured.

        When using explicit *models*, returns them as-is.
        When using sweep mode, builds one ModelSpec per scale point plus
        a base model (scale=0).  Each eval's per-eval sweep override is
        taken into account to ensure all required scale points are present.
        When using activation_cap mode, builds one ModelSpec per fraction plus
        a base model (fraction=0, stored as scale=None).
        """
        if self.activation_cap is not None:
            assert self.base_model is not None  # validated above
            fracs = sorted(f for f in self.activation_cap.fractions if f != 0.0)
            specs: list[ModelSpec] = [
                ModelSpec(name="base", base_model=self.base_model, scale=None)
            ]
            for frac in fracs:
                frac_tag = f"{frac:+.2f}".replace(".", "p")  # e.g. +1.00 → +1p00
                specs.append(
                    ModelSpec(
                        name=f"cap_{frac_tag}",
                        base_model=self.base_model,
                        scale=frac,
                    )
                )
            return specs

        if not self.sweep:
            return self.models

        assert self.base_model is not None  # validated above

        # Collect all scale points needed across all evals (union of grids).
        all_scales: set[float] = set()
        for eval_spec in self.evals:
            sweep = (
                eval_spec.sweep
                if isinstance(eval_spec, InspectBenchmarkSpec)
                and eval_spec.sweep is not None
                else self.sweep
            )
            all_scales.update(sweep.scale_points())

        specs: list[ModelSpec] = [
            ModelSpec(name="base", base_model=self.base_model, scale=None)
        ]
        for scale in sorted(all_scales):
            scale_tag = f"{scale:+.2f}".replace(".", "p")  # e.g. +1.25 -> +1p25
            adapter_configs = (
                [AdapterConfig(path=self.adapter, scale=scale)] if self.adapter else []
            )
            specs.append(
                ModelSpec(
                    name=f"lora_{scale_tag}x",
                    base_model=self.base_model,
                    adapters=adapter_configs,
                    scale=scale,
                )
            )
        return specs


class RunSummaryRow(BaseModel):
    """Standardized summary row for a single model/eval run."""

    backend: str = "inspect"
    model_name: str
    model_spec_name: str
    eval_name: str
    eval_kind: Literal["benchmark", "custom"]
    status: Literal["ok", "pending", "failed", "skipped"]
    output_dir: str
    run_info_path: str | None = None
    inspect_log_path: str | None = None
    error: str | None = None


class SuiteResult(BaseModel):
    """Result metadata for a suite run."""

    output_root: Path
    rows: list[RunSummaryRow] = Field(default_factory=list)

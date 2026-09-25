"""Editing stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src_dev.common.persona_registry import DEFAULT_PERSONA
from src_dev.persona_metrics.config import PersonaMetricSpec, JudgeLLMConfig


class RetryConfig(BaseModel):
    """API retry configuration."""

    max_retries: int = 3
    backoff_factor: float = 2.0


class AnthropicProviderConfig(BaseModel):
    """Anthropic-specific settings."""

    max_tokens: int = 1024


class OpenAIProviderConfig(BaseModel):
    """OpenAI-specific settings."""

    model: str | None = None  # Override model for OpenAI (if different from main model)
    max_tokens: int = 20000
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None


class CodeProviderConfig(BaseModel):
    """Code-based editor settings."""

    editor: str = "src_dev.lora_pipeline_legacy.editing.code_editors:reverse_text"


class QualityConfig(BaseModel):
    """Post-edit quality evaluation configuration.

    Evaluations run on both original and edited responses. For each metric key
    produced by an evaluation, the editing stage stores:
    - ``<metric>.original``
    - ``<metric>.edited``
    - ``<metric>.delta`` (numeric metrics only)
    """

    enabled: bool = True
    evaluations: list[str | PersonaMetricSpec] | None = None
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    metrics_key: str = "quality_metrics"
    persona: str = DEFAULT_PERSONA
    on_error: Literal["warn", "raise"] = "warn"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Support old config shape where `metrics` listed quality checks."""
        if not isinstance(data, dict):
            return data
        updated = dict(data)
        if "evaluations" not in updated and "metrics" in updated:
            updated["evaluations"] = updated.pop("metrics")
        updated.pop("reporters", None)
        return updated


class EditingConfig(BaseModel):
    """Configuration for the editing stage.

    Example:
        config = EditingConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            prompt_template="default_persona_shatter",
            max_concurrent=10,
            output_path=Path("scratch/edited_dataset.jsonl"),
        )
        dataset, result = run_editing(config, input_dataset)
    """

    # Provider settings
    provider: str = "anthropic"  # "anthropic", "openai", or "code"
    model: str = "claude-sonnet-4-20250514"
    prompt_template: str = "default_persona_shatter"

    # Sampling parameters for the editing LLM
    temperature: float = 0.7
    top_p: float = 0.95

    # Concurrency and timeout
    max_concurrent: int = 10
    timeout: int = 60

    # Retry settings
    retry: RetryConfig = RetryConfig()

    # Provider-specific settings
    anthropic: AnthropicProviderConfig = AnthropicProviderConfig()
    openai: OpenAIProviderConfig = OpenAIProviderConfig()
    code: CodeProviderConfig = CodeProviderConfig()

    # Quality metrics
    quality: QualityConfig = QualityConfig()

    # Output
    output_path: Path | None = None  # If None, returns dataset without saving
    run_dir: Path | None = None  # Canonical run directory under scratch/runs/<run_id>
    variant_name: str | None = None  # Required for canonical run-dir mode
    total_turns_hint: int | None = None
    max_attempts_per_sample: int | None = 3
    resume: bool = True
    overwrite_output: bool = False
    io_batch_size: int = 100


class EditingResult(BaseModel):
    """Result from running editing."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_samples: int = 0
    num_failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    quality_error: str | None = None

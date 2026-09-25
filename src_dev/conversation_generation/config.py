"""Configuration for multi-turn conversation dataset generation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.lora_pipeline_legacy.editing import EditingConfig
from src_dev.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)


class ResponderConfig(BaseModel):
    """Configuration for generating the next user turn."""

    provider: str = "openai"
    model: str = "gpt-5-nano-2025-08-07"
    prompt_template: str = "natural_partner"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    max_concurrent: int = 10
    timeout: int = 60
    retry: RetryConfig = Field(default_factory=RetryConfig)
    local: LocalProviderConfig = Field(default_factory=LocalProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(default_factory=OpenRouterProviderConfig)
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)


class ConversationGenerationConfig(BaseModel):
    """Configuration for the multi-turn conversation generator."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    run_dir: Path
    num_assistant_turns: int
    system_prompt: str | None = None

    assistant_inference: InferenceConfig
    editing: EditingConfig
    responder: ResponderConfig

    editing_variant: str
    responder_variant: str = "natural_partner"
    resume: bool = True
    overwrite_output: bool = False


class ConversationGenerationResult(BaseModel):
    """Result metadata for conversation generation."""

    output_path: Path | None = None
    num_conversations: int = 0
    num_completed: int = 0
    num_failed: int = 0
    num_assistant_turns_target: int = 0
    num_assistant_turns_completed: int = 0
    exports: dict[str, str] = Field(default_factory=dict)

"""Configuration models for long-context rollout generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.common.config import DatasetConfig, GenerationConfig
from src.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)


class UserSimulatorConfig(BaseModel):
    """Configuration for generating the next user turn with a strong LLM."""

    provider: str = "openai"
    model: str = "gpt-5-nano-2025-08-07"
    prompt_template: str = "typical_user"
    prompt_format: Literal["chat_messages", "single_turn_text"] = "single_turn_text"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    max_concurrent: int = 16
    timeout: int = 60
    retry: RetryConfig = Field(default_factory=RetryConfig)
    local: LocalProviderConfig = Field(default_factory=LocalProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    openrouter: OpenRouterProviderConfig = Field(
        default_factory=OpenRouterProviderConfig
    )
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)

    # Role-flipping: when True, swap user↔assistant roles in messages before
    # sending to the user simulator.  This lets the simulator see itself as
    # "assistant" while its output is still stored as a "user" turn.
    flip_roles_in_prompt: bool = False
    # Prepended as a {"role": "user"} message in the flipped view (e.g. a
    # filler greeting so the simulator's first generated message appears as
    # an "assistant" reply).  Only used when flip_roles_in_prompt is True.
    initial_message_in_flipped_view: str | None = None

    # Higher-level flip mode that supersedes flip_roles_in_prompt when set.
    #   "none"          — no role manipulation (default, same as flip_roles_in_prompt=False).
    #   "swap_roles"    — legacy swap, equivalent to flip_roles_in_prompt=True.
    #   "interlocutor"  — swap roles AND ensure the message sequence starts
    #                     with a "user" message (the AI assistant's side in
    #                     the flipped view).  This matches the standard
    #                     chat-completion convention where the first non-system
    #                     message is "user" and the model generates as
    #                     "assistant".  The user-sim's prior outputs appear as
    #                     "assistant" messages, the test model's responses
    #                     appear as "user" messages.
    flip_mode: Literal["none", "swap_roles", "interlocutor"] = "none"

    # Optional per-turn reminder injected as the final user message in the
    # flipped view before each user-sim generation.  Helps prevent the
    # user-sim from drifting into assistant-mode over long conversations.
    turn_reminder: str | None = None

    # Maximum number of recent conversation turns (user+assistant pairs) to
    # include in the user-sim prompt.  When set, older turns are dropped and
    # a short note is prepended telling the user-sim which turn the excerpt
    # starts from.  Helps prevent assistant-mode drift by keeping the context
    # window short and focused.  None means full history (no truncation).
    max_context_turns: int | None = None


class ContextPolicyConfig(BaseModel):
    """Context-window policy for rollout prompting."""

    mode: Literal["full_history", "token_budget"] = "full_history"
    assistant_max_context_tokens: int | None = None
    user_max_context_tokens: int | None = None


class FailurePolicyConfig(BaseModel):
    """Per-turn retry limits before marking a sample terminal."""

    assistant_max_attempts_per_turn: int = 3
    user_max_attempts_per_turn: int = 3


class RolloutGenerationConfig(BaseModel):
    """Configuration for assistant<->user long-context rollout generation."""

    model_config = {"arbitrary_types_allowed": True}

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    run_dir: Path
    num_assistant_turns: int
    num_rollouts_per_prompt: int = 1
    system_prompt: str | None = None

    assistant_inference: InferenceConfig
    user_simulator: UserSimulatorConfig = Field(default_factory=UserSimulatorConfig)

    transcript_variant: str = "rollout_base"
    context_policy: ContextPolicyConfig = Field(default_factory=ContextPolicyConfig)
    failure_policy: FailurePolicyConfig = Field(default_factory=FailurePolicyConfig)

    skip_final_user_turn: bool = True
    user_sim_generates_opening: bool = False
    resume: bool = True
    overwrite_output: bool = False
    # Optional list of sample_ids that were previously marked terminal and
    # should be retried from their current partial transcript on resume.
    # Excluded from serialized stage config so the same run directory can be
    # resumed with different retry selections without fingerprint mismatch.
    retry_terminal_sample_ids: list[str] = Field(default_factory=list, exclude=True)

    # Per-sample prompt template overrides for the user simulator.
    # Maps sample_id → registered template name.  When present for a sample,
    # takes precedence over user_simulator.prompt_template for that sample.
    # Use register_user_simulator_template to register templates before passing
    # them here.
    prompt_template_per_sample: dict[str, str] = Field(default_factory=dict)

    # Optional pre-built inference provider.  When set, model loading inside
    # run_rollout_generation is skipped and this provider is used directly.
    # Analogous to preloaded_model for the local provider — used by the vLLM
    # sweep to share one engine across all (scale, condition) cells.
    preloaded_provider: object | None = Field(default=None, exclude=True)


class RolloutGenerationResult(BaseModel):
    """Result metadata for rollout generation."""

    output_path: Path | None = None
    num_conversations: int = 0
    num_completed: int = 0
    num_failed: int = 0
    num_assistant_turns_target: int = 0
    num_assistant_turns_completed: int = 0
    exports: dict[str, str] = Field(default_factory=dict)

"""Canonical dataset and lineage schema for pipeline runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0.0"


class CanonicalMessage(BaseModel):
    """A canonical chat message."""

    model_config = ConfigDict(extra="allow")

    message_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_metadata: dict[str, Any] | None = None
    message_metadata: dict[str, Any] | None = None
    editable: bool = True


class CanonicalInput(BaseModel):
    """Canonical input bundle for a sample."""

    model_config = ConfigDict(extra="allow")

    messages: list[CanonicalMessage]
    assistant_prefill: str | None = None
    system_prompt_ref: str | None = None


class InferenceData(BaseModel):
    """Inference result metadata for a sample."""

    model_config = ConfigDict(extra="allow")

    status: Literal["pending", "success", "failed"] = "pending"
    model: str | None = None
    provider: str | None = None
    assistant_message_id: str | None = None
    assistant_prefill: str | None = None
    assistant_completion: str | None = None
    assistant_full: str | None = None
    system_prompt_ref: str | None = None
    attempt_no: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class EditOverlay(BaseModel):
    """An edit overlay that targets one message in the conversation."""

    model_config = ConfigDict(extra="allow")

    overlay_id: str
    target_message_id: str
    target_role: Literal["system", "user", "assistant", "tool"]
    original_content_hash: str
    edited_content: str
    status: Literal["pending", "success", "failed"]
    attempt_no: int
    editor_model: str | None = None
    editor_provider: str | None = None
    edit_prompt_hash: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    judge_metadata: dict[str, Any] | None = None
    timestamps: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class EditVariant(BaseModel):
    """Named collection of overlays for a sample."""

    model_config = ConfigDict(extra="allow")

    variant_name: str
    status: Literal["pending", "success", "failed"] = "pending"
    overlays: list[EditOverlay] = Field(default_factory=list)


class MetricAnnotationRecord(BaseModel):
    """A metric annotation attached to a candidate reference."""

    model_config = ConfigDict(extra="allow")

    annotation_id: str
    sample_id: str
    candidate_ref: str
    metrics_key: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    evaluator_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class SampleRecord(BaseModel):
    """Canonical sample row."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    input_group_id: str | None = None
    response_index: int = 0
    source_info: dict[str, Any] = Field(default_factory=dict)
    input: CanonicalInput
    messages: list[CanonicalMessage] = Field(default_factory=list)
    inference: InferenceData = Field(default_factory=InferenceData)
    edit_variants: list[EditVariant] = Field(default_factory=list)
    metrics: dict[str, list[MetricAnnotationRecord]] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)


class StageEventRecord(BaseModel):
    """Append-only stage event."""

    model_config = ConfigDict(extra="allow")

    event_id: str
    stage: str
    event_type: str
    created_at: str
    sample_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SystemPromptRecord(BaseModel):
    """System prompt metadata recorded in manifest."""

    model_config = ConfigDict(extra="allow")

    prompt_id: str
    prompt_hash: str
    inline_text: str | None = None
    created_at: str


class ExportProfile(BaseModel):
    """Export profile controls."""

    model_config = ConfigDict(extra="allow")

    name: str
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    rename: dict[str, str] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Run manifest for canonical dataset flows."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str
    updated_at: str
    git_commit_hash: str | None = None
    cli_command: str | None = None
    dataset_fingerprint: str | None = None
    stage_fingerprints: dict[str, str] = Field(default_factory=dict)
    system_prompts: dict[str, SystemPromptRecord] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    base_config: dict[str, Any] = Field(default_factory=dict)

"""Shared configuration types used across components.

These are the common building blocks that multiple components need.
Component-specific configs live in their respective modules.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Base model configuration for HuggingFace models."""

    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"


class GenerationConfig(BaseModel):
    """Text generation parameters."""

    max_new_tokens: int = 100000
    temperature: float = 0.7
    top_p: float = 1.0
    do_sample: bool = True
    batch_size: int = 8
    num_responses_per_prompt: int = 1


class DatasetConfig(BaseModel):
    """Dataset source configuration."""

    source: str = "huggingface"  # "huggingface", "local", or "canonical"
    name: str | None = None  # HuggingFace dataset name
    path: str | None = None  # Local path for source="local"
    split: str = "train"
    max_samples: int | None = None
    seed: int | None = None  # Random seed for deterministic shuffling before max_samples selection


class WandbConfig(BaseModel):
    """Weights & Biases tracking configuration."""

    enabled: bool = True
    project: str = "persona-shattering-v1"
    entity: str | None = None
    name: str | None = None  # Run name (shown in wandb UI)
    tags: list[str] = []
    group: str | None = None
    log_model: bool = True
    log_dataset: bool = True

    # Artifact upload flags - require human confirmation after successful run
    upload_inference_to_wandb: bool = False
    upload_editing_to_wandb: bool = False
    upload_checkpoints_to_wandb: bool = False

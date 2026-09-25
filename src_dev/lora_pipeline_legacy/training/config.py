"""Training stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src_dev.common.config import ModelConfig, WandbConfig
from src_dev.persona_metrics.config import PersonaMetricSpec, JudgeLLMConfig


class LoraConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    task_type: str = "CAUSAL_LM"


class SftConfig(BaseModel):
    """SFT training hyperparameters."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    fp16: bool = False
    bf16: bool = True


class CheckpointConfig(BaseModel):
    """Checkpoint saving configuration."""

    save_strategy: str = "epoch"  # "epoch" or "steps"
    save_steps: int = 100
    save_total_limit: int = 3


class TrainingMetricsConfig(BaseModel):
    """Training-time metric logging configuration."""

    enabled: bool = True
    log_grad_norm: bool = True
    log_param_norm: bool = True
    log_update_norm: bool = False


class TrainingEvaluationConfig(BaseModel):
    """Evaluation configuration for training-time model checks."""

    enabled: bool = True
    evaluations: list[str | PersonaMetricSpec] = ["count_o"]
    judge: JudgeLLMConfig = JudgeLLMConfig()

    # Generation settings for model-evaluated outputs
    num_samples: int = 20
    max_new_tokens: int = 128
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Scheduling
    eval_every_n_steps: int | None = None
    eval_every_n_epochs: int = 1

    # Dataset column mapping for evaluation
    response_column: str = "response"
    question_column: str | None = "question"
    metrics_key: str = "persona_metrics"

    # Optional W&B sample table logging
    log_samples: bool = True
    log_samples_every_n_evals: int = 1


class TrainingConfig(BaseModel):
    """Configuration for the training stage.

    Example:
        config = TrainingConfig(
            dataset_path=Path("scratch/data/train.jsonl"),
            user_column="question",
            assistant_column="response",
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
            checkpoint_dir=Path("scratch/checkpoints"),
        )
        val_dataset, result = run_training(config)
    """

    # Model configuration
    model: ModelConfig = ModelConfig()

    # LoRA configuration
    lora: LoraConfig = LoraConfig()

    # SFT configuration
    sft: SftConfig = SftConfig()

    # Checkpointing
    checkpoint: CheckpointConfig = CheckpointConfig()

    # Training data source and column mapping
    dataset_path: Path
    user_column: str
    assistant_column: str
    group_column: str | None = None

    # Prompt formatting
    plain_prompt_template: str = "### User:\n{user}\n\n### Assistant:\n"
    prompt_format: Literal["auto", "chat", "plain"] = "auto"
    chat_system_prompt: str | None = None

    # Wandb logging
    wandb: WandbConfig = WandbConfig()

    # Training-time metrics and evaluations
    metrics: TrainingMetricsConfig = TrainingMetricsConfig()
    evaluation: TrainingEvaluationConfig = TrainingEvaluationConfig()

    # Paths
    checkpoint_dir: Path | None = None  # Output directory for checkpoints

    # Training data
    val_split: float = 0.1
    seed: int = 42


class TrainingResult(BaseModel):
    """Result from running training."""

    class Config:
        arbitrary_types_allowed = True

    checkpoint_path: Path | None = None
    num_train_samples: int = 0
    num_val_samples: int = 0

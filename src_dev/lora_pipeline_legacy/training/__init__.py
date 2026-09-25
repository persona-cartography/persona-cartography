"""Training module for LoRA fine-tuning.

Example:
    from src_dev.lora_pipeline_legacy.training import run_training, TrainingConfig, LoraConfig, SftConfig
    from src_dev.common.config import ModelConfig

    config = TrainingConfig(
        dataset_path=Path("scratch/data/train.jsonl"),
        user_column="question",
        assistant_column="response",
        model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        lora=LoraConfig(r=16, lora_alpha=32),
        sft=SftConfig(num_train_epochs=3),
        checkpoint_dir=Path("scratch/checkpoints"),
    )
    val_dataset, result = run_training(config)
"""

from src_dev.lora_pipeline_legacy.training.config import (
    TrainingConfig,
    TrainingResult,
    LoraConfig,
    SftConfig,
    CheckpointConfig,
    TrainingMetricsConfig,
    TrainingEvaluationConfig,
)
from src_dev.lora_pipeline_legacy.training.run import run_training
from src_dev.lora_pipeline_legacy.training.cli import main

__all__ = [
    # Config classes
    "TrainingConfig",
    "TrainingResult",
    "LoraConfig",
    "SftConfig",
    "CheckpointConfig",
    "TrainingMetricsConfig",
    "TrainingEvaluationConfig",
    # Run function
    "run_training",
    # CLI entry point
    "main",
]

# Training

LoRA fine-tuning for causal language models using SFT (Supervised Fine-Tuning).
Training is dataset-driven and requires explicit user/assistant columns.

## CLI Usage

```bash
# Basic run
uv run python -m src_dev.training \
  --dataset-path scratch/data/train.jsonl \
  --user-column question \
  --assistant-column response \
  --checkpoint-dir scratch/my_exp/checkpoints

# With custom grouping + prompt settings
uv run python -m src_dev.training \
  --dataset-path scratch/data/train.jsonl \
  --user-column user_text \
  --assistant-column assistant_text \
  --group-column conversation_id \
  --prompt-format chat \
  --chat-system-prompt "You are a helpful assistant." \
  --checkpoint-dir scratch/my_exp/checkpoints

# Plain mode with explicit template
uv run python -m src_dev.training \
  --dataset-path scratch/data/train.jsonl \
  --user-column question \
  --assistant-column completion \
  --prompt-format plain \
  --plain-prompt-template "### User:\n{user}\n\n### Assistant:\n" \
  --checkpoint-dir scratch/my_exp/checkpoints
```

## Required Flags

- `--dataset-path`
- `--user-column`
- `--assistant-column`
- `--checkpoint-dir`

## Python Usage

```python
from pathlib import Path
from src_dev.training import (
    run_training,
    TrainingConfig,
    LoraConfig,
    SftConfig,
    TrainingEvaluationConfig,
)
from src_dev.common.config import ModelConfig

config = TrainingConfig(
    dataset_path=Path("scratch/data/train.jsonl"),
    user_column="question",
    assistant_column="response",
    model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
    lora=LoraConfig(r=16, lora_alpha=32),
    sft=SftConfig(num_train_epochs=3),
    evaluation=TrainingEvaluationConfig(
        evaluations=["count_o", "coherence"],
        eval_every_n_epochs=1,
    ),
    checkpoint_dir=Path("scratch/checkpoints"),
)
val_dataset, result = run_training(config)
```

## Notes

- User text is context only; assistant text is the training target.
- Training uses completion-only loss.
- Single-turn training is supported in this interface.

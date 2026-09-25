"""Core training logic for LoRA fine-tuning."""

from __future__ import annotations

import inspect
import math
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTTrainer, SFTConfig as TrlSftConfig

from src_dev.common.config import DatasetConfig
from src_dev.datasets import load_dataset_from_config
from src_dev.persona_metrics import PersonaMetricsConfig, run_persona_metrics
from src_dev.lora_pipeline_legacy.training.config import TrainingConfig, TrainingResult
from src_dev.utils import setup_logging


def _validate_plain_prompt_template(prompt_template: str) -> None:
    missing = [token for token in ("{user}",) if token not in prompt_template]
    if missing:
        raise ValueError(
            "plain_prompt_template must include {user} placeholder. "
            f"Missing: {missing}"
        )


def _format_plain_prompt(prompt_template: str, user_text: str) -> str:
    return prompt_template.format(user=user_text)


def _build_generation_prompt_plain(prompt_template: str, user_text: str) -> str:
    return _format_plain_prompt(prompt_template, user_text).rstrip()


def _resolve_prompt_format(tokenizer, configured_format: str) -> str:
    """Resolve prompt formatting mode using tokenizer metadata when auto."""
    if configured_format in {"chat", "plain"}:
        return configured_format

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template.strip():
        return "chat"
    return "plain"


def _format_for_sft_chat(
    tokenizer,
    user_text: str,
    assistant_text: str,
    chat_system_prompt: str | None,
    plain_prompt_template: str,
    logger,
) -> tuple[str, str]:
    messages: list[dict[str, str]] = []
    if chat_system_prompt:
        messages.append({"role": "system", "content": chat_system_prompt})
    messages.append({"role": "user", "content": user_text})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": assistant_text}],
            add_generation_prompt=False,
            tokenize=False,
        )
        if full_text.startswith(prompt_text):
            return prompt_text, full_text[len(prompt_text):]
        logger.warning(
            "Prompt/completion boundary mismatch after chat template; using plain fallback."
        )
    except Exception as exc:
        logger.warning(
            "Failed applying chat template for SFT prompt/completion; using plain fallback. error=%s",
            exc,
        )

    prompt_text = _format_plain_prompt(plain_prompt_template, user_text)
    return prompt_text, assistant_text


def _build_generation_prompt_chat(
    tokenizer,
    user_text: str,
    chat_system_prompt: str | None,
    plain_prompt_template: str,
    logger,
) -> str:
    messages: list[dict[str, str]] = []
    if chat_system_prompt:
        messages.append({"role": "system", "content": chat_system_prompt})
    messages.append({"role": "user", "content": user_text})

    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:
        logger.warning(
            "Failed applying chat template for eval prompt; falling back to "
            "plain_prompt_template format. error=%s",
            exc,
        )
        return _build_generation_prompt_plain(plain_prompt_template, user_text)


def _resolve_eos_token_id(model, tokenizer) -> int | list[int] | None:
    """Resolve EOS token ids without dropping model-specific stop ids."""
    eos_ids: list[int] = []

    model_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(model_eos, int):
        eos_ids.append(model_eos)
    elif isinstance(model_eos, list):
        eos_ids.extend(int(token_id) for token_id in model_eos)

    tokenizer_eos = tokenizer.eos_token_id
    if tokenizer_eos is not None:
        eos_ids.append(int(tokenizer_eos))

    eos_ids = list(dict.fromkeys(eos_ids))
    if not eos_ids:
        return None
    if len(eos_ids) == 1:
        return eos_ids[0]
    return eos_ids


def _global_norm(tensors: list[torch.Tensor], norm_type: float = 2.0) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        param_norm = tensor.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0 else 0.0


def _grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return _global_norm(grads) if grads else 0.0


def _param_norm(model: torch.nn.Module) -> float:
    params = [p.data for p in model.parameters() if p.requires_grad]
    return _global_norm(params) if params else 0.0


def _trainable_param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a shallow copy of all trainable parameter tensors (detached, on CPU)."""
    return {
        name: param.data.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _update_norm(
    prev_snapshot: dict[str, torch.Tensor],
    model: torch.nn.Module,
    norm_type: float = 2.0,
) -> float:
    """Compute ‖θ_t − θ_{t−1}‖ (true parameter update norm)."""
    total = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        prev = prev_snapshot.get(name)
        if prev is None:
            continue
        diff = param.data.detach().cpu() - prev
        total += diff.norm(norm_type).item() ** norm_type
    return total ** (1.0 / norm_type) if total > 0 else 0.0


class TrainingMetricsCallback(TrainerCallback):
    """Logs lightweight training metrics (e.g., grad norm) at logging steps."""

    def __init__(self, metrics_config) -> None:
        self.config = metrics_config
        self._prev_snapshot: dict[str, torch.Tensor] | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.config.enabled or logs is None:
            return

        model = kwargs.get("model")
        if model is None:
            return

        metrics: dict[str, float] = {}
        logs.setdefault("train/global_step", state.global_step)

        if self.config.log_grad_norm:
            metrics["train/grad_norm"] = _grad_norm(model)

        if self.config.log_param_norm:
            metrics["train/param_norm"] = _param_norm(model)

        if self.config.log_update_norm:
            if self._prev_snapshot is not None:
                metrics["train/update_norm"] = _update_norm(self._prev_snapshot, model)
            self._prev_snapshot = _trainable_param_snapshot(model)

        if metrics:
            logs.update(metrics)


class TrainingEvaluationCallback(TrainerCallback):
    """Runs configurable evaluations on model-generated samples during training."""

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        evaluation_config,
        plain_prompt_template: str,
        prompt_format: str,
        chat_system_prompt: str | None,
        logger,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.config = evaluation_config
        self.plain_prompt_template = plain_prompt_template
        self.prompt_format = prompt_format
        self.chat_system_prompt = chat_system_prompt
        self.eos_token_id = _resolve_eos_token_id(model, tokenizer)
        self.logger = logger
        self._eval_runs = 0

    def _should_run_step(self, state) -> bool:
        if not self.config.eval_every_n_steps:
            return False
        return state.global_step > 0 and state.global_step % self.config.eval_every_n_steps == 0

    def _should_run_epoch(self, state) -> bool:
        if not self.config.eval_every_n_epochs:
            return False
        if state.epoch is None:
            return False
        return int(state.epoch) % self.config.eval_every_n_epochs == 0

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_run_step(state):
            self._run_evaluation(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_run_epoch(state):
            self._run_evaluation(state)

    def _run_evaluation(self, state) -> None:
        if not self.config.enabled or not self.config.evaluations:
            return

        self._eval_runs += 1
        self.model.eval()
        device = next(self.model.parameters()).device

        num_samples = min(self.config.num_samples, len(self.eval_dataset))
        if num_samples <= 0:
            self.logger.warning("No samples available for evaluation.")
            return

        samples = self.eval_dataset.select(range(num_samples))
        records: list[dict[str, Any]] = []

        # NOTE: Generates responses sequentially. For large num_samples or
        # max_new_tokens values, consider batched generation.
        with torch.no_grad():
            for sample in samples:
                user_text = sample.get("question", "")
                if self.prompt_format == "chat":
                    prompt = _build_generation_prompt_chat(
                        tokenizer=self.tokenizer,
                        user_text=user_text,
                        chat_system_prompt=self.chat_system_prompt,
                        plain_prompt_template=self.plain_prompt_template,
                        logger=self.logger,
                    )
                else:
                    prompt = _build_generation_prompt_plain(
                        self.plain_prompt_template, user_text
                    )
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                ).to(device)

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )

                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )

                record = dict(sample)
                if self.config.question_column:
                    record[self.config.question_column] = user_text
                record[self.config.response_column] = response
                records.append(record)

        eval_dataset = Dataset.from_list(records)
        eval_config = PersonaMetricsConfig(
            evaluations=self.config.evaluations,
            response_column=self.config.response_column,
            question_column=self.config.question_column,
            judge=self.config.judge,
            metrics_key=self.config.metrics_key,
        )

        try:
            eval_dataset, result = run_persona_metrics(eval_config, dataset=eval_dataset)
        except Exception as exc:
            self.logger.warning("Training evaluation failed: %s", exc)
            self.model.train()
            return

        # Log aggregate metrics to W&B if enabled
        try:
            import wandb

            if wandb.run is not None:
                log_data = {"train/global_step": state.global_step}
                for key, value in result.aggregates.items():
                    if isinstance(value, (int, float)):
                        log_data[f"eval/{key}"] = value
                wandb.log(log_data, commit=False)

                if (
                    self.config.log_samples
                    and self.config.log_samples_every_n_evals > 0
                    and self._eval_runs % self.config.log_samples_every_n_evals == 0
                ):
                    metric_keys = set()
                    for record in eval_dataset:
                        metrics = record.get(self.config.metrics_key, {})
                        if isinstance(metrics, dict):
                            metric_keys.update(metrics.keys())
                    metric_columns = sorted(metric_keys)

                    columns = [
                        self.config.question_column or "question",
                        self.config.response_column,
                        *metric_columns,
                    ]
                    table = wandb.Table(columns=columns)
                    for record in eval_dataset:
                        metrics = record.get(self.config.metrics_key, {})
                        row = [
                            record.get(self.config.question_column or "question", ""),
                            record.get(self.config.response_column, ""),
                        ]
                        row.extend(
                            [metrics.get(k, "") if isinstance(metrics, dict) else "" for k in metric_columns]
                        )
                        table.add_data(*row)
                    wandb.log({"samples/eval_generations": table}, commit=False)
        except Exception as exc:
            self.logger.warning("Failed to log evaluation metrics to W&B: %s", exc)

        self.logger.info(
            "Evaluation complete at step %d. Aggregates: %s",
            state.global_step,
            result.aggregates,
        )

        self.model.train()


def load_model_for_training(config: TrainingConfig):
    """Load model and tokenizer, apply LoRA adapter.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model with LoRA, tokenizer).
    """
    model_config = config.model
    lora_config = config.lora

    dtype = getattr(torch, model_config.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {model_config.dtype}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        torch_dtype=dtype,
        device_map=model_config.device_map,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name, revision=model_config.revision, use_fast=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    task_type = getattr(TaskType, lora_config.task_type, TaskType.CAUSAL_LM)
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=task_type,
    )

    set_seed(config.seed)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _build_trl_sft_config(**kwargs):
    """Construct TrlSftConfig with version-aware parameter mapping."""
    signature = inspect.signature(TrlSftConfig.__init__)
    param_names = set(signature.parameters.keys())

    # Map common parameter name changes across TRL versions
    if "evaluation_strategy" in param_names and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in param_names and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "max_seq_length" in param_names and "max_length" in kwargs:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    if "max_length" in param_names and "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")

    try:
        return TrlSftConfig(**kwargs)
    except TypeError as exc:
        available = sorted(p for p in param_names if p != "self")
        provided = sorted(k for k in kwargs.keys())
        unsupported = sorted(set(provided) - set(available))
        raise ValueError(
            "Failed to construct TrlSftConfig. This may indicate a TRL version "
            "mismatch. Please verify your installed TRL version and supported "
            f"parameters. Unsupported params: {unsupported}. Available params: {available}. "
            f"Original error: {exc}"
        ) from exc


def _grouped_split(
    dataset: Dataset,
    test_size: float,
    seed: int,
    logger,
    group_column: str = "question",
) -> tuple[Dataset, Dataset]:
    """Split dataset into train/val, keeping all rows for a group key together.

    When the dataset has multiple responses per question (e.g. from
    ``num_responses_per_prompt > 1``), a naive random split lets the same
    question appear in both train and val, inflating validation metrics.
    This function groups rows by question text and assigns entire groups.
    """
    groups = dataset[group_column]
    unique_groups = list(dict.fromkeys(groups))  # preserves order

    if len(unique_groups) == len(dataset):
        # Every row has a unique question — fast path: use standard split.
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split["train"], split["test"]

    logger.info(
        "Detected %d unique '%s' values across %d rows; using grouped split.",
        len(unique_groups),
        group_column,
        len(dataset),
    )

    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    n_val = max(1, int(len(unique_groups) * test_size))
    val_groups = set(unique_groups[:n_val])

    train_indices = []
    val_indices = []
    for idx, group in enumerate(groups):
        if group in val_groups:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return dataset.select(train_indices), dataset.select(val_indices)


def _load_local_training_dataset(dataset_path: Path) -> Dataset:
    """Load a local JSON/JSONL dataset for training."""
    return load_dataset_from_config(
        DatasetConfig(source="local", path=str(dataset_path))
    )


def _normalize_training_dataset(
    dataset: Dataset,
    *,
    user_column: str,
    assistant_column: str,
    group_column: str | None,
) -> Dataset:
    """Normalize raw input rows into single-turn training schema."""
    missing = {
        col
        for col in (user_column, assistant_column)
        if col not in dataset.column_names
    }
    if missing:
        raise ValueError(
            f"Training dataset missing required columns: {sorted(missing)}. "
            f"Available: {sorted(dataset.column_names)}"
        )
    if group_column is not None and group_column not in dataset.column_names:
        raise ValueError(
            f"group_column '{group_column}' not found in dataset. "
            f"Available: {sorted(dataset.column_names)}"
        )

    rows = dataset.to_list()
    normalized: list[dict[str, str]] = []
    invalid_user: list[int] = []
    invalid_assistant: list[int] = []

    for idx, row in enumerate(rows):
        user_raw = row.get(user_column)
        assistant_raw = row.get(assistant_column)

        user_text = "" if user_raw is None else str(user_raw)
        assistant_text = "" if assistant_raw is None else str(assistant_raw)

        if not user_text.strip():
            invalid_user.append(idx)
            continue
        if not assistant_text.strip():
            invalid_assistant.append(idx)
            continue

        group_raw = row.get(group_column) if group_column else user_text
        group_id = user_text if group_raw is None else str(group_raw)
        if not group_id.strip():
            group_id = user_text

        normalized.append(
            {
                "question": user_text,
                "assistant_target": assistant_text,
                "group_id": group_id,
            }
        )

    if invalid_user:
        preview = invalid_user[:10]
        raise ValueError(
            "Found rows with empty user text in "
            f"column '{user_column}' at indices {preview} "
            f"(total={len(invalid_user)})."
        )
    if invalid_assistant:
        preview = invalid_assistant[:10]
        raise ValueError(
            "Found rows with empty assistant text in "
            f"column '{assistant_column}' at indices {preview} "
            f"(total={len(invalid_assistant)})."
        )
    if not normalized:
        raise ValueError("Training dataset has no valid rows after normalization.")

    return Dataset.from_list(normalized)


def run_training(
    config: TrainingConfig,
) -> tuple[Dataset, TrainingResult]:
    """Run SFT training with LoRA.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (validation dataset, TrainingResult metadata).

    Example:
        config = TrainingConfig(
            dataset_path=Path("scratch/data/train.jsonl"),
            user_column="question",
            assistant_column="response",
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
            lora=LoraConfig(r=16),
            checkpoint_dir=Path("scratch/checkpoints"),
        )
        val_dataset, result = run_training(config)
    """
    logger = setup_logging()
    set_seed(config.seed)

    if config.checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be specified in TrainingConfig")
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {config.dataset_path}")
    _validate_plain_prompt_template(config.plain_prompt_template)

    raw_dataset = _load_local_training_dataset(config.dataset_path)
    dataset = _normalize_training_dataset(
        raw_dataset,
        user_column=config.user_column,
        assistant_column=config.assistant_column,
        group_column=config.group_column,
    )

    # Split into train/val, grouping by selected group ids to prevent leakage.
    train_dataset, val_dataset = _grouped_split(
        dataset,
        test_size=config.val_split,
        seed=config.seed,
        logger=logger,
        group_column="group_id",
    )

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Setup W&B
    run_id = None
    if config.wandb.enabled:
        import wandb
        wandb.login()
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
            group=config.wandb.group,
            config={
                "model": config.model.name,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "learning_rate": config.sft.learning_rate,
                "epochs": config.sft.num_train_epochs,
                "batch_size": config.sft.per_device_train_batch_size,
            },
        )
        run_id = wandb.run.id if wandb.run is not None else None
        # Define metrics for proper plotting in wandb
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("samples/*", step_metric="train/global_step")

    # Load model with LoRA
    logger.info("Loading model with LoRA adapter...")
    model, tokenizer = load_model_for_training(config)
    prompt_format = _resolve_prompt_format(tokenizer, config.prompt_format)
    logger.info("Training prompt format: %s", prompt_format)

    def _format_for_sft_with_mode(example: dict) -> dict:
        user_text = str(example.get("question", ""))
        assistant_text = str(example.get("assistant_target", ""))
        if prompt_format == "chat":
            prompt_text, completion_text = _format_for_sft_chat(
                tokenizer=tokenizer,
                user_text=user_text,
                assistant_text=assistant_text,
                chat_system_prompt=config.chat_system_prompt,
                plain_prompt_template=config.plain_prompt_template,
                logger=logger,
            )
        else:
            prompt_text = _format_plain_prompt(config.plain_prompt_template, user_text)
            completion_text = assistant_text
        return {"prompt": prompt_text, "completion": completion_text}

    # Format for SFT
    columns_to_remove = [
        col for col in ("assistant_target", "group_id")
        if col in train_dataset.column_names
    ]
    train_dataset = train_dataset.map(
        _format_for_sft_with_mode,
        remove_columns=columns_to_remove,
    )
    val_dataset = val_dataset.map(
        _format_for_sft_with_mode,
        remove_columns=columns_to_remove,
    )

    # Setup output directory
    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    sft_cfg = config.sft
    steps_per_epoch = math.ceil(
        len(train_dataset)
        / max(1, sft_cfg.per_device_train_batch_size * sft_cfg.gradient_accumulation_steps)
    )
    total_steps = max(1, steps_per_epoch * sft_cfg.num_train_epochs)
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)

    save_strategy = config.checkpoint.save_strategy
    save_steps = None
    if save_strategy == "steps":
        save_steps = config.checkpoint.save_steps
    elif save_strategy != "epoch":
        raise ValueError(
            "checkpoint.save_strategy must be 'epoch' or 'steps'. "
            f"Got: {save_strategy}"
        )

    trl_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg.num_train_epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=sft_cfg.fp16,
        bf16=sft_cfg.bf16,
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=config.checkpoint.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=f"{run_id}-training" if run_id else None,
        seed=config.seed,
        max_seq_length=sft_cfg.max_seq_length,
    )
    trl_kwargs["completion_only_loss"] = True
    if save_steps is not None:
        trl_kwargs["save_steps"] = save_steps

    training_args = _build_trl_sft_config(**trl_kwargs)

    callbacks: list[TrainerCallback] = []
    if config.metrics.enabled:
        callbacks.append(TrainingMetricsCallback(config.metrics))

    if config.evaluation.enabled and config.evaluation.evaluations:
        callbacks.append(
            TrainingEvaluationCallback(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=val_dataset,
                evaluation_config=config.evaluation,
                plain_prompt_template=config.plain_prompt_template,
                prompt_format=prompt_format,
                chat_system_prompt=config.chat_system_prompt,
                logger=logger,
            )
        )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Saved final model to %s", final_path)

    # Conditionally log LoRA adapter as W&B artifact
    if config.wandb.enabled and config.wandb.upload_checkpoints_to_wandb:
        import wandb
        logger.info(
            "Training complete. upload_checkpoints_to_wandb is enabled. "
            "Waiting for human confirmation to upload checkpoint artifacts to W&B..."
        )
        logger.info(
            "To upload checkpoint to W&B, please confirm this run was successful "
            "and then the artifact will be logged."
        )

        # Prompt for human confirmation
        try:
            confirmation = input(
                "\nWas this training run successful? Upload checkpoint to W&B? (yes/no): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirmation = "no"
            print()

        if confirmation in ("yes", "y"):
            artifact = wandb.Artifact(
                name="lora-adapter",
                type="model",
                description=f"LoRA adapter (r={config.lora.r}, alpha={config.lora.lora_alpha})",
                metadata={
                    "base_model": config.model.name,
                    "lora_r": config.lora.r,
                    "lora_alpha": config.lora.lora_alpha,
                    "lora_dropout": config.lora.lora_dropout,
                    "target_modules": config.lora.target_modules,
                },
            )
            artifact.add_dir(str(final_path))
            wandb.log_artifact(artifact)
            logger.info("Logged LoRA adapter as W&B artifact")
        else:
            logger.info("Checkpoint upload to W&B skipped by user.")
        wandb.finish()
    elif config.wandb.enabled:
        import wandb
        wandb.finish()

    result = TrainingResult(
        checkpoint_path=final_path,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )

    return val_dataset, result

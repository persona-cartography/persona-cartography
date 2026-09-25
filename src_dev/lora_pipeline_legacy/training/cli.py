#!/usr/bin/env python3
"""CLI entry point for the training stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from src_dev.common.config import ModelConfig, WandbConfig
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.lora_pipeline_legacy.training.config import (
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
)
from src_dev.lora_pipeline_legacy.training.run import run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA fine-tuning on a local dataset with explicit user/assistant columns.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or HuggingFace path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Input / Output
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to local training dataset JSON/JSONL.",
    )
    parser.add_argument(
        "--user-column",
        type=str,
        required=True,
        help="Column containing user text (prompt context, not trained as target).",
    )
    parser.add_argument(
        "--assistant-column",
        type=str,
        required=True,
        help="Column containing assistant target text (trained completion).",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default=None,
        help=(
            "Optional grouping column used for train/val split to avoid leakage. "
            "If omitted, user text is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Prompt formatting mode (default: auto).",
    )
    parser.add_argument(
        "--chat-system-prompt",
        type=str,
        default=None,
        help="Optional system prompt used when prompt format resolves to chat.",
    )
    parser.add_argument(
        "--plain-prompt-template",
        type=str,
        default="### User:\n{user}\n\n### Assistant:\n",
        help=(
            "Template for plain prompt mode; must contain {user}. "
            "Default: '### User:\\n{user}\\n\\n### Assistant:\\n'"
        ),
    )

    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    # Validation
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )

    # Evaluations
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable training-time evaluations",
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        default=None,
        help="Evaluations to run during training (e.g., count_o coherence)",
    )
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=None,
        help="Run evaluations every N training steps (default: disabled)",
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=1,
        help="Run evaluations every N epochs (default: 1)",
    )
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=20,
        help="Number of samples to evaluate (default: 20)",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for eval generation (default: 128)",
    )
    parser.add_argument(
        "--eval-max-prompt-length",
        type=int,
        default=512,
        help="Max prompt length for eval generation (default: 512)",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for eval generation (default: 0.7)",
    )
    parser.add_argument(
        "--eval-top-p",
        type=float,
        default=0.9,
        help="Top-p for eval generation (default: 0.9)",
    )
    parser.add_argument(
        "--eval-do-sample",
        action="store_true",
        help="Enable sampling for eval generation (default: on)",
    )
    parser.add_argument(
        "--eval-no-sample",
        action="store_true",
        help="Disable sampling for eval generation",
    )
    parser.add_argument(
        "--eval-response-column",
        type=str,
        default="response",
        help="Response column name for evaluation outputs (default: response)",
    )
    parser.add_argument(
        "--eval-question-column",
        type=str,
        default="question",
        help="Question column name for evaluation inputs (default: question)",
    )
    parser.add_argument(
        "--eval-metrics-key",
        type=str,
        default="persona_metrics",
        help="Key to store evaluation metrics in records (default: persona_metrics)",
    )
    parser.add_argument(
        "--eval-log-samples",
        action="store_true",
        help="Log evaluation samples table to W&B",
    )
    parser.add_argument(
        "--eval-log-samples-every-n",
        type=int,
        default=1,
        help="Log sample table every N eval runs (default: 1)",
    )

    # Judge (LLM) config for evaluations
    parser.add_argument(
        "--eval-judge-provider",
        type=str,
        default="openai",
        help="Judge provider for LLM-based evals (default: openai)",
    )
    parser.add_argument(
        "--eval-judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model for LLM-based evals (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--eval-judge-api-key-env",
        type=str,
        default=None,
        help="Env var name for judge API key (default: provider default)",
    )
    parser.add_argument(
        "--eval-judge-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for judge model (default: 1024)",
    )
    parser.add_argument(
        "--eval-judge-temperature",
        type=float,
        default=0.0,
        help="Judge model temperature (default: 0.0)",
    )
    parser.add_argument(
        "--eval-judge-max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge requests (default: 10)",
    )
    parser.add_argument(
        "--eval-judge-timeout",
        type=int,
        default=60,
        help="Judge request timeout in seconds (default: 60)",
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="persona-pipeline-v1",
        help="Weights & Biases project name (default: persona-pipeline-v1)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--upload-checkpoints-to-wandb",
        action="store_true",
        help=(
            "Upload checkpoint artifacts to W&B after confirming successful run "
            "(default: False)"
        ),
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    eval_do_sample = True
    if args.eval_no_sample:
        eval_do_sample = False
    elif args.eval_do_sample:
        eval_do_sample = True

    evaluations = args.evaluations if args.evaluations is not None else None

    eval_config = TrainingEvaluationConfig(
        enabled=not args.no_eval,
        evaluations=evaluations or TrainingEvaluationConfig().evaluations,
        judge=JudgeLLMConfig(
            provider=args.eval_judge_provider,
            model=args.eval_judge_model,
            api_key_env=args.eval_judge_api_key_env,
            max_tokens=args.eval_judge_max_tokens,
            temperature=args.eval_judge_temperature,
            max_concurrent=args.eval_judge_max_concurrent,
            timeout=args.eval_judge_timeout,
        ),
        num_samples=args.eval_num_samples,
        max_new_tokens=args.eval_max_new_tokens,
        max_prompt_length=args.eval_max_prompt_length,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        do_sample=eval_do_sample,
        eval_every_n_steps=args.eval_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        response_column=args.eval_response_column,
        question_column=args.eval_question_column,
        metrics_key=args.eval_metrics_key,
        log_samples=args.eval_log_samples,
        log_samples_every_n_evals=args.eval_log_samples_every_n,
    )

    config = TrainingConfig(
        dataset_path=Path(args.dataset_path),
        user_column=args.user_column,
        assistant_column=args.assistant_column,
        group_column=args.group_column,
        model=ModelConfig(name=args.model),
        lora=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha),
        sft=SftConfig(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
        ),
        plain_prompt_template=args.plain_prompt_template,
        prompt_format=args.prompt_format,
        chat_system_prompt=args.chat_system_prompt,
        wandb=WandbConfig(
            enabled=not args.no_wandb,
            project=args.wandb_project,
            upload_checkpoints_to_wandb=args.upload_checkpoints_to_wandb,
        ),
        checkpoint_dir=Path(args.checkpoint_dir),
        val_split=args.val_split,
        evaluation=eval_config,
    )
    run_training(config)


if __name__ == "__main__":
    main()

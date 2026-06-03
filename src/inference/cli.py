#!/usr/bin/env python3
"""CLI entry point for the inference stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import DatasetConfig, GenerationConfig
from src.inference.config import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIBatchConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    RetryConfig,
)
from src.inference.run import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM inference on a dataset.",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or HuggingFace path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai", "openrouter", "anthropic"],
        default="local",
        help="Inference provider: local, openai, openrouter, or anthropic",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., vicgalle/alpaca-gpt4)",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["huggingface", "local"],
        default="huggingface",
        help="Dataset source type (default: huggingface)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local dataset path for --dataset-source local.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Generation settings
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100000,
        help="Maximum new tokens to generate (default: 100000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1,
        help="Number of responses per prompt (default: 1)",
    )
    parser.add_argument(
        "--local-prompt-format",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Local prompt formatting mode (default: auto).",
    )
    parser.add_argument(
        "--local-chat-system-prompt",
        type=str,
        default=None,
        help="Optional system prompt used when local prompt format resolves to chat.",
    )

    # Async + retry settings (remote providers)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60, use 0 to disable)",
    )
    parser.add_argument(
        "--retry-max-retries",
        type=int,
        default=3,
        help="Max retry attempts for API calls (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=2.0,
        help="Exponential backoff multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first API error instead of continuing",
    )

    # Output
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save output JSONL file",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Canonical run directory (e.g., scratch/runs/<run_id>).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt metadata pointer/content for canonical runs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing output rows; start from beginning.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite output_path before running instead of appending/resuming.",
    )
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=3,
        help="Max canonical attempts per response row before marking terminal; <=0 disables cap (default: 3).",
    )

    # OpenAI provider settings
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI API (default: official OpenAI endpoint)",
    )
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for OpenAI API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default=None,
        help="OpenAI reasoning effort (optional).",
    )
    parser.add_argument(
        "--openai-verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="OpenAI verbosity (optional).",
    )
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help="Use OpenAI Batch API (Responses endpoint).",
    )
    parser.add_argument(
        "--openai-batch-completion-window",
        type=str,
        default="24h",
        help="Batch completion window (default: 24h).",
    )
    parser.add_argument(
        "--openai-batch-poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--openai-batch-timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for batch completion.",
    )
    parser.add_argument(
        "--openai-batch-include-sampling",
        action="store_true",
        help="Include temperature/top_p in batch requests (default: omitted).",
    )
    parser.add_argument(
        "--openai-batch-run-dir",
        type=str,
        default=None,
        help="Run directory name under scratch/ for batch artifacts.",
    )
    parser.add_argument(
        "--openai-batch-resume",
        action="store_true",
        help="Resume an existing OpenAI batch using run-dir metadata.",
    )

    # OpenRouter provider settings
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--openrouter-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for OpenRouter API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-app-url",
        type=str,
        default=None,
        help="Optional app URL for OpenRouter attribution",
    )
    parser.add_argument(
        "--openrouter-app-name",
        type=str,
        default=None,
        help="Optional app name for OpenRouter attribution",
    )

    # Anthropic provider settings
    parser.add_argument(
        "--anthropic-api-key-env",
        type=str,
        default="ANTHROPIC_API_KEY",
        help="Environment variable name for Anthropic API key (default: ANTHROPIC_API_KEY)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_source == "local" and not args.dataset_path:
        raise ValueError("--dataset-path is required when --dataset-source local.")

    timeout = None if args.timeout <= 0 else args.timeout
    max_attempts = args.max_attempts_per_sample if args.max_attempts_per_sample > 0 else None

    config = InferenceConfig(
        model=args.model,
        provider=args.provider,
        dataset=DatasetConfig(
            source=args.dataset_source,
            name=args.dataset_name,
            path=args.dataset_path,
            max_samples=args.max_samples,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            num_responses_per_prompt=args.num_responses,
        ),
        max_concurrent=args.max_concurrent,
        timeout=timeout,
        retry=RetryConfig(
            max_retries=args.retry_max_retries,
            backoff_factor=args.retry_backoff_factor,
        ),
        continue_on_error=not args.fail_fast,
        openai=OpenAIProviderConfig(
            base_url=args.openai_base_url,
            api_key_env=args.openai_api_key_env,
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
            batch=OpenAIBatchConfig(
                enabled=args.openai_batch,
                completion_window=args.openai_batch_completion_window,
                poll_interval_seconds=args.openai_batch_poll_interval,
                timeout_seconds=args.openai_batch_timeout,
                include_sampling=args.openai_batch_include_sampling,
                run_dir=args.openai_batch_run_dir,
                resume=args.openai_batch_resume,
            ),
        ),
        openrouter=OpenRouterProviderConfig(
            base_url=args.openrouter_base_url,
            api_key_env=args.openrouter_api_key_env,
            app_url=args.openrouter_app_url,
            app_name=args.openrouter_app_name,
        ),
        anthropic=AnthropicProviderConfig(
            api_key_env=args.anthropic_api_key_env,
        ),
        local=LocalProviderConfig(
            prompt_format=args.local_prompt_format,
            chat_system_prompt=args.local_chat_system_prompt,
        ),
        output_path=Path(args.output_path) if args.output_path else None,
        run_dir=Path(args.run_dir),
        system_prompt=args.system_prompt,
        max_attempts_per_sample=max_attempts,
        resume=not args.no_resume,
        overwrite_output=args.overwrite_output,
    )

    run_inference(config)


if __name__ == "__main__":
    main()

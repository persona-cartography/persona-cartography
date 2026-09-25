"""Tests for the training CLI interface."""

from __future__ import annotations

import pytest

from src_dev.lora_pipeline_legacy.training.cli import parse_args


def test_cli_requires_dataset_user_assistant_columns() -> None:
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(
        [
            "--dataset-path",
            "scratch/data/train.jsonl",
            "--user-column",
            "question",
            "--assistant-column",
            "response",
            "--checkpoint-dir",
            "scratch/checkpoints",
        ]
    )
    assert args.dataset_path == "scratch/data/train.jsonl"
    assert args.user_column == "question"
    assert args.assistant_column == "response"


def test_cli_rejects_removed_legacy_args() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset-path",
                "scratch/data/train.jsonl",
                "--user-column",
                "question",
                "--assistant-column",
                "response",
                "--checkpoint-dir",
                "scratch/checkpoints",
                "--run-dir",
                "scratch/runs/old",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset-path",
                "scratch/data/train.jsonl",
                "--user-column",
                "question",
                "--assistant-column",
                "response",
                "--checkpoint-dir",
                "scratch/checkpoints",
                "--training-variant",
                "nano",
            ]
        )

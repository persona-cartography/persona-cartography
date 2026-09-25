"""Tests for dataset-driven training run utilities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from src_dev.common.config import ModelConfig, WandbConfig
from src_dev.lora_pipeline_legacy.training.config import (
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    TrainingMetricsConfig,
)
from src_dev.lora_pipeline_legacy.training.run import (
    _build_generation_prompt_chat,
    _build_generation_prompt_plain,
    _format_for_sft_chat,
    _grouped_split,
    _normalize_training_dataset,
    run_training,
)


class _DummyLogger:
    def warning(self, *_args, **_kwargs) -> None:
        return

    def info(self, *_args, **_kwargs) -> None:
        return


class _ChatTokenizer:
    chat_template = "dummy-chat-template"

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        del tokenize
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            return f"{text}|assistant:"
        return text


def test_training_requires_user_and_assistant_columns() -> None:
    dataset = Dataset.from_list([{"prompt": "u", "completion": "a"}])

    with pytest.raises(ValueError, match="missing required columns"):
        _normalize_training_dataset(
            dataset,
            user_column="question",
            assistant_column="response",
            group_column=None,
        )


def test_training_uses_completion_only_target(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "train.jsonl"
    records = [
        {"user_text": "Q1", "assistant_text": "A1"},
        {"user_text": "Q2", "assistant_text": "A2"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    captured_trl_kwargs: dict[str, object] = {}
    captured_train_dataset: dict[str, Dataset] = {}

    class _DummyTokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 1

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    class _DummyModel:
        generation_config = SimpleNamespace(eos_token_id=1)

    class _DummyTrainer:
        def __init__(
            self,
            *,
            model,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            callbacks,
        ) -> None:
            del model, args, eval_dataset, processing_class, callbacks
            captured_train_dataset["dataset"] = train_dataset

        def train(self) -> None:
            return

        def save_model(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _fake_build_trl_sft_config(**kwargs):
        captured_trl_kwargs.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "src_dev.lora_pipeline_legacy.training.run.load_model_for_training",
        lambda _config: (_DummyModel(), _DummyTokenizer()),
    )
    monkeypatch.setattr("src_dev.lora_pipeline_legacy.training.run.SFTTrainer", _DummyTrainer)
    monkeypatch.setattr("src_dev.lora_pipeline_legacy.training.run._build_trl_sft_config", _fake_build_trl_sft_config)

    config = TrainingConfig(
        dataset_path=dataset_path,
        user_column="user_text",
        assistant_column="assistant_text",
        model=ModelConfig(name="dummy"),
        sft=SftConfig(num_train_epochs=1, per_device_train_batch_size=1),
        wandb=WandbConfig(enabled=False),
        metrics=TrainingMetricsConfig(enabled=False),
        evaluation=TrainingEvaluationConfig(enabled=False, evaluations=[]),
        checkpoint_dir=tmp_path / "checkpoints",
        val_split=0.5,
        seed=123,
    )

    run_training(config)

    assert captured_trl_kwargs["completion_only_loss"] is True
    train_dataset = captured_train_dataset["dataset"]
    assert "prompt" in train_dataset.column_names
    assert "completion" in train_dataset.column_names
    assert "assistant_target" not in train_dataset.column_names
    assert "group_id" not in train_dataset.column_names

    completion = train_dataset[0]["completion"]
    prompt = train_dataset[0]["prompt"]
    assert completion in {"A1", "A2"}
    assert prompt.startswith("### User:\n")
    assert completion not in prompt


def test_grouped_split_uses_group_column_when_provided() -> None:
    raw = Dataset.from_list(
        [
            {"u": "q1", "a": "a1", "gid": "g1"},
            {"u": "q1_alt", "a": "a1_alt", "gid": "g1"},
            {"u": "q2", "a": "a2", "gid": "g2"},
            {"u": "q2_alt", "a": "a2_alt", "gid": "g2"},
        ]
    )
    normalized = _normalize_training_dataset(
        raw,
        user_column="u",
        assistant_column="a",
        group_column="gid",
    )

    train_dataset, val_dataset = _grouped_split(
        normalized,
        test_size=0.5,
        seed=7,
        logger=_DummyLogger(),
        group_column="group_id",
    )

    train_groups = set(train_dataset["group_id"])
    val_groups = set(val_dataset["group_id"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == {"g1", "g2"}


def test_plain_and_chat_prompt_modes_build_expected_prompts() -> None:
    plain_template = "User: {user}\nAssistant:"
    plain_prompt = _build_generation_prompt_plain(plain_template, "Hello")
    assert plain_prompt == "User: Hello\nAssistant:"

    tokenizer = _ChatTokenizer()
    logger = _DummyLogger()
    chat_prompt = _build_generation_prompt_chat(
        tokenizer=tokenizer,
        user_text="Hello",
        chat_system_prompt="Be concise.",
        plain_prompt_template=plain_template,
        logger=logger,
    )
    assert chat_prompt == "system:Be concise.|user:Hello|assistant:"

    prompt, completion = _format_for_sft_chat(
        tokenizer=tokenizer,
        user_text="Hello",
        assistant_text="Hi there",
        chat_system_prompt="Be concise.",
        plain_prompt_template=plain_template,
        logger=logger,
    )
    assert prompt == "system:Be concise.|user:Hello|assistant:"
    assert completion == "Hi there"

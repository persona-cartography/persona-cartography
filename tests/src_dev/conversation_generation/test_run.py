"""Tests for multi-turn conversation generation."""

from __future__ import annotations

from src_dev.common.config import DatasetConfig
from src_dev.conversation_generation import (
    ConversationGenerationConfig,
    ResponderConfig,
    run_conversation_generation,
)
from src_dev.lora_pipeline_legacy.editing import EditingConfig
from src_dev.inference import InferenceConfig
from src_dev.conversation_generation.run import _build_responder_messages, _format_turn_label
from src_dev.datasets import (
    ingest_source_dataset,
    load_samples,
    write_edit_overlay,
    write_inference_result,
)


class _QueuedProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.batch_sizes: list[int] = []

    async def generate_batch_with_details_async(self, prompts, **kwargs):
        del kwargs
        self.batch_sizes.append(len(prompts))
        responses = [self._outputs.pop(0) for _ in prompts]
        usages = [{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2} for _ in prompts]
        return responses, usages, 0


def test_run_conversation_generation_completes_two_turns(monkeypatch, tmp_path):
    providers = {
        "assistant-model": _QueuedProvider(["Base answer 1", "Base answer 2"]),
        "editor-model": _QueuedProvider(["Edited answer 1", "Edited answer 2"]),
        "responder-model": _QueuedProvider(["Follow-up question"]),
    }

    def _fake_get_provider(_provider_name, config):
        return providers[config.model]

    monkeypatch.setattr("src_dev.conversation_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed.jsonl"
    dataset_path.write_text('{"question":"Opening question"}\n', encoding="utf-8")

    config = ConversationGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run",
        num_assistant_turns=2,
        assistant_inference=InferenceConfig(model="assistant-model", provider="local"),
        editing=EditingConfig(
            provider="openai",
            model="editor-model",
            prompt_template="default_persona_shatter",
            quality={"enabled": False},
        ),
        responder=ResponderConfig(provider="openai", model="responder-model"),
        editing_variant="multiturn",
    )

    dataset, result = run_conversation_generation(config)

    assert result.num_completed == 1
    assert result.num_assistant_turns_completed == 2
    assert dataset[0]["assistant_turn_count"] == 2
    assert dataset[0]["messages"][-1]["content"] == "Edited answer 2"
    assert dataset[0]["messages"][1]["content"] == "Edited answer 1"


def test_run_conversation_generation_batches_across_chats(monkeypatch, tmp_path):
    providers = {
        "assistant-model": _QueuedProvider(["Base answer 1a", "Base answer 1b"]),
        "editor-model": _QueuedProvider(["Edited answer 1a", "Edited answer 1b"]),
        "responder-model": _QueuedProvider([]),
    }

    def _fake_get_provider(_provider_name, config):
        return providers[config.model]

    monkeypatch.setattr("src_dev.conversation_generation.run.get_provider", _fake_get_provider)

    dataset_path = tmp_path / "seed_two.jsonl"
    dataset_path.write_text(
        '{"question":"Opening question 1"}\n{"question":"Opening question 2"}\n',
        encoding="utf-8",
    )

    config = ConversationGenerationConfig(
        dataset=DatasetConfig(source="local", path=str(dataset_path)),
        run_dir=tmp_path / "run_two",
        num_assistant_turns=1,
        assistant_inference=InferenceConfig(model="assistant-model", provider="local"),
        editing=EditingConfig(
            provider="openai",
            model="editor-model",
            prompt_template="default_persona_shatter",
            max_concurrent=8,
            quality={"enabled": False},
        ),
        responder=ResponderConfig(provider="openai", model="responder-model"),
        editing_variant="multiturn",
    )

    dataset, result = run_conversation_generation(config)

    assert result.num_completed == 2
    assert len(dataset) == 2
    assert providers["assistant-model"].batch_sizes == [2]
    assert providers["editor-model"].batch_sizes == [2]


def test_responder_prompt_explicitly_requests_next_user_turn(tmp_path):
    run_dir = tmp_path / "run_prompt"
    sample = ingest_source_dataset(
        [{"question": "Give three tips for staying healthy."}],
        source_info={"source": "test"},
        system_prompt=None,
        run_dir=run_dir,
        overwrite=True,
    )[0]
    write_inference_result(
        run_dir,
        sample.sample_id,
        {
            "status": "success",
            "model": "base",
            "provider": "local",
            "assistant_message_id": "msg_assistant_1",
            "assistant_completion": "Drink water, sleep enough, and move regularly.",
            "assistant_full": "Drink water, sleep enough, and move regularly.",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_edit_overlay(
        run_dir,
        sample.sample_id,
        "multiturn",
        {
            "overlay_id": "ov_1",
            "target_message_id": "msg_assistant_1",
            "target_role": "assistant",
            "original_content_hash": "h1",
            "edited_content": "Try water, decent sleep, and some movement when you can.",
            "status": "success",
            "attempt_no": 1,
        },
        materialize=False,
    )

    sample = load_samples(run_dir)[0]
    messages = _build_responder_messages(sample, "multiturn", "natural_partner")

    assert messages[0]["role"] == "system"
    assert "next USER turn" in messages[0]["content"]
    assert "Do not answer the user's question as an assistant." in messages[0]["content"]
    assert messages[-1]["role"] == "assistant"


def test_format_turn_label_handles_single_and_mixed_turn_batches():
    assert _format_turn_label([0], total_turns=3) == "turn 1/3"
    assert _format_turn_label([0, 1, 1], total_turns=3) == "turns 1-2/3"

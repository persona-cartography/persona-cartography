"""Tests for multi-turn canonical editing."""

from __future__ import annotations

from datasets import Dataset

from src_dev.datasets import ingest_source_dataset, write_inference_result, write_message_append
from src_dev.lora_pipeline_legacy.editing import EditingConfig, run_editing


class _StubEditorProvider:
    async def generate_batch_with_details_async(self, prompts, **kwargs):
        del kwargs
        assert len(prompts) == 1
        prompt = prompts[0]
        assert "Second question" in prompt
        assert "Second answer base" in prompt
        return ["Second answer edited"], [{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}], 0


def test_canonical_editing_targets_latest_assistant_message(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_editing_multiturn"
    sample = ingest_source_dataset(
        [{"question": "First question"}],
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
            "assistant_completion": "First answer base",
            "assistant_full": "First answer base",
            "attempt_no": 1,
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        sample.sample_id,
        {
            "message_id": "msg_user_2",
            "role": "user",
            "content": "Second question",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )
    write_message_append(
        run_dir,
        sample.sample_id,
        {
            "message_id": "msg_assistant_2",
            "role": "assistant",
            "content": "Second answer base",
            "message_metadata": {"turn_index": 1},
        },
        materialize=False,
    )

    monkeypatch.setattr("src_dev.lora_pipeline_legacy.editing.run.get_provider", lambda _provider, _config: _StubEditorProvider())

    dataset, _ = run_editing(
        EditingConfig(
            provider="openai",
            model="editor-model",
            prompt_template="default_persona_shatter",
            run_dir=run_dir,
            variant_name="multiturn",
            quality={"enabled": False},
        )
    )

    assert isinstance(dataset, Dataset)
    assert dataset[0]["target_message_id"] == "msg_assistant_2"
    assert dataset[0]["edited_response"] == "Second answer edited"
    assert dataset[0]["messages_edited"][-1]["content"] == "Second answer edited"

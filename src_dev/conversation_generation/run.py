"""Multi-turn conversation dataset generation."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from src_dev.datasets import (
    export_dataset,
    ingest_source_dataset,
    init_run,
    load_samples,
    materialize_canonical_samples,
    record_stage_event,
    register_stage_fingerprint,
    render_messages,
    write_edit_overlay,
    write_inference_result,
    write_message_append,
)
from src_dev.datasets.loaders import load_dataset_from_config
from src_dev.datasets.schema import StageEventRecord
from src_dev.lora_pipeline_legacy.editing import EditingConfig
from src_dev.lora_pipeline_legacy.editing.prompts import EditPromptContext, get_prompt
from src_dev.lora_pipeline_legacy.editing.run import build_inference_config
from src_dev.inference import InferenceConfig
from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import InferenceProvider, TokenUsage
from src_dev.utils import setup_logging
from src_dev.common.conversation_runtime import (
    chunked,
    format_turn_label,
    log_phase_batch_progress,
    message_append_id,
    now_iso,
)

from .config import (
    ConversationGenerationConfig,
    ConversationGenerationResult,
    ResponderConfig,
)


RESPONDER_TEMPLATES: dict[str, str] = {
    "natural_partner": (
        "You are writing the next USER turn in this conversation. "
        "Reply as the human user speaking to the assistant. "
        "Write only the next user message, in plain text. "
        "Do not answer the user's question as an assistant. "
        "Do not offer help as if you are the assistant. "
        "Do not describe your role, and do not produce labels like 'User:' or 'Assistant:'. "
        "React naturally to the assistant's latest message, and when appropriate ask a follow-up question, "
        "share a preference, give feedback, or provide extra context that a user would plausibly give."
        "Return *nothing* except the user message."
    ),
}


def _format_turn_label(turn_indices: list[int], total_turns: int) -> str:
    """Backward-compatible alias for existing callers/tests."""
    return format_turn_label(turn_indices, total_turns)


def _assistant_turn_count(sample) -> int:
    return sum(1 for message in sample.messages if message.role == "assistant")


def _completed_edited_turns(sample, variant_name: str) -> int:
    assistant_ids = {message.message_id for message in sample.messages if message.role == "assistant"}
    successful_targets: set[str] = set()
    for variant in sample.edit_variants:
        if variant.variant_name != variant_name:
            continue
        for overlay in variant.overlays:
            if overlay.status == "success" and overlay.target_message_id in assistant_ids:
                successful_targets.add(overlay.target_message_id)
    return len(successful_targets)


def _latest_assistant(sample):
    return next((message for message in reversed(sample.messages) if message.role == "assistant"), None)


def _has_successful_overlay(sample, variant_name: str, target_message_id: str) -> bool:
    for variant in sample.edit_variants:
        if variant.variant_name != variant_name:
            continue
        for overlay in sorted(
            variant.overlays,
            key=lambda item: (item.attempt_no, item.overlay_id),
            reverse=True,
        ):
            if overlay.target_message_id == target_message_id and overlay.status == "success":
                return True
    return False


def _build_responder_inference_config(config: ResponderConfig) -> InferenceConfig:
    return InferenceConfig(
        model=config.model,
        provider=config.provider,
        generation=config.generation,
        max_concurrent=config.max_concurrent,
        timeout=config.timeout,
        retry=config.retry,
        continue_on_error=False,
        log_failures=True,
        local=config.local,
        openai=config.openai,
        openrouter=config.openrouter,
        anthropic=config.anthropic,
    )


def _build_responder_messages(sample, editing_variant: str, prompt_template: str) -> list[dict[str, str]]:
    rendered = [{"role": message.role, "content": message.content} for message in render_messages(sample, editing_variant)]
    instruction = RESPONDER_TEMPLATES.get(prompt_template)
    if instruction:
        return [{"role": "system", "content": instruction}, *rendered]
    return rendered


async def _generate_one(
    provider: InferenceProvider,
    prompt: str | list[dict[str, str]],
) -> tuple[str, TokenUsage | None]:
    responses, usages, _ = await provider.generate_batch_with_details_async([prompt], num_responses=1)
    if not responses:
        return "", None
    usage = usages[0] if usages else None
    return responses[0], usage


def _record_invalid_state(run_dir: Path, sample_id: str, reason: str) -> None:
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=f"evt_{hashlib.sha256(f'{sample_id}:{reason}:{now_iso()}'.encode('utf-8')).hexdigest()[:24]}",
            stage="conversation_generation",
            event_type="invalid_state",
            sample_id=sample_id,
            created_at=now_iso(),
            payload={"reason": reason},
        ),
    )


def _build_assistant_action(sample, assistant_config: InferenceConfig) -> dict[str, Any] | None:
    latest_message = sample.messages[-1] if sample.messages else None
    if latest_message is None:
        return None
    if latest_message.role != "user":
        return None
    assistant_turn_index = _assistant_turn_count(sample)
    return {
        "sample_id": sample.sample_id,
        "prompt": [{"role": message.role, "content": message.content} for message in sample.messages],
        "assistant_turn_index": assistant_turn_index,
        "parent_message_id": latest_message.message_id,
        "attempt_no": sample.inference.attempt_no + 1,
        "model": assistant_config.model,
        "provider": assistant_config.provider,
    }


def _build_edit_action(
    sample,
    config: ConversationGenerationConfig,
    editing_config: EditingConfig,
) -> dict[str, Any] | None:
    latest_message = sample.messages[-1] if sample.messages else None
    if latest_message is None or latest_message.role != "assistant":
        return None
    if _has_successful_overlay(sample, config.editing_variant, latest_message.message_id):
        return None

    turn_index = _assistant_turn_count(sample) - 1
    latest_user_message = next(
        (
            message.content
            for message in reversed(sample.messages[:-1])
            if message.role == "user"
        ),
        "",
    )
    prior_attempts = 0
    for variant in sample.edit_variants:
        if variant.variant_name != config.editing_variant:
            continue
        prior_attempts = max(
            (
                overlay.attempt_no
                for overlay in variant.overlays
                if overlay.target_message_id == latest_message.message_id
            ),
            default=0,
        )
        break
    prompt_context = EditPromptContext(
        conversation_history=[
            {"role": message.role, "content": message.content}
            for message in sample.messages[:-1]
        ],
        latest_user_message=latest_user_message,
        base_assistant_response=latest_message.content,
        turn_index=max(turn_index, 0),
        total_turns=config.num_assistant_turns,
    )
    prompt = get_prompt(editing_config.prompt_template, context=prompt_context)
    return {
        "sample_id": sample.sample_id,
        "prompt": prompt,
        "target_message_id": latest_message.message_id,
        "assistant_content": latest_message.content,
        "turn_index": max(turn_index, 0),
        "attempt_no": prior_attempts + 1,
        "editor_model": editing_config.model,
        "editor_provider": editing_config.provider,
    }


def _build_responder_action(
    sample,
    config: ConversationGenerationConfig,
    responder_config: InferenceConfig,
) -> dict[str, Any] | None:
    latest_message = sample.messages[-1] if sample.messages else None
    if latest_message is None or latest_message.role != "assistant":
        return None
    completed_turns = _completed_edited_turns(sample, config.editing_variant)
    if completed_turns >= config.num_assistant_turns:
        return None
    if not _has_successful_overlay(sample, config.editing_variant, latest_message.message_id):
        return None
    return {
        "sample_id": sample.sample_id,
        "prompt": _build_responder_messages(
            sample,
            config.editing_variant,
            config.responder.prompt_template,
        ),
        "turn_index": _assistant_turn_count(sample),
        "display_turn_index": max(_assistant_turn_count(sample) - 1, 0),
        "parent_message_id": latest_message.message_id,
        "provider": responder_config.provider,
        "model": responder_config.model,
    }


async def _generate_many(
    provider: InferenceProvider,
    prompts: list[str | list[dict[str, str]]],
) -> tuple[list[str], list[TokenUsage | None], int]:
    if not prompts:
        return [], [], 0
    return await provider.generate_batch_with_details_async(prompts, num_responses=1)


def _run_assistant_phase(
    logger: logging.Logger,
    run_dir: Path,
    actions: list[dict[str, Any]],
    assistant_provider: InferenceProvider,
    assistant_config: InferenceConfig,
    total_turns: int,
) -> tuple[int, set[str]]:
    progressed = 0
    failed_samples: set[str] = set()
    batch_size = max(1, assistant_config.generation.batch_size)
    batches = chunked(actions, batch_size)
    total_actions = len(actions)
    for batch_index, batch in enumerate(batches, start=1):
        log_phase_batch_progress(
            logger,
            stage_name="assistant",
            batch_index=batch_index,
            num_batches=len(batches),
            items_processed=min(batch_index * batch_size, total_actions),
            items_total=total_actions,
            turn_label=format_turn_label(
                [action["assistant_turn_index"] for action in batch],
                total_turns,
            ),
        )
        prompts = [action["prompt"] for action in batch]
        responses, usages, _ = asyncio.run(_generate_many(assistant_provider, prompts))
        usage_by_slot = list(usages) if len(usages) == len(responses) else [None] * len(responses)
        for action, response_text, usage in zip(batch, responses, usage_by_slot):
            status = "success" if isinstance(response_text, str) and response_text.strip() else "failed"
            if status == "failed":
                failed_samples.add(action["sample_id"])
            write_inference_result(
                run_dir,
                action["sample_id"],
                {
                    "status": status,
                    "model": action["model"],
                    "provider": action["provider"],
                    "assistant_message_id": message_append_id(
                        action["sample_id"],
                        "assistant",
                        action["assistant_turn_index"],
                    ),
                    "assistant_completion": response_text,
                    "assistant_full": response_text,
                    "assistant_message_metadata": {
                        "turn_index": action["assistant_turn_index"],
                        "source_stage": "assistant_base",
                        "provider": action["provider"],
                        "model": action["model"],
                        "token_usage": usage or {},
                        "parent_message_id": action["parent_message_id"],
                    },
                    "attempt_no": action["attempt_no"],
                    "token_usage": usage or {},
                    "started_at": now_iso(),
                    "completed_at": now_iso(),
                    "error": None if status == "success" else "empty_response",
                },
                materialize=False,
            )
            progressed += 1
    return progressed, failed_samples


def _run_edit_phase(
    logger: logging.Logger,
    run_dir: Path,
    actions: list[dict[str, Any]],
    editor_provider: InferenceProvider,
    editing_config: EditingConfig,
    total_turns: int,
) -> tuple[int, set[str]]:
    progressed = 0
    failed_samples: set[str] = set()
    batch_size = max(1, editing_config.max_concurrent)
    batches = chunked(actions, batch_size)
    total_actions = len(actions)
    for batch_index, batch in enumerate(batches, start=1):
        log_phase_batch_progress(
            logger,
            stage_name="edit",
            batch_index=batch_index,
            num_batches=len(batches),
            items_processed=min(batch_index * batch_size, total_actions),
            items_total=total_actions,
            turn_label=format_turn_label(
                [action["turn_index"] for action in batch],
                total_turns,
            ),
        )
        prompts = [action["prompt"] for action in batch]
        responses, usages, _ = asyncio.run(_generate_many(editor_provider, prompts))
        usage_by_slot = list(usages) if len(usages) == len(responses) else [None] * len(responses)
        for action, edited_text, usage in zip(batch, responses, usage_by_slot):
            status = "success" if isinstance(edited_text, str) and edited_text.strip() else "failed"
            if status == "failed":
                failed_samples.add(action["sample_id"])
            overlay_id = hashlib.sha256(
                f"{action['sample_id']}:{action['target_message_id']}:{edited_text}".encode("utf-8")
            ).hexdigest()[:24]
            write_edit_overlay(
                run_dir,
                sample_id=action["sample_id"],
                variant_name=editing_config.variant_name or "editing",
                overlay_payload={
                    "overlay_id": overlay_id,
                    "target_message_id": action["target_message_id"],
                    "target_role": "assistant",
                    "original_content_hash": hashlib.sha256(
                        action["assistant_content"].encode("utf-8")
                    ).hexdigest(),
                    "edited_content": edited_text,
                    "status": status,
                    "attempt_no": action["attempt_no"],
                    "editor_model": action["editor_model"],
                    "editor_provider": action["editor_provider"],
                    "edit_prompt_hash": hashlib.sha256(action["prompt"].encode("utf-8")).hexdigest(),
                    "token_usage": usage or {},
                    "judge_metadata": None,
                    "timestamps": {"created_at": now_iso()},
                    "error": None if status == "success" else "empty_edited_response",
                },
                materialize=False,
            )
            progressed += 1
    return progressed, failed_samples


def _run_responder_phase(
    logger: logging.Logger,
    run_dir: Path,
    actions: list[dict[str, Any]],
    responder_provider: InferenceProvider,
    responder_config: InferenceConfig,
    total_turns: int,
) -> tuple[int, set[str]]:
    progressed = 0
    failed_samples: set[str] = set()
    batch_size = max(1, responder_config.generation.batch_size)
    batches = chunked(actions, batch_size)
    total_actions = len(actions)
    for batch_index, batch in enumerate(batches, start=1):
        log_phase_batch_progress(
            logger,
            stage_name="responder",
            batch_index=batch_index,
            num_batches=len(batches),
            items_processed=min(batch_index * batch_size, total_actions),
            items_total=total_actions,
            turn_label=format_turn_label(
                [action["display_turn_index"] for action in batch],
                total_turns,
            ),
        )
        prompts = [action["prompt"] for action in batch]
        responses, usages, _ = asyncio.run(_generate_many(responder_provider, prompts))
        usage_by_slot = list(usages) if len(usages) == len(responses) else [None] * len(responses)
        for action, user_text, usage in zip(batch, responses, usage_by_slot):
            if not isinstance(user_text, str) or not user_text.strip():
                failed_samples.add(action["sample_id"])
                _record_invalid_state(run_dir, action["sample_id"], "empty_responder_user_turn")
                continue
            write_message_append(
                run_dir,
                action["sample_id"],
                {
                    "message_id": message_append_id(
                        action["sample_id"],
                        "user",
                        action["turn_index"],
                    ),
                    "role": "user",
                    "content": user_text,
                    "editable": True,
                    "message_metadata": {
                        "turn_index": action["turn_index"],
                        "source_stage": "responder_user",
                        "provider": action["provider"],
                        "model": action["model"],
                        "token_usage": usage or {},
                        "parent_message_id": action["parent_message_id"],
                    },
                },
                materialize=False,
            )
            progressed += 1
    return progressed, failed_samples


def run_conversation_generation(
    config: ConversationGenerationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, ConversationGenerationResult]:
    """Generate edited multi-turn conversations and export them."""
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    init_run(
        run_dir,
        base_config={"conversation_generation": config.model_dump(mode="json")},
    )
    register_stage_fingerprint(
        run_dir,
        "conversation_generation",
        config.model_dump(mode="json"),
    )

    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)

    source_info: dict[str, Any] = {
        "dataset_source": config.dataset.source,
        "dataset_name": config.dataset.name,
        "dataset_path": config.dataset.path,
        "dataset_split": config.dataset.split,
        "max_samples": config.dataset.max_samples,
    }
    ingest_source_dataset(
        dataset=dataset,
        source_info=source_info,
        system_prompt=config.system_prompt,
        run_dir=run_dir,
        overwrite=config.overwrite_output,
        responses_per_input=1,
    )

    assistant_config = config.assistant_inference.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "continue_on_error": False,
        }
    )
    editing_config = config.editing.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "variant_name": config.editing_variant,
            "total_turns_hint": config.num_assistant_turns,
            "quality": config.editing.quality.model_copy(update={"enabled": False}),
        }
    )
    responder_config = _build_responder_inference_config(config.responder)
    editor_inference_config = (
        None if editing_config.provider == "code" else build_inference_config(editing_config)
    )

    assistant_provider = get_provider(assistant_config.provider, assistant_config)
    editor_provider = (
        None
        if editing_config.provider == "code"
        else get_provider(editor_inference_config.provider, editor_inference_config)
    )
    responder_provider = get_provider(responder_config.provider, responder_config)

    failed_samples: set[str] = set()
    loop_index = 0

    while True:
        loop_index += 1
        materialize_canonical_samples(run_dir)
        samples = load_samples(run_dir)
        progressed = 0
        assistant_actions: list[dict[str, Any]] = []
        edit_actions: list[dict[str, Any]] = []
        responder_actions: list[dict[str, Any]] = []
        for sample in samples:
            completed_turns = _completed_edited_turns(sample, config.editing_variant)
            if completed_turns >= config.num_assistant_turns:
                continue

            latest_message = sample.messages[-1] if sample.messages else None
            if latest_message is None:
                failed_samples.add(sample.sample_id)
                _record_invalid_state(run_dir, sample.sample_id, "empty_messages")
                continue

            if latest_message.role == "user":
                action = _build_assistant_action(sample, assistant_config)
                if action is not None:
                    assistant_actions.append(action)
                continue

            if latest_message.role != "assistant":
                failed_samples.add(sample.sample_id)
                _record_invalid_state(
                    run_dir,
                    sample.sample_id,
                    f"unsupported_terminal_role:{latest_message.role}",
                )
                continue

            if not _has_successful_overlay(sample, config.editing_variant, latest_message.message_id):
                if editing_config.provider == "code":
                    raise NotImplementedError("Code-based editing is not supported in conversation generation.")
                action = _build_edit_action(sample, config, editing_config)
                if action is not None:
                    edit_actions.append(action)
                continue

            action = _build_responder_action(sample, config, responder_config)
            if action is not None:
                responder_actions.append(action)

        logger.info(
            "Pass %d | pending assistant=%d edit=%d responder=%d",
            loop_index,
            len(assistant_actions),
            len(edit_actions),
            len(responder_actions),
        )

        assistant_progressed, assistant_failed = _run_assistant_phase(
            logger,
            run_dir,
            assistant_actions,
            assistant_provider,
            assistant_config,
            config.num_assistant_turns,
        )
        progressed += assistant_progressed
        failed_samples.update(assistant_failed)

        edit_progressed = 0
        edit_failed: set[str] = set()
        if edit_actions:
            assert editor_provider is not None
            edit_progressed, edit_failed = _run_edit_phase(
                logger,
                run_dir,
                edit_actions,
                editor_provider,
                editing_config,
                config.num_assistant_turns,
            )
        progressed += edit_progressed
        failed_samples.update(edit_failed)

        responder_progressed, responder_failed = _run_responder_phase(
            logger,
            run_dir,
            responder_actions,
            responder_provider,
            responder_config,
            config.num_assistant_turns,
        )
        progressed += responder_progressed
        failed_samples.update(responder_failed)

        materialize_canonical_samples(run_dir)
        if progressed == 0:
            break

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name=config.editing_variant,
    )
    trace_path = export_dataset(
        run_dir,
        profile="conversation_trace",
        variant_name=config.editing_variant,
    )
    result_dataset = load_dataset_from_config(
        config.dataset.model_copy(
            update={
                "source": "canonical",
                "path": str(run_dir),
                "name": f"editing:{config.editing_variant}",
            }
        )
    )

    final_samples = load_samples(run_dir)
    completed = sum(
        1
        for sample in final_samples
        if _completed_edited_turns(sample, config.editing_variant) >= config.num_assistant_turns
    )
    completed_turns = sum(
        _completed_edited_turns(sample, config.editing_variant)
        for sample in final_samples
    )
    result = ConversationGenerationResult(
        output_path=Path(export_path),
        num_conversations=len(final_samples),
        num_completed=completed,
        num_failed=len(failed_samples),
        num_assistant_turns_target=config.num_assistant_turns,
        num_assistant_turns_completed=completed_turns,
        exports={
            "conversation_training": str(export_path),
            "conversation_trace": str(trace_path),
        },
    )
    logger.info(
        "Conversation generation complete: %d/%d complete, failed=%d.",
        completed,
        len(final_samples),
        len(failed_samples),
    )
    return result_dataset, result

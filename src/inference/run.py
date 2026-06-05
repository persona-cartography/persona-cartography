"""Core inference logic for running LLM inference on datasets."""

from __future__ import annotations

import asyncio
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.datasets import (
    format_for_inference,
    ingest_source_dataset,
    init_run,
    load_dataset_from_config,
    load_samples,
    materialize_canonical_samples,
    register_stage_fingerprint,
    resume_state,
    write_inference_result,
)
from src.inference.config import InferenceConfig, InferenceResult
from src.inference.providers import get_provider
from src.inference.providers.base import PromptInput
from src.inference.providers.base import TokenUsage, accumulate_usage, empty_usage
from src.utils import count_jsonl_rows, read_jsonl, setup_logging


def _config_for_inference_fingerprint(config: InferenceConfig) -> dict[str, Any]:
    """Return the subset of inference config that should affect reproducibility."""
    config_payload = config.model_dump(mode="json")
    generation_payload = config_payload.get("generation")
    if isinstance(generation_payload, dict):
        config_payload["generation"] = {
            key: value
            for key, value in generation_payload.items()
            if key != "batch_size"
        }
    return {
        key: value
        for key, value in config_payload.items()
        if key not in {"max_concurrent"}
    }


async def run_inference_async(
    config: InferenceConfig, dataset: Dataset | None = None
) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset asynchronously.

    Uses the provider specified in config.provider:
    - "local": HuggingFace transformers (default)
    - "openai": OpenAI API
    - "openrouter": OpenRouter API (OpenAI-compatible)
    - "anthropic": Anthropic API

    Args:
        config: Inference configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.dataset.

    Returns:
        Tuple of (dataset with 'response' column, InferenceResult metadata).
    """
    logger = setup_logging()
    if config.run_dir is not None:
        return await _run_inference_canonical_async(config, dataset)

    if dataset is None:
        dataset = load_dataset_from_config(config.dataset)
    dataset = format_for_inference(dataset)

    # Batch mode for OpenAI provider
    if config.provider == "openai" and config.openai.batch.enabled:
        from src.inference.openai_batch import run_openai_batch_inference

        logger.info("Using OpenAI Batch API for inference.")
        logger.info("Model: %s", config.model)
        return await asyncio.to_thread(run_openai_batch_inference, config, dataset)

    # Get the inference provider
    logger.info("Using inference provider: %s", config.provider)
    logger.info("Model: %s", config.model)
    provider = get_provider(config.provider, config)

    generation = config.generation
    in_memory_records: list[dict[str, str | int]] = []
    total_usage = empty_usage()
    failed_count = 0
    batch_size = max(1, generation.batch_size)
    num_responses = max(1, generation.num_responses_per_prompt)
    total_expected_rows = len(dataset) * num_responses
    save_path = Path(config.output_path) if config.output_path else None

    completed_rows = 0
    start_prompt_index = 0
    start_response_offset = 0
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if config.overwrite_output:
            save_path.write_text("")
        elif config.resume and save_path.exists():
            completed_rows = count_jsonl_rows(save_path)
            if completed_rows > total_expected_rows:
                raise ValueError(
                    f"Output already has {completed_rows} rows, but only "
                    f"{total_expected_rows} are expected for this run."
                )
            start_prompt_index = completed_rows // num_responses
            start_response_offset = completed_rows % num_responses
            if completed_rows:
                logger.info(
                    "Resuming inference from row %d/%d (%d prompts fully done, response offset %d).",
                    completed_rows,
                    total_expected_rows,
                    start_prompt_index,
                    start_response_offset,
                )
        else:
            save_path.write_text("")

    if num_responses > 1 and not generation.do_sample:
        raise ValueError(
            "num_responses_per_prompt > 1 requires do_sample=True. "
            "Set generation.do_sample to True."
        )

    logger.info(
        "Starting inference on %d samples (batch_size=%d, responses_per_prompt=%d).",
        len(dataset),
        batch_size,
        num_responses,
    )

    if completed_rows >= total_expected_rows:
        logger.info("Output already complete at %s. Skipping generation.", save_path)
    else:
        write_handle = (
            save_path.open("a", encoding="utf-8") if save_path is not None else None
        )
        try:
            for start in range(start_prompt_index, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch_questions = dataset[start:end]["question"]
                (
                    batch_responses,
                    batch_usage,
                    batch_failed,
                ) = await provider.generate_batch_with_metadata_async(
                    batch_questions, num_responses=num_responses
                )
                if len(batch_responses) != len(batch_questions) * num_responses:
                    raise ValueError(
                        "Provider returned unexpected number of responses. "
                        f"Expected {len(batch_questions) * num_responses}, got {len(batch_responses)}."
                    )
                accumulate_usage(total_usage, batch_usage)
                failed_count += batch_failed

                for question_index, question in enumerate(batch_questions):
                    prompt_index = start + question_index
                    first_response_index = 0
                    if prompt_index == start_prompt_index and start_response_offset > 0:
                        first_response_index = start_response_offset

                    for response_index in range(first_response_index, num_responses):
                        output_index = question_index * num_responses + response_index
                        record = {
                            "question": question,
                            "response": batch_responses[output_index],
                            "response_index": response_index,
                        }
                        if write_handle is not None:
                            write_handle.write(json.dumps(record) + "\n")
                        else:
                            in_memory_records.append(record)

                    if prompt_index == start_prompt_index and start_response_offset > 0:
                        start_response_offset = 0
                if write_handle is not None:
                    write_handle.flush()
                logger.info("Processed %d/%d samples", end, len(dataset))
        finally:
            if write_handle is not None:
                write_handle.close()

    # Release GPU memory held by the provider (model weights, KV cache, etc.)
    del provider
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    if save_path is not None:
        result_records = read_jsonl(save_path) if save_path.exists() else []
    else:
        result_records = in_memory_records
    result_dataset = Dataset.from_list(result_records)

    if failed_count:
        logger.warning("Inference completed with %d failed responses.", failed_count)

    # Save outputs if path specified
    result = InferenceResult(
        num_samples=len(result_dataset),
        num_failed=failed_count,
        total_input_tokens=total_usage["input_tokens"],
        total_output_tokens=total_usage["output_tokens"],
        total_tokens=total_usage["total_tokens"],
    )
    if save_path is not None:
        logger.info("Saved inference output to %s", save_path)
        result.output_path = save_path

    return result_dataset, result


async def _run_inference_canonical_async(
    config: InferenceConfig, dataset: Dataset | None = None
) -> tuple[Dataset, InferenceResult]:
    """Run inference against canonical run-dir storage."""
    logger = setup_logging()
    if config.provider == "openai" and config.openai.batch.enabled:
        raise ValueError(
            "OpenAI batch mode is not supported in canonical run-dir mode."
        )

    run_dir = Path(config.run_dir)
    config_payload = config.model_dump(mode="json")
    config_for_fingerprint = _config_for_inference_fingerprint(config)
    init_run(run_dir, base_config={"inference": config_payload})
    register_stage_fingerprint(
        run_dir,
        "inference",
        config_for_fingerprint,
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
        system_prompt=config.system_prompt or config.local.chat_system_prompt,
        run_dir=run_dir,
        overwrite=config.overwrite_output,
        responses_per_input=config.generation.num_responses_per_prompt,
    )

    state = resume_state(
        run_dir,
        "inference",
        max_attempts=config.max_attempts_per_sample,
    )
    pending_ids = (
        state["pending"]
        if config.resume
        else [sample.sample_id for sample in load_samples(run_dir)]
    )
    if state["terminal"]:
        logger.warning(
            "Skipping %d response rows that reached max attempts (%s).",
            len(state["terminal"]),
            config.max_attempts_per_sample,
        )

    all_samples = {sample.sample_id: sample for sample in load_samples(run_dir)}
    total_usage = empty_usage()
    failed_count = 0
    batch_size = max(1, config.generation.batch_size)

    pending_ids = [sample_id for sample_id in pending_ids if sample_id in all_samples]
    if not pending_ids:
        logger.info("No pending inference samples in run-dir %s.", run_dir)
    else:
        provider = get_provider(config.provider, config)
        for start in range(0, len(pending_ids), batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            prompts: list[str] = []
            batch_samples = []
            batch_started = datetime.now(timezone.utc).isoformat()
            for sample_id in batch_ids:
                sample = all_samples[sample_id]
                prompts.append(_canonical_provider_input_for_sample(sample, config))
                batch_samples.append(sample)

            (
                responses,
                usages,
                batch_failed,
            ) = await provider.generate_batch_with_details_async(
                prompts, num_responses=1
            )
            usage_by_slot: list[TokenUsage | None] = list(usages)
            if len(usage_by_slot) != len(responses):
                usage_by_slot = [None] * len(responses)
            for usage in usage_by_slot:
                accumulate_usage(total_usage, usage)
            failed_count += batch_failed

            for sample, response, usage in zip(batch_samples, responses, usage_by_slot):
                last_message = sample.messages[-1] if sample.messages else None
                if last_message is None or last_message.role != "user":
                    completed_at = datetime.now(timezone.utc).isoformat()
                    write_inference_result(
                        run_dir,
                        sample.sample_id,
                        {
                            "status": "failed",
                            "model": config.model,
                            "provider": config.provider,
                            "attempt_no": sample.inference.attempt_no + 1,
                            "token_usage": usage or {},
                            "started_at": batch_started,
                            "completed_at": completed_at,
                            "error": "last_message_not_user",
                        },
                        materialize=False,
                    )
                    failed_count += 1
                    continue

                response_text = response if isinstance(response, str) else ""
                status = "success" if response_text.strip() else "failed"
                completed_at = datetime.now(timezone.utc).isoformat()
                assistant_turn_index = sum(
                    1 for msg in sample.messages if msg.role == "assistant"
                )
                assistant_message_id = _canonical_assistant_message_id(
                    sample.sample_id,
                    assistant_turn_index,
                )
                payload: dict[str, object] = {
                    "status": status,
                    "model": config.model,
                    "provider": config.provider,
                    "system_prompt_ref": sample.input.system_prompt_ref,
                    "attempt_no": sample.inference.attempt_no + 1,
                    "token_usage": usage or {},
                    "started_at": batch_started,
                    "completed_at": completed_at,
                    "error": None if status == "success" else "empty_response",
                }
                if status == "success":
                    # Only write assistant message fields on success so that
                    # materialization does not append an empty assistant message,
                    # which would brick the retry logic (last_message_not_user).
                    payload.update(
                        {
                            "assistant_message_id": assistant_message_id,
                            "assistant_prefill": sample.input.assistant_prefill,
                            "assistant_completion": response_text,
                            "assistant_full": f"{sample.input.assistant_prefill or ''}{response_text}",
                            "assistant_message_metadata": {
                                "turn_index": assistant_turn_index,
                                "source_stage": "assistant_base",
                                "provider": config.provider,
                                "model": config.model,
                                "token_usage": usage or {},
                                "parent_message_id": last_message.message_id,
                            },
                        }
                    )
                write_inference_result(
                    run_dir,
                    sample.sample_id,
                    payload,
                    materialize=False,
                )

    if pending_ids:
        await provider.aclose()
        del provider
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    canonical_path = materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    output_rows = []
    for sample in samples:
        user_messages = [msg.content for msg in sample.messages if msg.role == "user"]
        assistant_messages = [
            msg.content for msg in sample.messages if msg.role == "assistant"
        ]
        output_rows.append(
            {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "response_index": sample.response_index,
                "question": user_messages[-1] if user_messages else "",
                "response": assistant_messages[-1] if assistant_messages else "",
            }
        )
    result_dataset = Dataset.from_list(output_rows)
    result = InferenceResult(
        output_path=canonical_path,
        num_samples=len(result_dataset),
        num_failed=failed_count,
        total_input_tokens=total_usage["input_tokens"],
        total_output_tokens=total_usage["output_tokens"],
        total_tokens=total_usage["total_tokens"],
    )
    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            "\n".join(json.dumps(row) for row in output_rows)
            + ("\n" if output_rows else "")
        )
    return result_dataset, result


def _canonical_provider_input_for_sample(
    sample, config: InferenceConfig
) -> PromptInput:
    """Render one canonical sample into the provider input type."""
    if not sample.messages:
        raise ValueError(f"sample_id={sample.sample_id} has no messages.")
    if sample.messages[-1].role != "user":
        return [{"role": msg.role, "content": msg.content} for msg in sample.messages]

    if config.provider == "local" and config.local.prompt_format == "plain":
        return _plain_transcript(sample.messages)

    return [{"role": msg.role, "content": msg.content} for msg in sample.messages]


def _plain_transcript(messages) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"{message.role.capitalize()}:\n{message.content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _canonical_assistant_message_id(sample_id: str, assistant_turn_index: int) -> str:
    suffix = sample_id.split("sample_")[-1]
    return f"msg_{suffix}_assistant_{assistant_turn_index}"


def run_inference(
    config: InferenceConfig, dataset: Dataset | None = None
) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset (sync wrapper).

    Use run_inference_async for async contexts. This wrapper will fail if
    called while an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_inference_async(config, dataset))
    raise RuntimeError(
        "run_inference called inside a running event loop. "
        "Use run_inference_async instead."
    )

"""OpenAI Batch API helper for inference using Responses endpoint."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from openai import OpenAI

from src.utils import write_jsonl, setup_logging

if TYPE_CHECKING:
    from datasets import Dataset
    from src.inference.config import InferenceConfig, InferenceResult

logger = logging.getLogger(__name__)


def _extract_output_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, dict):
        text = body.get("output_text")
        if isinstance(text, str):
            return text.strip()
        outputs = body.get("output") or []
        parts: list[str] = []
        for item in outputs:
            content = item.get("content") or []
            for block in content:
                block_type = block.get("type")
                if block_type in ("output_text", "text"):
                    block_text = block.get("text")
                    if block_text:
                        parts.append(block_text)
        return "".join(parts).strip()
    return str(body).strip()


def _read_file_content(file_content: Any) -> str:
    if isinstance(file_content, (bytes, bytearray)):
        return file_content.decode("utf-8")
    text_attr = getattr(file_content, "text", None)
    if callable(text_attr):
        return text_attr()
    if isinstance(text_attr, str):
        return text_attr
    read_attr = getattr(file_content, "read", None)
    if callable(read_attr):
        data = read_attr()
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
    return str(file_content)


def run_openai_batch_inference(
    config: "InferenceConfig",
    dataset: "Dataset",
) -> tuple["Dataset", "InferenceResult"]:
    """Run inference using the OpenAI Batch API against the Responses endpoint."""
    raise NotImplementedError(
        "OpenAI Batch inference is not yet tested. Remove this guard before use."
    )
    setup_logging()
    openai_cfg = config.openai
    batch_cfg = openai_cfg.batch
    gen_cfg = config.generation

    api_key = os.environ.get(openai_cfg.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key not found. Set the {openai_cfg.api_key_env} environment variable."
        )

    client_kwargs: dict[str, object] = {"api_key": api_key}
    if openai_cfg.base_url:
        client_kwargs["base_url"] = openai_cfg.base_url
        logger.info("Using custom base URL: %s", openai_cfg.base_url)

    client = OpenAI(**client_kwargs)

    run_id = batch_cfg.run_dir or datetime.now().strftime("openai-batch-%Y%m%d-%H%M%S")
    output_dir = Path("scratch") / run_id
    if config.output_path:
        output_dir = Path(config.output_path).parent / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "openai_batch_input.jsonl"
    raw_output_path = output_dir / "openai_batch_output.jsonl"
    error_output_path = output_dir / "openai_batch_errors.jsonl"
    metadata_path = output_dir / "openai_batch_metadata.json"

    prompts = dataset["question"]
    num_responses = max(1, gen_cfg.num_responses_per_prompt)
    if num_responses > 1 and not gen_cfg.do_sample:
        raise ValueError(
            "num_responses_per_prompt > 1 requires do_sample=True. "
            "Set generation.do_sample to True."
        )

    if batch_cfg.resume:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Resume requested but metadata not found: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        batch_id = metadata.get("batch_id")
        if not batch_id:
            raise RuntimeError(
                f"Resume metadata missing batch_id: {metadata_path}"
            )
        input_path = Path(metadata.get("input_path", input_path))
        raw_output_path = Path(metadata.get("output_path", raw_output_path))
        error_output_path = Path(metadata.get("error_path", error_output_path))
        batch = client.batches.retrieve(batch_id)
    else:
        requests: list[dict[str, Any]] = []
        max_output_tokens = gen_cfg.max_new_tokens

        for question_index, prompt in enumerate(prompts):
            for response_index in range(num_responses):
                body: dict[str, Any] = {
                    "model": config.model,
                    "input": [{"role": "user", "content": prompt}],
                    "max_output_tokens": max_output_tokens,
                    "text": {"format": {"type": "text"}},
                }
                if openai_cfg.verbosity:
                    body["text"]["verbosity"] = openai_cfg.verbosity
                if openai_cfg.reasoning_effort:
                    body["reasoning"] = {"effort": openai_cfg.reasoning_effort}
                if batch_cfg.include_sampling:
                    body["temperature"] = gen_cfg.temperature
                    body["top_p"] = gen_cfg.top_p

                requests.append(
                    {
                        "custom_id": f"{question_index}:{response_index}",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }
                )

        write_jsonl(requests, input_path)
        logger.info("Wrote batch input file: %s", input_path)

        with input_path.open("rb") as handle:
            input_file = client.files.create(file=handle, purpose="batch")

        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window=batch_cfg.completion_window,
        )

        logger.info("Submitted batch job: %s", batch.id)
        metadata_path.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "input_path": str(input_path),
                    "output_path": str(raw_output_path),
                    "error_path": str(error_output_path),
                    "created_at": datetime.now().isoformat(),
                },
                indent=2,
            )
        )

    start_time = time.time()
    while True:
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        metadata_path.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "status": status,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": batch.error_file_id,
                    "updated_at": datetime.now().isoformat(),
                    "input_path": str(input_path),
                    "output_path": str(raw_output_path),
                    "error_path": str(error_output_path),
                },
                indent=2,
            )
        )
        if status in {"completed", "failed", "cancelled", "expired"}:
            break
        if batch_cfg.timeout_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed > batch_cfg.timeout_seconds:
                raise TimeoutError(
                    f"Batch job timed out after {batch_cfg.timeout_seconds} seconds."
                )
        time.sleep(max(1, batch_cfg.poll_interval_seconds))

    logger.info("Batch job %s finished with status: %s", batch.id, batch.status)
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_output_path.write_text(_read_file_content(error_content))
        logger.warning("Batch error file saved to %s", error_output_path)

    if batch.status != "completed":
        raise RuntimeError(f"Batch job did not complete successfully: {batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed without an output file id.")

    output_content = client.files.content(batch.output_file_id)
    raw_output_path.write_text(_read_file_content(output_content))
    logger.info("Batch output saved to %s", raw_output_path)

    outputs: dict[tuple[int, int], str] = {}
    errors: list[str] = []
    for line in raw_output_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if not custom_id:
            continue

        try:
            question_index_str, response_index_str = custom_id.split(":", 1)
            question_index = int(question_index_str)
            response_index = int(response_index_str)
        except ValueError:
            logger.warning("Unexpected custom_id format: %s", custom_id)
            continue

        response_obj = item.get("response")
        if isinstance(response_obj, dict) and "body" in response_obj:
            body = response_obj.get("body")
        else:
            body = response_obj

        if item.get("error"):
            errors.append(custom_id)
            outputs[(question_index, response_index)] = ""
            continue

        outputs[(question_index, response_index)] = _extract_output_text(body)

    if errors:
        logger.warning("Batch completed with %d request errors.", len(errors))

    questions_out: list[str] = []
    responses_out: list[str] = []
    response_indices: list[int] = []

    for question_index, question in enumerate(prompts):
        for response_index in range(num_responses):
            questions_out.append(question)
            responses_out.append(outputs.get((question_index, response_index), ""))
            response_indices.append(response_index)

    from datasets import Dataset
    from src.inference.config import InferenceResult

    result_dataset = Dataset.from_list(
        [
            {
                "question": question,
                "response": response,
                "response_index": response_index,
            }
            for question, response, response_index in zip(
                questions_out, responses_out, response_indices
            )
        ]
    )

    result = InferenceResult(
        num_samples=len(result_dataset),
        output_path=None,
        batch_id=batch.id,
        batch_status=batch.status,
    )

    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(result_dataset.to_list(), save_path)
        logger.info("Saved inference output to %s", save_path)
        result.output_path = save_path

    return result_dataset, result

"""Core canonical dataset and lineage operations."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.datasets.io import (
    append_jsonl,
    json_dumps,
    read_jsonl_tolerant,
    write_jsonl_atomic,
)
from src.datasets.schema import (
    SCHEMA_VERSION,
    CanonicalInput,
    CanonicalMessage,
    EditOverlay,
    EditVariant,
    ExportProfile,
    InferenceData,
    MetricAnnotationRecord,
    RunManifest,
    SampleRecord,
    StageEventRecord,
    SystemPromptRecord,
)


def get_run_paths(run_dir: str | Path) -> dict[str, Path]:
    """Return canonical file paths for a run directory."""
    root = Path(run_dir)
    return {
        "run_dir": root,
        "manifest": root / "manifest.json",
        "sample_inputs": root / "datasets" / "sample_inputs.jsonl",
        "canonical_samples": root / "datasets" / "canonical_samples.jsonl",
        "edit_events": root / "datasets" / "edit_events.jsonl",
        "message_events": root / "datasets" / "message_events.jsonl",
        "metric_events": root / "datasets" / "metric_events.jsonl",
        "stage_events": root / "events" / "stage_events.jsonl",
        "exports_dir": root / "exports",
        "reports_dir": root / "reports",
    }


def init_run(
    run_dir: str | Path, base_config: dict[str, Any] | None = None
) -> RunManifest:
    """Initialize or load a canonical run manifest."""
    paths = get_run_paths(run_dir)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["sample_inputs"].parent.mkdir(parents=True, exist_ok=True)
    paths["stage_events"].parent.mkdir(parents=True, exist_ok=True)
    paths["exports_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    manifest_path = paths["manifest"]
    if manifest_path.exists():
        manifest = load_manifest(run_dir)
        if base_config:
            manifest.base_config = manifest.base_config or deepcopy(base_config)
            _save_manifest(paths, manifest)
        return manifest

    now = _now_iso()
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=paths["run_dir"].name,
        created_at=now,
        updated_at=now,
        git_commit_hash=_get_git_commit_hash(),
        cli_command=" ".join(sys.argv),
        files={
            "sample_inputs": str(paths["sample_inputs"]),
            "canonical_samples": str(paths["canonical_samples"]),
            "edit_events": str(paths["edit_events"]),
            "message_events": str(paths["message_events"]),
            "metric_events": str(paths["metric_events"]),
            "stage_events": str(paths["stage_events"]),
        },
        base_config=deepcopy(base_config) if base_config else {},
    )
    _save_manifest(paths, manifest)
    return manifest


def load_manifest(run_dir: str | Path) -> RunManifest:
    """Load run manifest."""
    paths = get_run_paths(run_dir)
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RunManifest.model_validate(payload)


def register_stage_fingerprint(
    run_dir: str | Path,
    stage: str,
    config_payload: dict[str, Any],
) -> str:
    """Register stage config fingerprint and fail on mismatch."""
    paths = get_run_paths(run_dir)
    manifest = init_run(run_dir)
    normalized_payload = _normalize_for_hash(config_payload)
    digest = _hash_object(normalized_payload)
    stage_configs = manifest.progress.setdefault("stage_configs", {})
    existing_payload = stage_configs.get(stage)
    existing = manifest.stage_fingerprints.get(stage)
    if existing and existing != digest:
        diff = _describe_payload_diff(existing_payload, normalized_payload)
        raise ValueError(
            f"Stage fingerprint mismatch for '{stage}'. "
            f"manifest={existing} current={digest}. diff={diff}"
        )
    manifest.stage_fingerprints[stage] = digest
    stage_configs[stage] = normalized_payload
    _save_manifest(paths, manifest)
    return digest


def ingest_source_dataset(
    dataset: Any,
    source_info: dict[str, Any] | None,
    system_prompt: str | None,
    run_dir: str | Path,
    *,
    overwrite: bool = False,
    responses_per_input: int = 1,
) -> list[SampleRecord]:
    """Ingest source dataset into canonical sample inputs."""
    paths = get_run_paths(run_dir)
    manifest = init_run(run_dir)
    rows = _normalize_rows(dataset)

    prompt_id = _register_system_prompt(manifest, system_prompt)
    samples = _build_samples(
        rows,
        source_info=source_info or {},
        prompt_ref=prompt_id,
        system_prompt_text=system_prompt,
        responses_per_input=max(1, int(responses_per_input)),
        manifest=manifest,
    )
    _save_manifest(paths, manifest)
    dataset_fingerprint = _compute_dataset_fingerprint(samples)

    if paths["sample_inputs"].exists() and not overwrite:
        if (
            manifest.dataset_fingerprint
            and manifest.dataset_fingerprint != dataset_fingerprint
        ):
            existing = _load_sample_inputs(run_dir)
            current_ids = {sample.sample_id for sample in samples}
            existing_ids = {sample.sample_id for sample in existing}
            missing_in_current = sorted(existing_ids - current_ids)[:5]
            new_in_current = sorted(current_ids - existing_ids)[:5]
            raise ValueError(
                "Dataset fingerprint mismatch during resume. "
                f"manifest={manifest.dataset_fingerprint} current={dataset_fingerprint}. "
                f"existing_count={len(existing_ids)} current_count={len(current_ids)} "
                f"missing_in_current={missing_in_current} new_in_current={new_in_current}"
            )
        existing = _load_sample_inputs(run_dir)
        return existing

    manifest.dataset_fingerprint = dataset_fingerprint
    _save_manifest(paths, manifest)
    write_jsonl_atomic(
        paths["sample_inputs"], [sample.model_dump() for sample in samples]
    )
    materialize_canonical_samples(run_dir)
    return samples


def load_sample_inputs(run_dir: str | Path) -> list[SampleRecord]:
    """Load immutable sample input rows."""
    return _load_sample_inputs(run_dir)


def load_samples(run_dir: str | Path) -> list[SampleRecord]:
    """Load materialized canonical samples."""
    paths = get_run_paths(run_dir)
    materialize_canonical_samples(run_dir)
    rows, recovered = read_jsonl_tolerant(paths["canonical_samples"])
    if recovered:
        record_stage_event(
            run_dir,
            StageEventRecord(
                event_id=_event_id("warning", "truncated_row_recovery"),
                stage="datasets",
                event_type="truncated_row_recovery",
                created_at=_now_iso(),
                payload={"file": str(paths["canonical_samples"])},
            ),
        )
    return [SampleRecord.model_validate(row) for row in rows]


def record_stage_event(
    run_dir: str | Path,
    event: StageEventRecord,
    *,
    lightweight: bool = False,
) -> None:
    """Append a stage event.

    Args:
        run_dir: Run directory path.
        event: The event record to persist.
        lightweight: If True, skip fsync and manifest rewrite for speed.
            Use in high-throughput loops (e.g. per-turn rollout events)
            where losing a few events on crash is acceptable.
    """
    paths = get_run_paths(run_dir)
    append_jsonl(paths["stage_events"], event.model_dump(), fsync=not lightweight)
    if not lightweight:
        manifest = load_manifest(run_dir)
        manifest.updated_at = _now_iso()
        _save_manifest(paths, manifest)


def write_inference_result(
    run_dir: str | Path,
    sample_id: str,
    inference_payload: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Record inference output for a sample."""
    event = StageEventRecord(
        event_id=_event_id(sample_id, "inference_result"),
        stage="inference",
        event_type="inference_result",
        sample_id=sample_id,
        created_at=_now_iso(),
        payload=deepcopy(inference_payload),
    )
    record_stage_event(run_dir, event, lightweight=not materialize)
    if materialize:
        materialize_canonical_samples(run_dir)


def write_edit_overlay(
    run_dir: str | Path,
    sample_id: str,
    variant_name: str,
    overlay_payload: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Record a single edit overlay."""
    paths = get_run_paths(run_dir)
    payload = {
        "event_id": _event_id(sample_id, f"edit:{variant_name}"),
        "sample_id": sample_id,
        "variant_name": variant_name,
        "created_at": _now_iso(),
        "overlay": deepcopy(overlay_payload),
    }
    append_jsonl(paths["edit_events"], payload)
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=_event_id(sample_id, f"edit_event:{variant_name}"),
            stage="editing",
            event_type="edit_overlay",
            sample_id=sample_id,
            created_at=_now_iso(),
            payload={
                "variant_name": variant_name,
                "overlay_id": overlay_payload.get("overlay_id"),
            },
        ),
    )
    if materialize:
        materialize_canonical_samples(run_dir)


def write_message_append(
    run_dir: str | Path,
    sample_id: str,
    message_payload: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Record one appended conversation message."""
    paths = get_run_paths(run_dir)
    payload = {
        "event_id": _event_id(sample_id, "message_append"),
        "sample_id": sample_id,
        "created_at": _now_iso(),
        "message": deepcopy(message_payload),
    }
    append_jsonl(paths["message_events"], payload, fsync=materialize)
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=_event_id(sample_id, "message_append_event"),
            stage="conversation",
            event_type="message_append",
            sample_id=sample_id,
            created_at=_now_iso(),
            payload={
                "message_id": message_payload.get("message_id"),
                "role": message_payload.get("role"),
            },
        ),
        lightweight=not materialize,
    )
    if materialize:
        materialize_canonical_samples(run_dir)


def write_metric_annotation(
    run_dir: str | Path,
    sample_id: str,
    candidate_ref: str,
    metrics_payload: dict[str, Any],
    *,
    metrics_key: str,
    evaluator_metadata: dict[str, Any] | None = None,
    materialize: bool = True,
) -> None:
    """Record metric annotations for a sample candidate."""
    paths = get_run_paths(run_dir)
    payload = {
        "annotation_id": _hash_text(
            f"{sample_id}:{candidate_ref}:{metrics_key}:{json_dumps(metrics_payload)}"
        )[:24],
        "sample_id": sample_id,
        "candidate_ref": candidate_ref,
        "metrics_key": metrics_key,
        "metrics": deepcopy(metrics_payload),
        "evaluator_metadata": deepcopy(evaluator_metadata)
        if evaluator_metadata
        else {},
        "created_at": _now_iso(),
    }
    append_jsonl(paths["metric_events"], payload)
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=_event_id(sample_id, f"metric:{metrics_key}"),
            stage="persona_metrics",
            event_type="metric_annotation",
            sample_id=sample_id,
            created_at=_now_iso(),
            payload={"candidate_ref": candidate_ref, "metrics_key": metrics_key},
        ),
    )
    if materialize:
        materialize_canonical_samples(run_dir)


def materialize_canonical_samples(run_dir: str | Path) -> Path:
    """Materialize canonical rows from immutable inputs and append-only events."""
    paths = get_run_paths(run_dir)
    samples = _load_sample_inputs(run_dir)
    index: dict[str, SampleRecord] = {
        sample.sample_id: deepcopy(sample) for sample in samples
    }

    stage_events, _ = read_jsonl_tolerant(paths["stage_events"])
    for raw in stage_events:
        if raw.get("event_type") != "inference_result":
            continue
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in index:
            continue
        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            continue
        sample = index[sample_id]
        inference = InferenceData.model_validate(
            {**sample.inference.model_dump(), **payload}
        )
        sample.inference = inference
        sample.lineage["inference"] = {
            "status": inference.status,
            "attempt_no": inference.attempt_no,
            "updated_at": raw.get("created_at"),
            "model": inference.model,
            "provider": inference.provider,
        }
        if inference.status == "success" and inference.assistant_completion is not None:
            full = inference.assistant_full
            if full is None:
                prefill = (
                    inference.assistant_prefill or sample.input.assistant_prefill or ""
                )
                full = f"{prefill}{inference.assistant_completion}"
            message_id = inference.assistant_message_id or _assistant_message_id(
                sample.sample_id
            )
            payload_metadata = payload.get("assistant_message_metadata")
            assistant_message = CanonicalMessage(
                message_id=message_id,
                role="assistant",
                content=full,
                message_metadata=deepcopy(payload_metadata)
                if isinstance(payload_metadata, dict)
                else None,
                editable=True,
            )
            sample.messages = _upsert_message(sample.messages, assistant_message)
            sample.inference.assistant_message_id = message_id
            sample.inference.assistant_full = full

    message_events, _ = read_jsonl_tolerant(paths["message_events"])
    for raw in message_events:
        sample_id = raw.get("sample_id")
        message_payload = raw.get("message")
        if (
            not isinstance(sample_id, str)
            or sample_id not in index
            or not isinstance(message_payload, dict)
        ):
            continue
        sample = index[sample_id]
        message = CanonicalMessage.model_validate(message_payload)
        sample.messages = _upsert_message(sample.messages, message)
        conversation_lineage = sample.lineage.setdefault("conversation", {})
        if isinstance(conversation_lineage, dict):
            conversation_lineage[message.message_id] = {
                "role": message.role,
                "updated_at": raw.get("created_at"),
                "source_stage": (message.message_metadata or {}).get("source_stage"),
            }

    edit_events, _ = read_jsonl_tolerant(paths["edit_events"])
    for raw in edit_events:
        sample_id = raw.get("sample_id")
        variant_name = raw.get("variant_name")
        overlay_payload = raw.get("overlay")
        if (
            not isinstance(sample_id, str)
            or sample_id not in index
            or not isinstance(variant_name, str)
            or not isinstance(overlay_payload, dict)
        ):
            continue
        sample = index[sample_id]
        variant = _get_or_create_variant(sample, variant_name)
        overlay = EditOverlay.model_validate(overlay_payload)
        _upsert_overlay(variant, overlay)
        variant.status = _resolve_variant_status(variant.overlays)
        editing_lineage = sample.lineage.setdefault("editing", {})
        if isinstance(editing_lineage, dict):
            editing_lineage[variant_name] = {
                "status": variant.status,
                "attempt_no": overlay.attempt_no,
                "updated_at": raw.get("created_at"),
                "editor_model": overlay.editor_model,
                "editor_provider": overlay.editor_provider,
            }

    metric_events, _ = read_jsonl_tolerant(paths["metric_events"])
    for raw in metric_events:
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in index:
            continue
        sample = index[sample_id]
        annotation = MetricAnnotationRecord.model_validate(raw)
        bucket = sample.metrics.setdefault(annotation.metrics_key, [])
        existing_idx = _find_annotation_index(bucket, annotation.annotation_id)
        if existing_idx is None:
            bucket.append(annotation)
        else:
            bucket[existing_idx] = annotation
        metrics_lineage = sample.lineage.setdefault("metrics", {})
        if isinstance(metrics_lineage, dict):
            metrics_lineage[annotation.metrics_key] = {
                "candidate_ref": annotation.candidate_ref,
                "updated_at": annotation.created_at,
            }

    materialized_rows: list[dict[str, Any]] = []
    for sample in samples:
        rendered = index[sample.sample_id]
        rendered.messages = _sort_messages(rendered.messages)
        rendered.lineage = {**rendered.lineage, "last_materialized_at": _now_iso()}
        materialized_rows.append(rendered.model_dump())

    write_jsonl_atomic(paths["canonical_samples"], materialized_rows)
    manifest = load_manifest(run_dir)
    manifest.updated_at = _now_iso()
    _save_manifest(paths, manifest)
    return paths["canonical_samples"]


def resume_state(
    run_dir: str | Path,
    stage: str,
    variant_name: str | None = None,
    *,
    max_attempts: int | None = None,
) -> dict[str, list[str]]:
    """Return pending/complete/failed sample IDs for a stage."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    pending: list[str] = []
    complete: list[str] = []
    failed: list[str] = []
    terminal: list[str] = []

    for sample in samples:
        if stage == "inference":
            status = sample.inference.status
            attempts = int(sample.inference.attempt_no)
            if status == "success":
                complete.append(sample.sample_id)
            else:
                exhausted = max_attempts is not None and attempts >= max_attempts
                if exhausted:
                    terminal.append(sample.sample_id)
                else:
                    pending.append(sample.sample_id)
                    if status == "failed":
                        failed.append(sample.sample_id)
            continue

        if stage == "editing":
            if not variant_name:
                raise ValueError("variant_name is required for editing resume state.")
            variant = _find_variant(sample, variant_name)
            latest_assistant = next(
                (
                    message
                    for message in reversed(sample.messages)
                    if message.role == "assistant"
                ),
                None,
            )
            latest_overlay = (
                _latest_success_overlay_for_target(variant, latest_assistant.message_id)
                if variant is not None and latest_assistant is not None
                else None
            )
            if latest_assistant is not None and latest_overlay is not None:
                complete.append(sample.sample_id)
            else:
                attempts = (
                    max(
                        (
                            overlay.attempt_no
                            for overlay in variant.overlays
                            if latest_assistant is None
                            or overlay.target_message_id == latest_assistant.message_id
                        ),
                        default=0,
                    )
                    if variant is not None
                    else 0
                )
                exhausted = max_attempts is not None and attempts >= max_attempts
                if exhausted:
                    terminal.append(sample.sample_id)
                else:
                    pending.append(sample.sample_id)
                    if variant and variant.status == "failed":
                        failed.append(sample.sample_id)
            continue

        raise ValueError(f"Unsupported stage for resume_state: {stage}")

    return {
        "pending": sorted(pending),
        "complete": sorted(complete),
        "failed": sorted(failed),
        "terminal": sorted(terminal),
    }


def render_messages(
    sample: SampleRecord, variant_name: str | None = None
) -> list[CanonicalMessage]:
    """Render one sample's messages with latest successful overlays applied."""
    rendered = [message.model_copy(deep=True) for message in sample.messages]
    if variant_name is None:
        return rendered

    variant = _find_variant(sample, variant_name)
    if variant is None:
        return rendered

    overlays_by_target: dict[str, EditOverlay] = {}
    for overlay in variant.overlays:
        if overlay.status != "success":
            continue
        previous = overlays_by_target.get(overlay.target_message_id)
        if previous is None or (overlay.attempt_no, overlay.overlay_id) > (
            previous.attempt_no,
            previous.overlay_id,
        ):
            overlays_by_target[overlay.target_message_id] = overlay

    updated: list[CanonicalMessage] = []
    for message in rendered:
        overlay = overlays_by_target.get(message.message_id)
        if overlay is None:
            updated.append(message)
            continue
        updated.append(
            CanonicalMessage(
                message_id=message.message_id,
                role=message.role,
                content=overlay.edited_content,
                name=message.name,
                tool_metadata=deepcopy(message.tool_metadata),
                message_metadata=deepcopy(message.message_metadata),
                editable=message.editable,
            )
        )
    return updated


def select_training_candidates(
    run_dir: str | Path,
    training_variant: str,
    *,
    skip_failed_rows: bool = False,
):
    """Return a single-turn training dataset for one edit variant.

    Multi-turn training is intentionally unimplemented in phase 1.
    """
    from datasets import Dataset

    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    rows: list[dict[str, Any]] = []
    inference_not_success: list[str] = []
    variant_missing: list[str] = []
    variant_without_success: list[str] = []
    for sample in samples:
        user_messages = [msg for msg in sample.messages if msg.role == "user"]
        assistant_messages = [msg for msg in sample.messages if msg.role == "assistant"]
        if len(user_messages) != 1 or len(assistant_messages) > 1:
            raise ValueError(
                "Multi-turn training is unimplemented in phase 1. "
                f"sample_id={sample.sample_id} has {len(user_messages)} user turns and "
                f"{len(assistant_messages)} assistant turns."
            )

        if (
            sample.inference.status != "success"
            or sample.inference.assistant_completion is None
        ):
            inference_not_success.append(sample.sample_id)
            continue

        variant = _find_variant(sample, training_variant)
        if variant is None:
            variant_missing.append(sample.sample_id)
            continue
        latest = _latest_success_overlay(variant)
        if latest is None:
            variant_without_success.append(sample.sample_id)
            continue

        question = user_messages[0].content
        edited = latest.edited_content
        base_response = sample.inference.assistant_completion or ""
        rows.append(
            {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "response_index": sample.response_index,
                "question": question,
                "response": base_response,
                "edited_response": edited,
                "prompt_messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in sample.input.messages
                ],
                "assistant_prefill": sample.input.assistant_prefill,
                "assistant_completion": edited,
                "training_variant": training_variant,
            }
        )

    if not skip_failed_rows:
        problem_count = (
            len(inference_not_success)
            + len(variant_missing)
            + len(variant_without_success)
        )
        if problem_count:
            raise ValueError(
                "Canonical training candidates include rows that are not ready. "
                f"inference_not_success={len(inference_not_success)} "
                f"variant_missing={len(variant_missing)} "
                f"variant_without_success={len(variant_without_success)}. "
                "Resolve upstream failures or pass skip_failed_rows=True "
                "(--skip-failed-rows) to train on completed rows only."
            )

    if not rows:
        raise ValueError(
            f"No successful edited samples found for training_variant='{training_variant}'."
        )
    return Dataset.from_list(rows)


def export_dataset(
    run_dir: str | Path,
    profile: str = "minimal_train_eval",
    variant_name: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    rename: dict[str, str] | None = None,
) -> Path:
    """Export canonical dataset rows using a minimal profile by default."""
    include = include or []
    exclude = exclude or []
    rename = rename or {}
    export_profile = ExportProfile(
        name=profile, include=include, exclude=exclude, rename=rename
    )

    paths = get_run_paths(run_dir)
    materialize_canonical_samples(run_dir)
    manifest = load_manifest(run_dir)
    samples = load_samples(run_dir)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        rendered_messages = render_messages(sample)
        user_messages = [msg.content for msg in rendered_messages if msg.role == "user"]
        variants: dict[str, str] = {}
        for variant in sample.edit_variants:
            latest = _latest_success_overlay(variant)
            if latest is not None:
                variants[variant.variant_name] = latest.edited_content
        if profile == "conversation_training":
            final_messages = render_messages(sample, variant_name)
            row = {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "messages": [msg.model_dump() for msg in final_messages],
                "assistant_turn_count": sum(
                    1 for msg in final_messages if msg.role == "assistant"
                ),
                "editing_variant": variant_name,
                "source_info": deepcopy(sample.source_info),
            }
        elif profile == "conversation_trace":
            edited_by_message: dict[str, str] = {}
            for variant in sample.edit_variants:
                if variant_name and variant.variant_name != variant_name:
                    continue
                for overlay in variant.overlays:
                    if overlay.status == "success":
                        edited_by_message[overlay.target_message_id] = (
                            overlay.edited_content
                        )
            row = {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "base_messages": [msg.model_dump() for msg in sample.messages],
                "edited_messages": [
                    msg.model_dump() for msg in render_messages(sample, variant_name)
                ],
                "edited_assistant_messages": edited_by_message,
                "editing_variants": [
                    variant.variant_name for variant in sample.edit_variants
                ],
                "source_info": deepcopy(sample.source_info),
            }
        else:
            row = {
                "sample_id": sample.sample_id,
                "input_group_id": sample.input_group_id or sample.sample_id,
                "response_index": sample.response_index,
                "user_messages": user_messages,
                "base_response": sample.inference.assistant_completion,
                "edited_variants": variants,
                "system_prompt_ref": sample.input.system_prompt_ref,
                "inference_model": sample.inference.model,
                "inference_provider": sample.inference.provider,
                "git_commit_hash": manifest.git_commit_hash,
            }
        rows.append(_apply_export_transform(row, export_profile))

    output_path = paths["exports_dir"] / f"{profile}.jsonl"
    write_jsonl_atomic(output_path, rows)
    return output_path


def migrate_legacy_jsonl(
    input_path: str | Path,
    run_dir: str | Path,
    *,
    system_prompt: str | None = None,
    source_info: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Migrate legacy flat JSONL into canonical run artifacts."""
    legacy_path = Path(input_path)
    rows, recovered = read_jsonl_tolerant(legacy_path)

    init_run(run_dir, base_config={"migration_source": str(legacy_path)})
    samples = ingest_source_dataset(
        rows,
        source_info=source_info or {"source": "legacy_jsonl", "path": str(legacy_path)},
        system_prompt=system_prompt,
        run_dir=run_dir,
        overwrite=True,
    )

    mapping: list[dict[str, Any]] = []
    for idx, (sample, row) in enumerate(zip(samples, rows)):
        mapping.append({"legacy_row_index": idx, "sample_id": sample.sample_id})
        response = row.get("response")
        if isinstance(response, str) and response.strip():
            write_inference_result(
                run_dir,
                sample.sample_id,
                {
                    "status": "success",
                    "model": str(row.get("inference_model", "legacy")),
                    "provider": str(row.get("inference_provider", "legacy")),
                    "assistant_message_id": _assistant_message_id(sample.sample_id),
                    "assistant_completion": response,
                    "assistant_prefill": sample.input.assistant_prefill,
                    "assistant_full": f"{sample.input.assistant_prefill or ''}{response}",
                    "attempt_no": 1,
                    "started_at": _now_iso(),
                    "completed_at": _now_iso(),
                },
                materialize=False,
            )

        edited = row.get("edited_response")
        if isinstance(edited, str) and edited.strip():
            variant_name = str(row.get("variant_name", "legacy_default"))
            overlay_id = _hash_text(f"{sample.sample_id}:{variant_name}:{idx}")[:24]
            write_edit_overlay(
                run_dir,
                sample.sample_id,
                variant_name=variant_name,
                overlay_payload={
                    "overlay_id": overlay_id,
                    "target_message_id": _assistant_message_id(sample.sample_id),
                    "target_role": "assistant",
                    "original_content_hash": _hash_text(str(response or "")),
                    "edited_content": edited,
                    "status": "success",
                    "attempt_no": 1,
                    "editor_model": str(row.get("editor_model", "legacy")),
                    "editor_provider": str(row.get("editor_provider", "legacy")),
                    "edit_prompt_hash": str(row.get("edit_prompt_hash", "legacy")),
                    "timestamps": {"created_at": _now_iso()},
                },
                materialize=False,
            )

        for metrics_key in ("quality_metrics", "persona_metrics"):
            payload = row.get(metrics_key)
            if isinstance(payload, dict) and payload:
                write_metric_annotation(
                    run_dir,
                    sample.sample_id,
                    candidate_ref="legacy",
                    metrics_payload=payload,
                    metrics_key=metrics_key,
                    evaluator_metadata={"source": "legacy_migration"},
                    materialize=False,
                )

    materialize_canonical_samples(run_dir)
    report_path = get_run_paths(run_dir)["reports_dir"] / "migration_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(legacy_path),
                "recovered_truncated_row": recovered,
                "mappings": mapping,
                "num_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Path(run_dir), report_path


def validate_run(run_dir: str | Path) -> None:
    """Validate canonical files and schema parseability."""
    init_run(run_dir)
    materialize_canonical_samples(run_dir)
    _ = load_samples(run_dir)


def find_consecutive_assistant_turn_sample_ids(run_dir: str | Path) -> set[str]:
    """Return IDs of samples that contain consecutive assistant turns.

    Consecutive assistant turns arise from a resume-after-interrupt bug: when
    the process is killed after an assistant turn is written to disk but before
    the following user turn is attempted, the next resume starts the assistant
    loop at the wrong index, generating an assistant turn whose parent is
    another assistant turn rather than a user turn.

    These samples are structurally invalid for downstream analysis (questionnaire
    responses will be confounded) and should be excluded.

    Args:
        run_dir: Path to the rollout run directory.

    Returns:
        Set of sample_id strings for affected samples.
    """
    samples = load_samples(run_dir)
    bad: set[str] = set()
    for sample in samples:
        msgs = sample.messages
        for i in range(1, len(msgs)):
            if msgs[i].role == "assistant" and msgs[i - 1].role == "assistant":
                bad.add(sample.sample_id)
                break
    return bad


def _build_samples(
    rows: list[dict[str, Any]],
    *,
    source_info: dict[str, Any],
    prompt_ref: str | None,
    system_prompt_text: str | None,
    responses_per_input: int,
    manifest: RunManifest | None = None,
) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for row_idx, row in enumerate(rows):
        input_messages_raw = _build_input_messages_from_row(
            row,
            prompt_ref=prompt_ref,
            system_prompt_text=system_prompt_text,
        )
        effective_prompt_ref = prompt_ref
        row_system_prompt = row.get("system_prompt")
        if (
            isinstance(row_system_prompt, str)
            and row_system_prompt
            and manifest is not None
        ):
            effective_prompt_ref = (
                _register_system_prompt(manifest, row_system_prompt) or prompt_ref
            )
        assistant_prefill_raw = row.get("assistant_prefill")
        assistant_prefill = (
            assistant_prefill_raw if isinstance(assistant_prefill_raw, str) else None
        )
        input_group_id = _sample_id(
            messages=[_message_for_hash(msg) for msg in input_messages_raw],
            assistant_prefill=assistant_prefill,
            system_prompt_ref=effective_prompt_ref,
        )
        for response_index in range(max(1, responses_per_input)):
            sample_id = _response_sample_id(
                input_group_id=input_group_id,
                response_index=response_index,
                responses_per_input=responses_per_input,
            )
            input_messages = _assign_message_ids(sample_id, input_messages_raw)
            canonical_input = CanonicalInput(
                messages=input_messages,
                assistant_prefill=assistant_prefill,
                system_prompt_ref=effective_prompt_ref,
            )

            sample = SampleRecord(
                sample_id=sample_id,
                input_group_id=input_group_id,
                response_index=response_index,
                source_info={
                    **source_info,
                    "row_index": row_idx,
                },
                input=canonical_input,
                messages=deepcopy(input_messages),
                inference=InferenceData(status="pending"),
                edit_variants=[],
                metrics={},
                lineage={"created_at": _now_iso()},
            )
            samples.append(sample)
    return samples


def _build_input_messages_from_row(
    row: dict[str, Any],
    *,
    prompt_ref: str | None,
    system_prompt_text: str | None,
) -> list[CanonicalMessage]:
    system_prompt = None
    if isinstance(row.get("system_prompt"), str):
        system_prompt = row["system_prompt"]
    elif system_prompt_text:
        system_prompt = system_prompt_text

    raw_messages = row.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        parsed_messages: list[CanonicalMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, dict):
                continue
            role = raw.get("role")
            content = raw.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            editable_default = role != "system"
            parsed_messages.append(
                CanonicalMessage(
                    message_id=str(raw.get("message_id", "")),
                    role=role,
                    content=content,
                    name=raw.get("name") if isinstance(raw.get("name"), str) else None,
                    tool_metadata=raw.get("tool_metadata")
                    if isinstance(raw.get("tool_metadata"), dict)
                    else None,
                    message_metadata=raw.get("message_metadata")
                    if isinstance(raw.get("message_metadata"), dict)
                    else None,
                    editable=bool(raw.get("editable", editable_default)),
                )
            )
        if parsed_messages:
            if isinstance(system_prompt, str):
                existing_system = [
                    msg.content for msg in parsed_messages if msg.role == "system"
                ]
                if existing_system and any(
                    content != system_prompt for content in existing_system
                ):
                    raise ValueError(
                        "Conflicting system prompts found in input row. "
                        "Provide either row-level system messages or one run-level system prompt."
                    )
                parsed_messages = [
                    msg for msg in parsed_messages if msg.role != "system"
                ]
                parsed_messages.insert(
                    0,
                    CanonicalMessage(
                        message_id="",
                        role="system",
                        content=system_prompt,
                        editable=False,
                    ),
                )
            return parsed_messages

    messages: list[CanonicalMessage] = []
    if isinstance(system_prompt, str):
        messages.append(
            CanonicalMessage(
                message_id="",
                role="system",
                content=system_prompt,
                editable=False,
            )
        )

    question = _extract_question(row)
    if question is None:
        raise ValueError(
            "Could not derive canonical input messages from row. Expected either "
            "'messages' or one of 'question'/'instruction'/'prompt'/'text'."
        )
    messages.append(
        CanonicalMessage(
            message_id="",
            role="user",
            content=question,
            editable=True,
        )
    )
    return messages


def _extract_question(row: dict[str, Any]) -> str | None:
    for key in ("question", "instruction", "prompt", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            extra = row.get("input")
            if isinstance(extra, str) and extra.strip():
                return f"{value}\n\n{extra}"
            return value
    return None


def _assign_message_ids(
    sample_id: str, messages: list[CanonicalMessage]
) -> list[CanonicalMessage]:
    assigned: list[CanonicalMessage] = []
    conversational_turn = 0
    for idx, message in enumerate(messages):
        mid = message.message_id.strip() if message.message_id else ""
        if not mid:
            mid = _hash_text(f"{sample_id}:{idx}:{message.role}:{message.content}")[:24]
        turn_index = -1 if message.role == "system" else conversational_turn
        metadata = (
            deepcopy(message.message_metadata) if message.message_metadata else {}
        )
        metadata.setdefault("source_stage", "seed")
        metadata.setdefault("turn_index", turn_index)
        assigned.append(
            CanonicalMessage(
                message_id=f"msg_{mid}",
                role=message.role,
                content=message.content,
                name=message.name,
                tool_metadata=deepcopy(message.tool_metadata),
                message_metadata=metadata,
                editable=message.editable,
            )
        )
        if message.role in {"user", "assistant"}:
            conversational_turn += 1
    return assigned


def _load_sample_inputs(run_dir: str | Path) -> list[SampleRecord]:
    paths = get_run_paths(run_dir)
    rows, recovered = read_jsonl_tolerant(paths["sample_inputs"])
    if recovered:
        record_stage_event(
            run_dir,
            StageEventRecord(
                event_id=_event_id("warning", "sample_inputs_truncated_row_recovery"),
                stage="datasets",
                event_type="truncated_row_recovery",
                created_at=_now_iso(),
                payload={"file": str(paths["sample_inputs"])},
            ),
        )
    return [SampleRecord.model_validate(row) for row in rows]


def _compute_dataset_fingerprint(samples: list[SampleRecord]) -> str:
    payload = [
        {
            "sample_id": sample.sample_id,
            "input": _normalize_for_hash(sample.input.model_dump()),
        }
        for sample in samples
    ]
    payload.sort(key=lambda item: item["sample_id"])
    return _hash_object(payload)


def _register_system_prompt(
    manifest: RunManifest, system_prompt: str | None
) -> str | None:
    if system_prompt is None:
        return None
    prompt_hash = _hash_text(system_prompt)
    prompt_id = f"sp_{prompt_hash[:16]}"
    if prompt_id not in manifest.system_prompts:
        manifest.system_prompts[prompt_id] = SystemPromptRecord(
            prompt_id=prompt_id,
            prompt_hash=prompt_hash,
            inline_text=system_prompt if len(system_prompt) <= 10_000 else None,
            created_at=_now_iso(),
        )
    return prompt_id


def register_system_prompt(
    run_dir: str | Path, system_prompt: str | None
) -> str | None:
    """Register a system prompt in the run manifest so the full text is recoverable.

    Args:
        run_dir: Path to the run directory.
        system_prompt: The system prompt text, or None.

    Returns:
        The prompt_id (``sp_<hash16>``) if a prompt was registered, else None.
    """
    if system_prompt is None:
        return None
    paths = get_run_paths(run_dir)
    manifest = load_manifest(run_dir)
    prompt_id = _register_system_prompt(manifest, system_prompt)
    _save_manifest(paths, manifest)
    return prompt_id


def _save_manifest(paths: dict[str, Path], manifest: RunManifest) -> None:
    payload = manifest.model_dump()
    payload["updated_at"] = _now_iso()
    paths["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_rows(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "to_list"):
        records = dataset.to_list()
    elif isinstance(dataset, list):
        records = dataset
    elif isinstance(dataset, Iterable):
        records = list(dataset)
    else:
        raise TypeError(
            f"Unsupported dataset type for ingestion: {type(dataset).__name__}"
        )

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Dataset rows must be dictionaries.")
        normalized.append(deepcopy(record))
    return normalized


def _upsert_message(
    messages: list[CanonicalMessage], message: CanonicalMessage
) -> list[CanonicalMessage]:
    updated = []
    replaced = False
    for existing in messages:
        if existing.message_id == message.message_id:
            updated.append(message)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(message)
    return updated


def _sort_messages(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    """Sort messages: seed first, then within each turn assistant before user."""

    def _sort_key(item: tuple[int, CanonicalMessage]) -> tuple[int, int, int]:
        original_index, message = item
        metadata = message.message_metadata or {}
        turn_index_raw = metadata.get("turn_index")
        turn_index = (
            int(turn_index_raw) if isinstance(turn_index_raw, int) else original_index
        )
        source_stage = metadata.get("source_stage") or ""
        if source_stage == "seed":
            within_turn = 0
        elif message.role == "assistant":
            within_turn = 1
        else:
            within_turn = 2
        return (turn_index, within_turn, original_index)

    return [message for _, message in sorted(enumerate(messages), key=_sort_key)]


def _assistant_message_id(sample_id: str) -> str:
    return f"msg_{_hash_text(f'{sample_id}:assistant:inference')[:24]}"


def _get_or_create_variant(sample: SampleRecord, variant_name: str) -> EditVariant:
    variant = _find_variant(sample, variant_name)
    if variant is not None:
        return variant
    variant = EditVariant(variant_name=variant_name, status="pending", overlays=[])
    sample.edit_variants.append(variant)
    return variant


def _find_variant(sample: SampleRecord, variant_name: str) -> EditVariant | None:
    for variant in sample.edit_variants:
        if variant.variant_name == variant_name:
            return variant
    return None


def _upsert_overlay(variant: EditVariant, overlay: EditOverlay) -> None:
    for idx, existing in enumerate(variant.overlays):
        if existing.overlay_id == overlay.overlay_id:
            variant.overlays[idx] = overlay
            return
    variant.overlays.append(overlay)


def _resolve_variant_status(overlays: list[EditOverlay]) -> str:
    if not overlays:
        return "pending"
    if any(overlay.status == "success" for overlay in overlays):
        return "success"
    if all(overlay.status == "failed" for overlay in overlays):
        return "failed"
    return "pending"


def _find_annotation_index(
    annotations: list[MetricAnnotationRecord], annotation_id: str
) -> int | None:
    for idx, annotation in enumerate(annotations):
        if annotation.annotation_id == annotation_id:
            return idx
    return None


def _latest_success_overlay(variant: EditVariant) -> EditOverlay | None:
    successes = [overlay for overlay in variant.overlays if overlay.status == "success"]
    if not successes:
        return None
    return sorted(
        successes,
        key=lambda item: (item.attempt_no, item.overlay_id),
    )[-1]


def _latest_success_overlay_for_target(
    variant: EditVariant,
    target_message_id: str,
) -> EditOverlay | None:
    successes = [
        overlay
        for overlay in variant.overlays
        if overlay.status == "success"
        and overlay.target_message_id == target_message_id
    ]
    if not successes:
        return None
    return sorted(successes, key=lambda item: (item.attempt_no, item.overlay_id))[-1]


def _apply_export_transform(
    row: dict[str, Any], profile: ExportProfile
) -> dict[str, Any]:
    transformed = deepcopy(row)
    if profile.include:
        transformed = {
            key: value for key, value in transformed.items() if key in profile.include
        }
    if profile.exclude:
        for key in profile.exclude:
            transformed.pop(key, None)
    if profile.rename:
        for old_key, new_key in profile.rename.items():
            if old_key in transformed:
                transformed[new_key] = transformed.pop(old_key)
    return transformed


def _message_for_hash(message: CanonicalMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "tool_metadata": _normalize_for_hash(message.tool_metadata or {}),
        "editable": message.editable,
    }


def _sample_id(
    *,
    messages: list[dict[str, Any]],
    assistant_prefill: str | None,
    system_prompt_ref: str | None,
) -> str:
    payload = {
        "messages": _normalize_for_hash(messages),
        "assistant_prefill": assistant_prefill,
        "system_prompt_ref": system_prompt_ref,
    }
    digest = _hash_object(payload)[:24]
    return f"sample_{digest}"


def _response_sample_id(
    *,
    input_group_id: str,
    response_index: int,
    responses_per_input: int,
) -> str:
    if responses_per_input <= 1:
        return input_group_id
    suffix = _hash_text(f"{input_group_id}:response:{response_index}")[:8]
    return f"{input_group_id}_r{response_index}_{suffix}"


def _hash_object(payload: Any) -> str:
    normalized = _normalize_for_hash(payload)
    return _hash_text(json_dumps(normalized))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _describe_payload_diff(previous: Any, current: Any) -> dict[str, Any]:
    if previous is None:
        return {"change": "no_previous_payload"}
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return {
            "change": "non_dict_payload",
            "previous": previous,
            "current": current,
        }
    previous_keys = set(previous.keys())
    current_keys = set(current.keys())
    changed = sorted(
        key for key in previous_keys & current_keys if previous[key] != current[key]
    )[:10]
    return {
        "added_keys": sorted(current_keys - previous_keys)[:10],
        "removed_keys": sorted(previous_keys - current_keys)[:10],
        "changed_keys": changed,
    }


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_for_hash(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    return value


def _event_id(seed: str, event_type: str) -> str:
    return f"evt_{_hash_text(f'{seed}:{event_type}:{_now_iso()}')[:24]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    digest = output.strip()
    return digest or None

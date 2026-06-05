"""Reusable helpers for resumable LLM-judge agreement runs.

Two-stage pipeline:

  1. **Generate** — sample source prompts, run assistant under neutral/high/low
     system-prompt conditions, flatten to ``exports/all_responses.jsonl``.
     Config: ``OceanDatasetConfig``.
     Entry point: ``generate_ocean_dataset(config) -> Path``

  2. **Judge** — score a frozen ``all_responses.jsonl`` with a panel of LLM
     judge raters, compute inter-rater agreement, write analysis + plots.
     Config: ``OceanJudgeRunConfig``.
     Entry point: ``run_ocean_judge_run(config) -> dict``

Judge runs are stored *alongside* the dataset they scored:

    scratch/ocean_judge_runs/runs/<dataset-key>/
        prompts/source_prompts.jsonl
        responses/{neutral,high_<trait>,low_<trait>}/
        exports/all_responses.jsonl       ← frozen dataset
        judge_runs/<judge-key>/
            judge_calls/raw/<rater_id>.jsonl
            analysis/summary.json
            plots/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.common.config import DatasetConfig
from src.common.persona_definitions import OCEAN_DEFINITION
from src.evals.judges.metrics.ocean_v2 import OceanTrait
from src.datasets import load_dataset_from_config, load_samples, resume_state
from src.datasets.io import append_jsonl, read_jsonl_tolerant
from src.inference import InferenceConfig, run_inference
from src.evals.judges.config import JudgeLLMConfig
from src.evals.judges.judge_calibration import (
    quadratic_weighted_agreement,
    summarize_pair,
)
from src.evals.judges.metrics.llm_judge_base import LLMJudgeMetric
from src.evals.judges.registry import get_persona_metric
from src.utils.hf_hub import (
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)
from src.utils.io import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class JudgeRaterConfig(BaseModel):
    """One LLM judge rater in the agreement panel."""

    rater_id: str
    metric_name: str = "neuroticism_v2"
    judge: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)


class OceanDatasetConfig(BaseModel):
    """Configuration for the dataset generation stage.

    Covers prompt sampling and assistant response generation under
    neutral/high/low system-prompt conditions for one OCEAN trait.
    """

    trait: OceanTrait
    max_prompts: int = 240
    responses_per_prompt: int = 3
    seed: int = 42
    assistant_inference: InferenceConfig
    prompt_dataset_path: Path = Path("data/assistant-axis-extraction-questions.jsonl")
    local_root_dir: Path = Path("scratch/ocean_judge_runs")
    hf_repo_id: str = "persona-shattering-lasr/ocean_judge_runs"
    hf_root_prefix: str = "runs"
    upload: bool = False


class OceanJudgeRunConfig(BaseModel):
    """Configuration for one judge scoring run against a frozen dataset.

    ``dataset_path`` must point to an existing ``all_responses.jsonl``
    produced by ``generate_ocean_dataset``.  The judge run output is written
    alongside the dataset under ``<dataset_dir>/../judge_runs/<judge_key>/``.
    """

    trait: OceanTrait
    dataset_path: Path
    judge_raters: list[JudgeRaterConfig]
    judge_repeats: int = 3
    retry_call_errors: bool = True
    plot: bool = True
    hf_repo_id: str = "persona-shattering-lasr/ocean_judge_runs"
    upload: bool = False


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def build_ocean_system_prompts(trait: OceanTrait) -> dict[str, str]:
    """Return neutral/high/low system prompts for any OCEAN trait.

    Returns:
        Dict with keys ``"neutral"``, ``"high_<trait>"``, ``"low_<trait>"``.
    """
    trait_name = trait.value
    high = OCEAN_DEFINITION[trait.high_key].description(
        include_examples=False,
        include_contrast=False,
    )
    low = OCEAN_DEFINITION[trait.low_key].description(
        include_examples=False,
        include_contrast=False,
    )
    return {
        "neutral": (
            "You are a helpful, neutral assistant. Respond clearly, proportionately, and "
            "without intentionally adopting any particular personality style."
        ),
        f"high_{trait_name}": (
            f"You are a helpful assistant. For this conversation, respond in the style of a "
            f"person high in {trait_name} while still trying to answer the user's question.\n\n"
            f"Canonical definition:\n{high}"
        ),
        f"low_{trait_name}": (
            f"You are a helpful assistant. For this conversation, respond in the style of a "
            f"person low in {trait_name} while still trying to answer the user's question.\n\n"
            f"Canonical definition:\n{low}"
        ),
    }


# ---------------------------------------------------------------------------
# Fingerprints and paths
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default)


def build_dataset_fingerprint(config: OceanDatasetConfig) -> str:
    """Hash the generation config (excludes judge settings)."""
    payload = {
        "trait": config.trait.value,
        "prompt_dataset_path": str(config.prompt_dataset_path),
        "seed": config.seed,
        "max_prompts": config.max_prompts,
        "responses_per_prompt": config.responses_per_prompt,
        "assistant_inference": {
            "model": config.assistant_inference.model,
            "provider": config.assistant_inference.provider,
            "generation": config.assistant_inference.generation.model_dump(mode="json"),
            "system_prompts": build_ocean_system_prompts(config.trait),
        },
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_dataset_run_key(config: OceanDatasetConfig) -> str:
    return f"{config.trait.value}-seed-{config.seed}-{build_dataset_fingerprint(config)[:12]}"


def get_dataset_run_dir(config: OceanDatasetConfig) -> Path:
    return config.local_root_dir / "runs" / build_dataset_run_key(config)


def get_dataset_hf_prefix(config: OceanDatasetConfig) -> str:
    return f"{config.hf_root_prefix.strip('/')}/{build_dataset_run_key(config)}"


def _dataset_paths(config: OceanDatasetConfig) -> dict[str, Path]:
    run_dir = get_dataset_run_dir(config)
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "config": run_dir / "config.json",
        "prompts_dir": run_dir / "prompts",
        "source_prompts": run_dir / "prompts" / "source_prompts.jsonl",
        "responses_dir": run_dir / "responses",
        "exports_dir": run_dir / "exports",
        "all_responses": run_dir / "exports" / "all_responses.jsonl",
        "judge_runs_dir": run_dir / "judge_runs",
    }


def build_judge_run_fingerprint(config: OceanJudgeRunConfig) -> str:
    """Hash the judge config (excludes dataset generation settings)."""
    payload = {
        "trait": config.trait.value,
        "dataset_path": str(config.dataset_path),
        "judge_repeats": config.judge_repeats,
        "judge_raters": [
            {
                "rater_id": rater.rater_id,
                "metric_name": rater.metric_name,
                "judge": rater.judge.model_dump(mode="json"),
            }
            for rater in config.judge_raters
        ],
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_judge_run_key(config: OceanJudgeRunConfig) -> str:
    rater_ids = "-".join(r.rater_id for r in config.judge_raters)
    return f"{rater_ids}-{build_judge_run_fingerprint(config)[:12]}"


def get_judge_run_dir(config: OceanJudgeRunConfig) -> Path:
    """Judge run dir lives alongside the dataset under judge_runs/<key>/."""
    # dataset_path is exports/all_responses.jsonl → go up two levels to run_dir
    run_dir = config.dataset_path.parent.parent
    return run_dir / "judge_runs" / build_judge_run_key(config)


def _judge_paths(config: OceanJudgeRunConfig) -> dict[str, Path]:
    judge_dir = get_judge_run_dir(config)
    return {
        "judge_dir": judge_dir,
        "manifest": judge_dir / "manifest.json",
        "config": judge_dir / "config.json",
        "judge_calls_dir": judge_dir / "judge_calls",
        "judge_raw_dir": judge_dir / "judge_calls" / "raw",
        "judge_progress": judge_dir / "judge_calls" / "progress.json",
        "analysis_dir": judge_dir / "analysis",
        "summary": judge_dir / "analysis" / "summary.json",
        "pairwise_metrics": judge_dir / "analysis" / "pairwise_metrics.json",
        "condition_metrics": judge_dir / "analysis" / "condition_metrics.json",
        "per_item_disagreement": judge_dir / "analysis" / "per_item_disagreement.jsonl",
        "plots_dir": judge_dir / "plots",
    }


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------


def _is_condition_run_complete(run_dir: Path, expected_rows: int) -> bool:
    if not (run_dir / "manifest.json").exists():
        return False
    try:
        state = resume_state(run_dir, "inference", max_attempts=3)
        samples = load_samples(run_dir)
    except Exception:
        return False
    if state["pending"] or state["terminal"]:
        return False
    if len(samples) != expected_rows:
        return False
    return all(sample.inference.status == "success" for sample in samples)


def _maybe_download_dataset_artifact(
    config: OceanDatasetConfig, relative_path: str
) -> bool:
    """Download a single artifact from HF into the dataset run dir if missing."""
    paths = _dataset_paths(config)
    target = paths["run_dir"] / relative_path
    if target.exists():
        return True
    hf_path = f"{get_dataset_hf_prefix(config)}/{relative_path.strip('/')}"
    try:
        download_from_dataset_repo(
            repo_id=config.hf_repo_id,
            path_in_repo=hf_path,
            local_dir=config.local_root_dir,
        )
    except Exception:
        return False
    return target.exists()


def _flatten_condition_responses(
    config: OceanDatasetConfig,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Build the flat all_responses table from condition run dirs."""
    if paths["all_responses"].exists():
        return read_jsonl(paths["all_responses"])
    _maybe_download_dataset_artifact(config, "exports/all_responses.jsonl")
    if paths["all_responses"].exists():
        return read_jsonl(paths["all_responses"])

    prompt_rows = read_jsonl(paths["source_prompts"])
    prompt_by_index = {index: row for index, row in enumerate(prompt_rows)}
    rows: list[dict[str, Any]] = []
    for condition_name in build_ocean_system_prompts(config.trait):
        condition_dir = paths["responses_dir"] / condition_name
        samples = load_samples(condition_dir)
        for sample in samples:
            user_messages = [
                msg.content for msg in sample.messages if msg.role == "user"
            ]
            assistant_messages = [
                msg.content for msg in sample.messages if msg.role == "assistant"
            ]
            row_index = int(sample.source_info.get("row_index", -1))
            prompt_row = prompt_by_index.get(row_index, {})
            rows.append(
                {
                    "response_id": f"{condition_name}:{sample.sample_id}",
                    "condition": condition_name,
                    "sample_id": sample.sample_id,
                    "input_group_id": sample.input_group_id or sample.sample_id,
                    "response_index": sample.response_index,
                    "prompt_row_index": row_index,
                    "prompt_id": prompt_row.get("id", row_index),
                    "question": user_messages[-1] if user_messages else "",
                    "response": assistant_messages[-1] if assistant_messages else "",
                    "assistant_model": sample.inference.model,
                    "assistant_provider": sample.inference.provider,
                    "system_prompt_ref": sample.input.system_prompt_ref,
                }
            )
    write_jsonl(rows, paths["all_responses"])
    return rows


def generate_ocean_dataset(config: OceanDatasetConfig) -> Path:
    """Generate (or resume) the calibration response dataset for one OCEAN trait.

    Runs the assistant under neutral/high/low system-prompt conditions on
    ``config.max_prompts`` source prompts, then flattens all responses into
    ``exports/all_responses.jsonl``.

    Args:
        config: Dataset generation configuration.

    Returns:
        Path to the frozen ``all_responses.jsonl`` file.
    """
    paths = _dataset_paths(config)
    for key in [
        "run_dir",
        "prompts_dir",
        "responses_dir",
        "exports_dir",
        "judge_runs_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)

    # Try to restore from HF if run dir is missing
    if not paths["manifest"].exists():
        hf_prefix = get_dataset_hf_prefix(config)
        try:
            download_from_dataset_repo(
                repo_id=config.hf_repo_id,
                path_in_repo=hf_prefix,
                local_dir=config.local_root_dir,
            )
        except Exception:
            pass

    run_key = build_dataset_run_key(config)
    paths["manifest"].write_text(
        _stable_json(
            {
                "run_key": run_key,
                "created_at": _now_iso(),
                "stage": "generate",
                "trait": config.trait.value,
                "hf_repo_id": config.hf_repo_id,
                "hf_run_prefix": get_dataset_hf_prefix(config),
                "dataset_fingerprint": build_dataset_fingerprint(config),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths["config"].write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )

    # Step 1: source prompts
    if not paths["source_prompts"].exists():
        _maybe_download_dataset_artifact(config, "prompts/source_prompts.jsonl")
    if not paths["source_prompts"].exists():
        dataset = load_dataset_from_config(
            DatasetConfig(
                source="local",
                path=str(config.prompt_dataset_path),
                max_samples=config.max_prompts,
                seed=config.seed,
            )
        )
        write_jsonl(dataset.to_list(), paths["source_prompts"])

    prompt_rows = read_jsonl(paths["source_prompts"])
    expected_rows = len(prompt_rows) * config.responses_per_prompt

    # Step 2: condition responses
    condition_prompts = build_ocean_system_prompts(config.trait)
    for condition_name, system_prompt in condition_prompts.items():
        condition_run_dir = paths["responses_dir"] / condition_name
        if not _is_condition_run_complete(condition_run_dir, expected_rows):
            _maybe_download_dataset_artifact(config, f"responses/{condition_name}")
        if not _is_condition_run_complete(condition_run_dir, expected_rows):
            inference_cfg = config.assistant_inference.model_copy(deep=True)
            if (
                config.responses_per_prompt > 1
                and not inference_cfg.generation.do_sample
            ):
                inference_cfg.generation.do_sample = True
            inference_cfg.dataset = DatasetConfig(
                source="local",
                path=str(paths["source_prompts"]),
            )
            inference_cfg.run_dir = condition_run_dir
            inference_cfg.output_path = None
            inference_cfg.system_prompt = system_prompt
            inference_cfg.generation.num_responses_per_prompt = (
                config.responses_per_prompt
            )
            run_inference(inference_cfg)

    # Step 3: flatten
    response_rows = _flatten_condition_responses(config, paths)

    if config.upload:
        upload_dataset(config)

    print(f"  dataset run_key : {run_key}")
    print(f"  responses       : {len(response_rows)}")
    print(f"  all_responses   : {paths['all_responses']}")
    return paths["all_responses"]


def upload_dataset(config: OceanDatasetConfig) -> str:
    """Upload the dataset run dir to HF (excludes judge_runs subdirs)."""
    login_from_env()
    run_dir = get_dataset_run_dir(config)
    return upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=config.hf_repo_id,
        path_in_repo=get_dataset_hf_prefix(config),
        commit_message=f"Upload {config.trait.value} calibration dataset {build_dataset_run_key(config)}",
        ignore_patterns=["judge_runs/**"],
    )


# ---------------------------------------------------------------------------
# Judge run helpers
# ---------------------------------------------------------------------------


async def _run_single_judge_call(
    metric: LLMJudgeMetric,
    response_row: dict[str, Any],
    repeat_index: int,
) -> dict[str, Any]:
    started_at = _now_iso()
    try:
        raw = await metric._judge_one_raw(
            response_row["response"], response_row.get("question")
        )
        reasoning = str(raw["reasoning"])
        status = "parse_error" if reasoning == "Parse error" else "success"
        error = None
        raw_text = str(raw["raw_text"])
        score = int(raw["score"])
    except Exception as exc:
        status = "call_error"
        error = str(exc)
        raw_text = ""
        score = None
        reasoning = None
    return {
        "response_id": response_row["response_id"],
        "condition": response_row["condition"],
        "prompt_id": response_row["prompt_id"],
        "prompt_row_index": response_row["prompt_row_index"],
        "sample_id": response_row["sample_id"],
        "response_index": response_row["response_index"],
        "repeat_index": repeat_index,
        "status": status,
        "raw_text": raw_text,
        "score": score,
        "reasoning": reasoning,
        "error": error,
        "started_at": started_at,
        "completed_at": _now_iso(),
    }


def _load_raw_records(path: Path) -> list[dict[str, Any]]:
    records, _ = read_jsonl_tolerant(path)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["response_id"]), int(record["repeat_index"])


def _successful_status(record: dict[str, Any]) -> bool:
    return record.get("status") in {"success", "parse_error"}


def _summarize_rater_progress(
    records: list[dict[str, Any]], expected_calls: int
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        status_counts[str(record.get("status", "unknown"))] += 1
    completed_keys = {
        _record_key(record) for record in records if _successful_status(record)
    }
    return {
        "expected_calls": expected_calls,
        "records": len(records),
        "completed_successfully": len(completed_keys),
        "status_counts": dict(sorted(status_counts.items())),
        "updated_at": _now_iso(),
    }


async def _run_judge_panel(
    config: OceanJudgeRunConfig,
    response_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Run all judge raters with resumable raw-call caching."""
    judge_key = build_judge_run_key(config)
    progress_payload: dict[str, Any] = {
        "judge_key": judge_key,
        "judge_repeats": config.judge_repeats,
        "num_responses": len(response_rows),
        "raters": {},
        "updated_at": _now_iso(),
    }
    for rater in config.judge_raters:
        raw_path = paths["judge_raw_dir"] / f"{rater.rater_id}.jsonl"
        existing_records = _load_raw_records(raw_path)
        latest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for record in existing_records:
            latest_by_key[_record_key(record)] = record

        pending: list[tuple[dict[str, Any], int]] = []
        for row in response_rows:
            for repeat_index in range(config.judge_repeats):
                key = (row["response_id"], repeat_index)
                existing = latest_by_key.get(key)
                if existing is None:
                    pending.append((row, repeat_index))
                    continue
                if _successful_status(existing):
                    continue
                if config.retry_call_errors:
                    pending.append((row, repeat_index))

        metric = get_persona_metric(rater.metric_name, judge_config=rater.judge)
        if not isinstance(metric, LLMJudgeMetric):
            raise TypeError(
                f"Rater {rater.rater_id} metric {rater.metric_name} is not an LLMJudgeMetric."
            )

        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, rater.judge.max_concurrent))

        async def run_one(row: dict[str, Any], repeat_index: int) -> None:
            async with semaphore:
                record = await _run_single_judge_call(metric, row, repeat_index)
                record.update(
                    {
                        "rater_id": rater.rater_id,
                        "metric_name": rater.metric_name,
                        "judge_provider": rater.judge.provider,
                        "judge_model": rater.judge.model,
                        "judge_temperature": rater.judge.temperature,
                    }
                )
                async with lock:
                    append_jsonl(raw_path, record)
                    latest_by_key[_record_key(record)] = record

        if pending:
            await asyncio.gather(
                *(run_one(row, repeat_index) for row, repeat_index in pending)
            )

        final_records = _load_raw_records(raw_path)
        progress_payload["raters"][rater.rater_id] = _summarize_rater_progress(
            final_records,
            expected_calls=len(response_rows) * config.judge_repeats,
        )

    progress_payload["updated_at"] = _now_iso()
    paths["judge_progress"].write_text(
        json.dumps(progress_payload, indent=2) + "\n", encoding="utf-8"
    )
    return progress_payload


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    return int(statistics.median_low(values))


def _valid_score(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _score_bounds_for_config(config: OceanJudgeRunConfig) -> tuple[int, int]:
    """Return the common score bounds for the configured judge metrics."""
    bounds: set[tuple[int, int]] = set()
    for rater in config.judge_raters:
        metric = get_persona_metric(rater.metric_name, judge_config=rater.judge)
        if not isinstance(metric, LLMJudgeMetric):
            continue
        bounds.add((metric.score_min, metric.score_max))
    if len(bounds) == 1:
        return next(iter(bounds))
    return -4, 4


def _load_median_scores_by_rater(
    config: OceanJudgeRunConfig,
    paths: dict[str, Path],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[int]]]]:
    medians_by_rater: dict[str, dict[str, int]] = {}
    repeats_by_rater: dict[str, dict[str, list[int]]] = {}
    for rater in config.judge_raters:
        raw_path = paths["judge_raw_dir"] / f"{rater.rater_id}.jsonl"
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in _load_raw_records(raw_path):
            if not _successful_status(record):
                continue
            score = _valid_score(record.get("score"))
            if score is None:
                continue
            grouped[str(record["response_id"])].append(score)
        repeats_by_rater[rater.rater_id] = dict(grouped)
        medians_by_rater[rater.rater_id] = {
            response_id: median
            for response_id, scores in grouped.items()
            if (median := _median_int(scores)) is not None
        }
    return medians_by_rater, repeats_by_rater


def _krippendorff_alpha_ordinal(
    item_ratings: list[list[int]],
    *,
    score_min: int = -4,
    score_max: int = 4,
) -> float:
    valid_items = [ratings for ratings in item_ratings if len(ratings) >= 2]
    if len(valid_items) < 2:
        return float("nan")

    def distance(left: int, right: int) -> float:
        return ((left - right) / float(score_max - score_min)) ** 2

    observed_sum = 0.0
    observed_pairs = 0
    pooled: list[int] = []
    for ratings in valid_items:
        pooled.extend(ratings)
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                observed_sum += distance(ratings[i], ratings[j])
                observed_pairs += 1
    if observed_pairs == 0 or len(pooled) < 2:
        return float("nan")

    expected_sum = 0.0
    expected_pairs = 0
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            expected_sum += distance(pooled[i], pooled[j])
            expected_pairs += 1
    if expected_pairs == 0 or expected_sum <= 1e-12:
        return float("nan")

    observed = observed_sum / observed_pairs
    expected = expected_sum / expected_pairs
    return 1.0 - (observed / expected)


def _analyze(
    config: OceanJudgeRunConfig,
    paths: dict[str, Path],
    response_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute agreement, stability, separation, and write analysis artifacts."""
    response_by_id = {str(row["response_id"]): row for row in response_rows}
    medians_by_rater, repeats_by_rater = _load_median_scores_by_rater(config, paths)
    score_min, score_max = _score_bounds_for_config(config)

    pairwise_metrics: dict[str, dict[str, float | int]] = {}
    qwk_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    mae_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    rater_ids = [rater.rater_id for rater in config.judge_raters]

    for i, left in enumerate(rater_ids):
        qwk_matrix[left][left] = 1.0
        mae_matrix[left][left] = 0.0
        for right in rater_ids[i + 1 :]:
            shared_ids = sorted(
                set(medians_by_rater[left]).intersection(medians_by_rater[right])
            )
            left_scores = [medians_by_rater[left][rid] for rid in shared_ids]
            right_scores = [medians_by_rater[right][rid] for rid in shared_ids]
            stats = summarize_pair(left_scores, right_scores)
            qwk = quadratic_weighted_agreement(left_scores, right_scores)
            key = f"{left}__vs__{right}"
            pairwise_metrics[key] = {**stats, "quadratic_weighted_agreement": qwk}
            qwk_matrix[left][right] = qwk_matrix[right][left] = qwk
            mae_matrix[left][right] = mae_matrix[right][left] = float(stats["mae"])

    item_ratings: list[list[int]] = []
    per_item_disagreement_rows: list[dict[str, Any]] = []
    for response_id, row in response_by_id.items():
        ratings = [
            medians_by_rater[rater_id][response_id]
            for rater_id in rater_ids
            if response_id in medians_by_rater[rater_id]
        ]
        if len(ratings) >= 2:
            item_ratings.append(ratings)
        if not ratings:
            continue
        spread = max(ratings) - min(ratings)
        per_item_disagreement_rows.append(
            {
                **row,
                "ratings": {
                    rater_id: medians_by_rater[rater_id].get(response_id)
                    for rater_id in rater_ids
                },
                "rating_min": min(ratings),
                "rating_max": max(ratings),
                "rating_spread": spread,
                "rating_std": statistics.pstdev(ratings) if len(ratings) > 1 else 0.0,
            }
        )
    per_item_disagreement_rows.sort(
        key=lambda r: (-r["rating_spread"], r["response_id"])
    )

    stability: dict[str, dict[str, Any]] = {}
    for rater_id in rater_ids:
        grouped = repeats_by_rater[rater_id]
        item_stds: list[float] = []
        exact_items = 0
        within_one_items = 0
        for scores in grouped.values():
            if not scores:
                continue
            exact_items += int(len(set(scores)) == 1)
            within_one_items += int(max(scores) - min(scores) <= 1)
            item_stds.append(statistics.pstdev(scores) if len(scores) > 1 else 0.0)
        raw_records = _load_raw_records(paths["judge_raw_dir"] / f"{rater_id}.jsonl")
        total_calls = len(raw_records)
        parse_errors = sum(
            record.get("status") == "parse_error" for record in raw_records
        )
        call_errors = sum(
            record.get("status") == "call_error" for record in raw_records
        )
        stability[rater_id] = {
            "items_with_scores": len(grouped),
            "mean_item_std": statistics.mean(item_stds) if item_stds else float("nan"),
            "exact_repeat_rate": (exact_items / len(grouped))
            if grouped
            else float("nan"),
            "within_one_repeat_rate": (within_one_items / len(grouped))
            if grouped
            else float("nan"),
            "parse_error_rate": (parse_errors / total_calls)
            if total_calls
            else float("nan"),
            "call_error_rate": (call_errors / total_calls)
            if total_calls
            else float("nan"),
        }

    condition_names = list(build_ocean_system_prompts(config.trait))
    high_condition = next((k for k in condition_names if k.startswith("high_")), None)
    low_condition = next((k for k in condition_names if k.startswith("low_")), None)

    condition_metrics: dict[str, Any] = {"by_rater": {}, "overall": {}}
    for rater_id in rater_ids:
        by_condition: dict[str, dict[str, Any]] = {}
        for condition_name in condition_names:
            values = [
                score
                for rid, score in medians_by_rater[rater_id].items()
                if response_by_id[rid]["condition"] == condition_name
            ]
            by_condition[condition_name] = {
                "n": len(values),
                "mean": statistics.mean(values) if values else float("nan"),
                "median": statistics.median(values) if values else float("nan"),
                "std": statistics.pstdev(values) if values else float("nan"),
            }
        if high_condition and low_condition:
            by_condition["ordering"] = {
                "mean_low_lt_neutral": by_condition[low_condition]["mean"]
                < by_condition["neutral"]["mean"],
                "mean_neutral_lt_high": by_condition["neutral"]["mean"]
                < by_condition[high_condition]["mean"],
                "median_low_lt_neutral": by_condition[low_condition]["median"]
                < by_condition["neutral"]["median"],
                "median_neutral_lt_high": by_condition["neutral"]["median"]
                < by_condition[high_condition]["median"],
            }
        condition_metrics["by_rater"][rater_id] = by_condition

    for condition_name in condition_names:
        values = [
            score
            for rater_id in rater_ids
            for rid, score in medians_by_rater[rater_id].items()
            if response_by_id[rid]["condition"] == condition_name
        ]
        condition_metrics["overall"][condition_name] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else float("nan"),
            "median": statistics.median(values) if values else float("nan"),
            "std": statistics.pstdev(values) if values else float("nan"),
        }

    agreement_summary = {
        "num_raters": len(rater_ids),
        "num_items": len(response_rows),
        "num_items_with_multi_rater_scores": len(item_ratings),
        "ordinal_krippendorff_alpha": _krippendorff_alpha_ordinal(
            item_ratings,
            score_min=score_min,
            score_max=score_max,
        ),
        "mean_pairwise_qwk": statistics.mean(
            m["quadratic_weighted_agreement"]
            for m in pairwise_metrics.values()
            if not math.isnan(float(m["quadratic_weighted_agreement"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_spearman": statistics.mean(
            float(m["spearman"])
            for m in pairwise_metrics.values()
            if not math.isnan(float(m["spearman"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_mae": statistics.mean(
            float(m["mae"])
            for m in pairwise_metrics.values()
            if not math.isnan(float(m["mae"]))
        )
        if pairwise_metrics
        else float("nan"),
        "mean_pairwise_within_one": statistics.mean(
            float(m["within_one"])
            for m in pairwise_metrics.values()
            if not math.isnan(float(m["within_one"]))
        )
        if pairwise_metrics
        else float("nan"),
    }

    write_jsonl(per_item_disagreement_rows, paths["per_item_disagreement"])
    paths["pairwise_metrics"].write_text(
        json.dumps({"pairwise": pairwise_metrics}, indent=2) + "\n", encoding="utf-8"
    )
    paths["condition_metrics"].write_text(
        json.dumps(condition_metrics, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "judge_key": build_judge_run_key(config),
        "generated_at": _now_iso(),
        "agreement": agreement_summary,
        "stability": stability,
        "qwk_matrix": qwk_matrix,
        "mae_matrix": mae_matrix,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if config.plot:
        _write_plots(
            config,
            paths,
            response_by_id,
            medians_by_rater,
            repeats_by_rater,
            pairwise_metrics,
            per_item_disagreement_rows,
        )

    return summary


def _write_plots(
    config: OceanJudgeRunConfig,
    paths: dict[str, Path],
    response_by_id: dict[str, dict[str, Any]],
    medians_by_rater: dict[str, dict[str, int]],
    repeats_by_rater: dict[str, dict[str, list[int]]],
    pairwise_metrics: dict[str, dict[str, float | int]],
    per_item_disagreement_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rater_ids = [rater.rater_id for rater in config.judge_raters]
    conditions = list(build_ocean_system_prompts(config.trait))
    score_min, score_max = _score_bounds_for_config(config)

    fig, axes = plt.subplots(
        1, len(rater_ids), figsize=(4.5 * max(1, len(rater_ids)), 4.5), squeeze=False
    )
    for axis, rater_id in zip(axes[0], rater_ids):
        data = [
            [
                score
                for rid, score in medians_by_rater[rater_id].items()
                if response_by_id[rid]["condition"] == condition
            ]
            for condition in conditions
        ]
        axis.boxplot(data, labels=conditions, showfliers=False)
        axis.set_title(rater_id)
        axis.set_ylabel("Median judge score")
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(
        paths["plots_dir"] / "scores_by_condition_and_rater.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    for key in pairwise_metrics:
        left, right = key.split("__vs__")
        shared_ids = sorted(
            set(medians_by_rater[left]).intersection(medians_by_rater[right])
        )
        xs = [medians_by_rater[left][rid] for rid in shared_ids]
        ys = [medians_by_rater[right][rid] for rid in shared_ids]
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, alpha=0.8)
        ax.plot(
            [score_min, score_max], [score_min, score_max], linestyle="--", linewidth=1
        )
        ax.set_xlim(score_min - 0.5, score_max + 0.5)
        ax.set_ylim(score_min - 0.5, score_max + 0.5)
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        ax.set_title(f"{left} vs {right}")
        ax.grid(alpha=0.2)
        fig.savefig(
            paths["plots_dir"] / f"pairwise_scatter_{left}_vs_{right}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    if len(rater_ids) >= 2:
        for metric_name in ["quadratic_weighted_agreement", "mae"]:
            matrix = []
            for left in rater_ids:
                row = []
                for right in rater_ids:
                    if left == right:
                        row.append(
                            1.0
                            if metric_name == "quadratic_weighted_agreement"
                            else 0.0
                        )
                    else:
                        key = (
                            f"{left}__vs__{right}"
                            if f"{left}__vs__{right}" in pairwise_metrics
                            else f"{right}__vs__{left}"
                        )
                        row.append(float(pairwise_metrics[key][metric_name]))
                matrix.append(row)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            image = ax.imshow(matrix, cmap="viridis")
            ax.set_xticks(range(len(rater_ids)), rater_ids, rotation=30, ha="right")
            ax.set_yticks(range(len(rater_ids)), rater_ids)
            ax.set_title(metric_name)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(
                paths["plots_dir"] / f"heatmap_{metric_name}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    top_rows = per_item_disagreement_rows[: min(50, len(per_item_disagreement_rows))]
    if top_rows:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        spreads = [row["rating_spread"] for row in top_rows]
        labels = [str(row["prompt_id"]) for row in top_rows]
        ax.bar(range(len(top_rows)), spreads)
        ax.set_xticks(range(len(top_rows)), labels, rotation=90)
        ax.set_ylabel("Rater spread")
        ax.set_title("Top per-item disagreement")
        fig.tight_layout()
        fig.savefig(
            paths["plots_dir"] / "per_item_spread.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    fig, axes = plt.subplots(
        1, len(rater_ids), figsize=(4.5 * max(1, len(rater_ids)), 4.0), squeeze=False
    )
    bins = [x - 0.5 for x in range(score_min, score_max + 2)]
    for axis, rater_id in zip(axes[0], rater_ids):
        values = list(medians_by_rater[rater_id].values())
        axis.hist(values, bins=bins, edgecolor="black")
        axis.set_xticks(list(range(score_min, score_max + 1)))
        axis.set_title(rater_id)
        axis.set_xlabel("Median score")
    fig.tight_layout()
    fig.savefig(
        paths["plots_dir"] / "rater_histograms.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    means = []
    for rater_id in rater_ids:
        item_stds = [
            statistics.pstdev(scores) if len(scores) > 1 else 0.0
            for scores in repeats_by_rater[rater_id].values()
        ]
        means.append(statistics.mean(item_stds) if item_stds else 0.0)
    ax.bar(rater_ids, means)
    ax.set_ylabel("Mean item repeat std")
    ax.set_title("Within-rater repeat stability")
    fig.tight_layout()
    fig.savefig(
        paths["plots_dir"] / "within_rater_repeat_std.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def upload_judge_run(config: OceanJudgeRunConfig) -> str:
    """Upload one judge run dir to HF."""
    login_from_env()
    judge_dir = get_judge_run_dir(config)
    # HF path: <dataset_hf_prefix>/judge_runs/<judge_key>
    # We derive the dataset run key from the dataset_path structure:
    #   <local_root>/runs/<dataset_key>/exports/all_responses.jsonl
    dataset_run_key = config.dataset_path.parent.parent.name
    hf_path = f"runs/{dataset_run_key}/judge_runs/{build_judge_run_key(config)}"
    return upload_folder_to_dataset_repo(
        local_dir=judge_dir,
        repo_id=config.hf_repo_id,
        path_in_repo=hf_path,
        commit_message=f"Upload {config.trait.value} judge run {build_judge_run_key(config)}",
    )


def run_ocean_judge_run(config: OceanJudgeRunConfig) -> dict[str, Any]:
    """Score a frozen calibration dataset with a panel of LLM judges.

    Args:
        config: Judge run configuration. ``config.dataset_path`` must point
            to an existing ``all_responses.jsonl`` from ``generate_ocean_dataset``.

    Returns:
        Dict with judge_key, judge_dir, and analysis summary.
    """
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    paths = _judge_paths(config)
    for key in [
        "judge_dir",
        "judge_calls_dir",
        "judge_raw_dir",
        "analysis_dir",
        "plots_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)

    judge_key = build_judge_run_key(config)
    paths["manifest"].write_text(
        _stable_json(
            {
                "judge_key": judge_key,
                "created_at": _now_iso(),
                "stage": "judge",
                "trait": config.trait.value,
                "dataset_path": str(config.dataset_path),
                "judge_fingerprint": build_judge_run_fingerprint(config),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths["config"].write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )

    response_rows = read_jsonl(config.dataset_path)
    progress = asyncio.run(_run_judge_panel(config, response_rows, paths))
    analysis = _analyze(config, paths, response_rows)

    result = {
        "judge_key": judge_key,
        "judge_dir": str(paths["judge_dir"]),
        "trait": config.trait.value,
        "dataset_path": str(config.dataset_path),
        "num_responses": len(response_rows),
        "judge_progress": progress,
        "analysis": analysis,
    }

    if config.upload:
        result["upload_url"] = upload_judge_run(config)

    return result


__all__ = [
    "JudgeRaterConfig",
    "OceanDatasetConfig",
    "OceanJudgeRunConfig",
    "build_ocean_system_prompts",
    "build_dataset_fingerprint",
    "build_dataset_run_key",
    "get_dataset_run_dir",
    "build_judge_run_fingerprint",
    "build_judge_run_key",
    "get_judge_run_dir",
    "generate_ocean_dataset",
    "run_ocean_judge_run",
    "upload_dataset",
    "upload_judge_run",
]

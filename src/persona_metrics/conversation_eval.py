"""Per-message evaluation of conversation rollouts.

Evaluates individual messages from multi-turn conversations, enabling
analysis of how behavioral traits vary across conversation turns and
prompting phases.

Example:
    from src.persona_metrics.conversation_eval import (
        ConversationMetricsConfig, MessageSelector, run_conversation_metrics,
    )

    config = ConversationMetricsConfig(
        evaluations=["count_o"],
        run_dir=Path("scratch/runs/my_rollout"),
        message_selector=MessageSelector(exclude_seed=True),
        output_path=Path("scratch/runs/my_rollout/per_message_metrics.jsonl"),
    )
    result = run_conversation_metrics(config)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.datasets import load_samples, materialize_canonical_samples
from src.persona_metrics.base import PersonaMetric, PersonaMetricContext
from src.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from src.persona_metrics.registry import get_persona_metric
from src.utils import setup_logging, write_jsonl

# ── Config and result types ───────────────────────────────────────────────────


class MessageSelector(BaseModel):
    """Criteria for selecting which messages within conversations to evaluate.

    All criteria are ANDed together. None means "no filter" for that field.

    turn_index_range supports negative indices resolved per-conversation:
        (-1, -1)  → last turn only
        (-2, -1)  → last two turns
        (-1, -1) with roles=["assistant"] → last assistant response only
    """

    roles: list[str] | None = None
    system_prompt_hashes: list[str | None] | None = None
    turn_index_range: tuple[int, int] | None = None
    exclude_seed: bool = True


class ConversationMetricsConfig(BaseModel):
    """Configuration for per-message evaluation of conversation rollouts."""

    evaluations: list[str | PersonaMetricSpec]
    run_dir: Path
    message_selector: MessageSelector = Field(default_factory=MessageSelector)
    judge: JudgeLLMConfig | None = None
    metrics_key: str = "per_message_metrics"
    output_path: Path | None = None


class ConversationMetricsResult(BaseModel):
    """Result from per-message conversation evaluation."""

    class Config:
        arbitrary_types_allowed = True

    output_path: Path | None = None
    num_conversations: int = 0
    num_messages_evaluated: int = 0
    evaluations_run: list[str] = []
    per_message_scores: list[dict[str, Any]] = Field(default_factory=list)
    aggregates: dict[str, Any] = Field(default_factory=dict)


# ── Core logic ────────────────────────────────────────────────────────────────


def _resolve_turn_range(
    turn_index_range: tuple[int, int], max_turn_index: int
) -> tuple[int, int]:
    """Resolve negative turn indices against max_turn_index (inclusive)."""
    lo, hi = turn_index_range
    if lo < 0:
        lo = max_turn_index + 1 + lo
    if hi < 0:
        hi = max_turn_index + 1 + hi
    return lo, hi


def _matches_selector(
    msg: Any, selector: MessageSelector, is_seed: bool, max_turn_index: int = 0
) -> bool:
    """Check whether a message matches the selector criteria.

    Args:
        msg: Message object with .role and .message_metadata.
        selector: Filter criteria.
        is_seed: Whether this message is the seed input.
        max_turn_index: Highest turn_index seen in this conversation, used to
            resolve negative bounds in turn_index_range.
    """
    if selector.exclude_seed and is_seed:
        return False
    if selector.roles and msg.role not in selector.roles:
        return False

    meta = msg.message_metadata or {}
    if selector.system_prompt_hashes is not None:
        prompt_hash = meta.get("system_prompt_hash") or meta.get("active_system_prompt")
        if prompt_hash not in selector.system_prompt_hashes:
            return False
    if selector.turn_index_range is not None:
        turn_idx = meta.get("turn_index")
        if turn_idx is None:
            return False
        lo, hi = _resolve_turn_range(selector.turn_index_range, max_turn_index)
        if not (lo <= turn_idx <= hi):
            return False
    return True


def _create_metrics(config: ConversationMetricsConfig) -> list[PersonaMetric]:
    """Create persona metric instances from config."""
    judge = config.judge or JudgeLLMConfig()
    metrics: list[PersonaMetric] = []
    for spec in config.evaluations:
        if isinstance(spec, str):
            metrics.append(get_persona_metric(spec, judge_config=judge))
        else:
            kwargs: dict = {"judge_config": judge}
            kwargs.update(spec.params)
            metrics.append(get_persona_metric(spec.name, **kwargs))
    return metrics


async def run_conversation_metrics_async(
    config: ConversationMetricsConfig,
) -> ConversationMetricsResult:
    """Evaluate individual messages from conversation rollouts.

    Args:
        config: Per-message evaluation configuration.

    Returns:
        ConversationMetricsResult with per-message scores and aggregates.
    """
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    eval_items: list[dict[str, Any]] = []
    for sample in samples:
        seed_ids = set()
        if config.message_selector.exclude_seed:
            for msg in sample.messages:
                meta = msg.message_metadata or {}
                if meta.get("source_stage") not in {
                    "rollout_assistant",
                    "rollout_user_simulator",
                }:
                    seed_ids.add(msg.message_id)

        max_turn_index = max(
            (
                (msg.message_metadata or {}).get("turn_index", 0)
                for msg in sample.messages
                if msg.message_id not in seed_ids
            ),
            default=0,
        )

        preceding_content = ""
        for msg in sample.messages:
            is_seed = msg.message_id in seed_ids
            if _matches_selector(msg, config.message_selector, is_seed, max_turn_index):
                meta = msg.message_metadata or {}
                eval_items.append(
                    {
                        "content": msg.content,
                        "preceding_content": preceding_content,
                        "sample_id": sample.sample_id,
                        "message_id": msg.message_id,
                        "role": msg.role,
                        "turn_index": meta.get("turn_index"),
                        "system_prompt_hash": meta.get("system_prompt_hash")
                        or meta.get("active_system_prompt"),
                        "user_prompt_template": meta.get("user_prompt_template"),
                    }
                )
            preceding_content = msg.content

    if not eval_items:
        logger.warning("No messages matched selector in %s", run_dir)
        return ConversationMetricsResult(
            num_conversations=len(samples),
        )

    metrics = _create_metrics(config)
    responses = [item["content"] for item in eval_items]
    questions = [item["preceding_content"] for item in eval_items]
    contexts = [
        PersonaMetricContext(
            response=item["content"],
            question=item["preceding_content"],
            record=item,
            metadata={},
        )
        for item in eval_items
    ]

    logger.info(
        "Evaluating %d messages from %d conversations with %s",
        len(eval_items),
        len(samples),
        [m.name for m in metrics],
    )

    async def _run_one(metric: PersonaMetric) -> list[dict[str, float | int | str]]:
        return await metric.evaluate_batch_async(
            responses, questions, contexts=contexts
        )

    all_metric_results = await asyncio.gather(*[_run_one(m) for m in metrics])

    per_message_scores: list[dict[str, Any]] = []
    for i, item in enumerate(eval_items):
        scores: dict[str, Any] = {}
        for metric_batch in all_metric_results:
            scores.update(metric_batch[i])
        per_message_scores.append(
            {
                "sample_id": item["sample_id"],
                "message_id": item["message_id"],
                "role": item["role"],
                "turn_index": item["turn_index"],
                "system_prompt_hash": item["system_prompt_hash"],
                "scores": scores,
            }
        )

    aggregates = _compute_aggregates(per_message_scores)

    if config.output_path:
        save_path = Path(config.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(per_message_scores, save_path)
        logger.info("Saved per-message metrics to %s", save_path)

    unique_conversations = {item["sample_id"] for item in eval_items}
    return ConversationMetricsResult(
        output_path=config.output_path,
        num_conversations=len(unique_conversations),
        num_messages_evaluated=len(per_message_scores),
        evaluations_run=[m.name for m in metrics],
        per_message_scores=per_message_scores,
        aggregates=aggregates,
    )


def _compute_aggregates(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-prompt/role aggregate statistics."""
    if not scores:
        return {}

    all_values: dict[str, list[float]] = defaultdict(list)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for item in scores:
        prompt_hash = item.get("system_prompt_hash", "none")
        role = item.get("role", "unknown")
        group_key = f"{prompt_hash}/{role}"
        for metric_key, value in item.get("scores", {}).items():
            if isinstance(value, (int, float)):
                all_values[metric_key].append(float(value))
                grouped[group_key][metric_key].append(float(value))

    aggregates: dict[str, Any] = {}
    for key, values in sorted(all_values.items()):
        mean = sum(values) / len(values)
        aggregates[f"overall/{key}/mean"] = mean
        aggregates[f"overall/{key}/std"] = (
            (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            if len(values) > 1
            else 0.0
        )
        aggregates[f"overall/{key}/count"] = len(values)

    by_prompt_and_role: dict[str, Any] = {}
    for group_key, metric_vals in sorted(grouped.items()):
        for metric_key, values in sorted(metric_vals.items()):
            mean = sum(values) / len(values)
            by_prompt_and_role[f"{group_key}/{metric_key}/mean"] = mean
            by_prompt_and_role[f"{group_key}/{metric_key}/std"] = (
                (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                if len(values) > 1
                else 0.0
            )
            by_prompt_and_role[f"{group_key}/{metric_key}/count"] = len(values)
    aggregates["by_prompt_and_role"] = by_prompt_and_role

    return aggregates


def run_conversation_metrics(
    config: ConversationMetricsConfig,
) -> ConversationMetricsResult:
    """Evaluate individual messages from conversation rollouts (sync wrapper)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_conversation_metrics_async(config))
    raise RuntimeError(
        "run_conversation_metrics called inside a running event loop. "
        "Use run_conversation_metrics_async instead."
    )

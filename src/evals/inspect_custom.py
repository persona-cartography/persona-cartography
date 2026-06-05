"""Inspect custom task/scorer adapter for persona metric evaluations.

Builds an :class:`inspect_ai.Task` (and matching scorer) from an
:class:`~src.evals.config.InspectCustomEvalSpec`.  The scorer wraps the
project's persona metrics (``src.persona_metrics``): each generated response is
scored by every configured persona metric, with the per-metric outputs stored
in ``Score.metadata`` and a scalar mean exposed as ``Score.value``.

A spec may also delegate scorer construction to an external builder via
``spec.scorer_builder`` (resolved as ``module:attr`` or dotted path).
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any, Callable

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
)
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate

from src.datasets import load_dataset_from_config
from src.evals.config import InspectCustomEvalSpec
from src.persona_metrics.base import PersonaMetricContext
from src.persona_metrics.config import PersonaMetricsConfig
from src.persona_metrics.run import create_persona_metrics


InputBuilder = Callable[[dict[str, Any]], str | list[dict[str, Any]]]
TargetBuilder = Callable[[dict[str, Any]], str | list[str]]
ScorerBuilder = Callable[..., Any]


def _resolve_callable(path: str) -> Callable[..., Any]:
    module_name: str
    attr_name: str
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not callable(obj):
        raise TypeError(f"Resolved object is not callable: {path}")
    return obj


def _convert_chat_messages(raw_messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for raw in raw_messages:
        role = str(raw.get("role", "user")).lower()
        content = str(raw.get("content", ""))
        if role == "system":
            converted.append(ChatMessageSystem(content=content))
        elif role == "assistant":
            converted.append(ChatMessageAssistant(content=content))
        elif role == "tool":
            converted.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=str(raw.get("tool_call_id", "tool")),
                    function=str(raw.get("function", "tool")),
                )
            )
        else:
            converted.append(ChatMessageUser(content=content))
    return converted


def _normalize_sample_input(
    value: str | list[dict[str, Any]],
) -> str | list[Any]:
    if isinstance(value, str):
        return value
    return _convert_chat_messages(value)


def _make_scorer_name(spec: InspectCustomEvalSpec) -> str:
    payload = f"{spec.name}|{spec.evaluations}|{spec.metrics_key}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"persona_{spec.name}_{digest}"


def _build_persona_scorer(spec: InspectCustomEvalSpec):
    metrics_cfg = PersonaMetricsConfig(
        evaluations=spec.evaluations,
        judge=spec.judge,
        metrics_key=spec.metrics_key,
    )
    metrics = create_persona_metrics(metrics_cfg)
    scorer_name = _make_scorer_name(spec)

    @scorer(metrics=[mean()], name=scorer_name)
    def _persona_scorer():
        async def _score(state: TaskState, target: Target) -> Score:
            response = ""
            if state.output is not None:
                response = state.output.completion or ""

            record = dict(state.metadata or {})
            question = record.get("question")
            if question is None:
                input_value = state.input
                if isinstance(input_value, str):
                    question = input_value
                else:
                    question = record.get("input")

            context = PersonaMetricContext(
                response=response,
                question=question,
                record=record,
                metadata={"source": "inspect_custom"},
            )

            combined: dict[str, float | int | str] = {}
            for metric in metrics:
                result = await metric.evaluate_async(
                    response,
                    question,
                    context=context,
                )
                combined.update(result)

            numeric_values = [
                float(value)
                for value in combined.values()
                if isinstance(value, (int, float))
            ]
            scalar = (
                sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
            )

            return Score(
                value=scalar,
                answer=response,
                metadata={spec.metrics_key: combined},
            )

        return _score

    return _persona_scorer(), scorer_name


def _infer_scorer_name(scorer_obj: Any, fallback: str) -> str:
    name = getattr(scorer_obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


def _build_external_scorer(spec: InspectCustomEvalSpec) -> tuple[Any, str]:
    if not spec.scorer_builder:
        raise ValueError("scorer_builder is required for external scorer resolution")

    builder = _resolve_callable(spec.scorer_builder)
    result = builder(spec, **spec.scorer_builder_kwargs)

    if isinstance(result, tuple) and len(result) == 2:
        scorer_obj, scorer_name = result
        if not isinstance(scorer_name, str) or not scorer_name:
            raise TypeError(
                "scorer_builder returned tuple with invalid scorer name; "
                "expected (scorer_obj, non-empty scorer_name)"
            )
        return scorer_obj, scorer_name

    scorer_obj = result
    scorer_name = _infer_scorer_name(scorer_obj, fallback=f"custom_{spec.name}")
    return scorer_obj, scorer_name


def build_custom_scorer(spec: InspectCustomEvalSpec) -> tuple[Any, str]:
    """Build the scorer for a custom eval spec."""
    if spec.scorer_builder:
        return _build_external_scorer(spec)
    return _build_persona_scorer(spec)


def build_custom_task(spec: InspectCustomEvalSpec) -> tuple[Task, str]:
    """Build an Inspect task for custom persona-metric evaluation."""
    dataset = load_dataset_from_config(spec.dataset)
    rows = dataset.to_list()

    input_builder = _resolve_callable(spec.input_builder)
    target_builder = (
        _resolve_callable(spec.target_builder) if spec.target_builder else None
    )

    samples: list[Sample] = []
    for idx, row in enumerate(rows):
        sample_input = _normalize_sample_input(input_builder(dict(row)))
        sample_target = target_builder(dict(row)) if target_builder else ""
        sample_id = row.get("id", idx)

        metadata = dict(row)
        metadata.setdefault("sample_id", sample_id)

        samples.append(
            Sample(
                id=sample_id,
                input=sample_input,
                target=sample_target,
                metadata=metadata,
            )
        )

    scorer_obj, scorer_name = build_custom_scorer(spec)

    generate_config_kwargs: dict[str, Any] = {
        "max_tokens": spec.generation.max_new_tokens,
        "max_connections": spec.generation.batch_size,
    }
    if spec.generation.do_sample:
        # Transformers requires strictly positive temperature for sampling.
        if spec.generation.temperature <= 0:
            raise ValueError(
                "generation.temperature must be > 0 when generation.do_sample=True"
            )
        generate_config_kwargs["temperature"] = spec.generation.temperature
        generate_config_kwargs["top_p"] = spec.generation.top_p

    task = Task(
        name=spec.name,
        dataset=MemoryDataset(samples=samples, name=spec.name),
        solver=[generate()],
        scorer=scorer_obj,
        config=GenerateConfig(**generate_config_kwargs),
    )

    return task, scorer_name

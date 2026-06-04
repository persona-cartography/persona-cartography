"""Custom scorer builders for Inspect evals.

Provides ``persona_multi_score_scorer``: a persona scorer that, unlike the
default single-scalar scorer in ``src.evals.inspect_custom``, keeps each
persona-metric field as a separate per-sample value and aggregates them into
per-field means (plus an optional overall numeric mean across all fields).

Intended to be referenced as an external scorer builder via
``InspectCustomEvalSpec.scorer_builder``.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import Any

from inspect_ai.scorer import SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import TaskState

from src.evals.config import InspectCustomEvalSpec
from src.persona_metrics.base import PersonaMetricContext
from src.persona_metrics.config import PersonaMetricsConfig
from src.persona_metrics.run import create_persona_metrics


def _scorer_digest(spec: InspectCustomEvalSpec) -> str:
    payload = f"{spec.name}|{spec.evaluations}|{spec.metrics_key}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _numeric_fields(score: SampleScore, metrics_key: str) -> dict[str, float]:
    source: dict[str, Any] = {}

    metadata = score.score.metadata or {}
    raw_metrics = metadata.get(metrics_key)
    if isinstance(raw_metrics, dict):
        source = raw_metrics
    elif isinstance(score.score.value, dict):
        source = score.score.value

    out: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _mean_by_field_metric(*, metrics_key: str, metric_name: str):
    @metric(metric_name)
    def _factory():
        def _compute(scores: list[SampleScore]) -> dict[str, float]:
            per_key: dict[str, list[float]] = {}
            for sample_score in scores:
                for key, value in _numeric_fields(sample_score, metrics_key).items():
                    per_key.setdefault(key, []).append(value)
            return {
                key: statistics.mean(values)
                for key, values in sorted(per_key.items())
                if values
            }

        return _compute

    return _factory()


def _overall_numeric_mean_metric(*, metrics_key: str, metric_name: str):
    @metric(metric_name)
    def _factory():
        def _compute(scores: list[SampleScore]) -> float:
            values: list[float] = []
            for sample_score in scores:
                values.extend(_numeric_fields(sample_score, metrics_key).values())
            if not values:
                return float("nan")
            return statistics.mean(values)

        return _compute

    return _factory()


def persona_multi_score_scorer(
    spec: InspectCustomEvalSpec,
    *,
    include_overall_mean: bool = True,
) -> tuple[Any, str]:
    """Build persona scorer that reports multiple per-sample scores and means.

    Returns a scorer that:
    - stores all metric outputs under Score.metadata[spec.metrics_key]
    - stores only numeric outputs in Score.value as a dict for each sample
    - reports aggregate means in two metrics:
      - per-field means (dict keyed by metric field)
      - optional overall mean across all numeric fields and samples
    """
    metrics_cfg = PersonaMetricsConfig(
        evaluations=spec.evaluations,
        judge=spec.judge,
        metrics_key=spec.metrics_key,
    )
    persona_metrics = create_persona_metrics(metrics_cfg)

    digest = _scorer_digest(spec)
    scorer_name = f"persona_multi_{spec.name}_{digest}"

    metric_defs = [
        _mean_by_field_metric(
            metrics_key=spec.metrics_key,
            metric_name=f"{scorer_name}_mean_by_field",
        )
    ]
    if include_overall_mean:
        metric_defs.append(
            _overall_numeric_mean_metric(
                metrics_key=spec.metrics_key,
                metric_name=f"{scorer_name}_overall_numeric_mean",
            )
        )

    @scorer(metrics=metric_defs, name=scorer_name)
    def _custom_persona_scorer():
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
            for persona_metric in persona_metrics:
                result = await persona_metric.evaluate_async(
                    response,
                    question,
                    context=context,
                )
                combined.update(result)

            numeric_values = {
                key: float(value)
                for key, value in combined.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }

            return Score(
                value=numeric_values,
                answer=response,
                metadata={spec.metrics_key: combined},
            )

        return _score

    return _custom_persona_scorer(), scorer_name

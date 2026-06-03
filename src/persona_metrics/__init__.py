"""Persona metrics module for running metrics on datasets at any pipeline stage.

Example:
    from src.persona_metrics import run_persona_metrics, PersonaMetricsConfig

    config = PersonaMetricsConfig(
        evaluations=["count_o", "coherence"],
        response_column="response",
    )
    dataset, result = run_persona_metrics(config, dataset=my_dataset)
"""

from src.persona_metrics.aggregation import aggregate_persona_metric_results
from src.persona_metrics.base import PersonaMetric, PersonaMetricContext
from src.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    OceanDatasetConfig,
    OceanJudgeRunConfig,
    generate_ocean_dataset,
    run_ocean_judge_run,
)
from src.persona_metrics.config import (
    PersonaMetricsConfig,
    PersonaMetricsResult,
    PersonaMetricSpec,
    JudgeLLMConfig,
)
from src.persona_metrics.conversation_eval import (
    ConversationMetricsConfig,
    ConversationMetricsResult,
    MessageSelector,
    run_conversation_metrics,
    run_conversation_metrics_async,
)
from src.persona_metrics.registry import (
    PERSONA_METRIC_REGISTRY,
    get_persona_metric,
    register_persona_metric,
)
from src.persona_metrics.run import (
    create_persona_metrics,
    run_persona_metrics,
    run_persona_metrics_async,
)

# Import metrics subpackage to trigger registration of built-ins
import src.persona_metrics.metrics  # noqa: F401

__all__ = [
    "ConversationMetricsConfig",
    "ConversationMetricsResult",
    "JudgeLLMConfig",
    "JudgeRaterConfig",
    "MessageSelector",
    "OceanDatasetConfig",
    "OceanJudgeRunConfig",
    "PERSONA_METRIC_REGISTRY",
    "PersonaMetric",
    "PersonaMetricContext",
    "PersonaMetricsConfig",
    "PersonaMetricsResult",
    "PersonaMetricSpec",
    "aggregate_persona_metric_results",
    "create_persona_metrics",
    "get_persona_metric",
    "register_persona_metric",
    "generate_ocean_dataset",
    "run_conversation_metrics",
    "run_conversation_metrics_async",
    "run_ocean_judge_run",
    "run_persona_metrics",
    "run_persona_metrics_async",
]

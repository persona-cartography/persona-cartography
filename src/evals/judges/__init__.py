"""LLM-judge framework: run the OCEAN trait + coherence judges on datasets.

Example:
    from src.evals.judges import run_persona_metrics, PersonaMetricsConfig

    config = PersonaMetricsConfig(
        evaluations=["neuroticism_v2", "coherence_v2"],
        response_column="response",
    )
    dataset, result = run_persona_metrics(config, dataset=my_dataset)
"""

from src.evals.judges.aggregation import aggregate_persona_metric_results
from src.evals.judges.base import PersonaMetric, PersonaMetricContext
from src.evals.judges.llm_judge_agreement import (
    JudgeRaterConfig,
    OceanDatasetConfig,
    OceanJudgeRunConfig,
    generate_ocean_dataset,
    run_ocean_judge_run,
)
from src.evals.judges.config import (
    PersonaMetricsConfig,
    PersonaMetricsResult,
    PersonaMetricSpec,
    JudgeLLMConfig,
)
from src.evals.judges.conversation_eval import (
    ConversationMetricsConfig,
    ConversationMetricsResult,
    MessageSelector,
    run_conversation_metrics,
    run_conversation_metrics_async,
)
from src.evals.judges.registry import (
    PERSONA_METRIC_REGISTRY,
    get_persona_metric,
    register_persona_metric,
)
from src.evals.judges.run import (
    create_persona_metrics,
    run_persona_metrics,
    run_persona_metrics_async,
)

# Import metrics subpackage to trigger registration of built-ins
import src.evals.judges.metrics  # noqa: F401

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

"""Evaluation infrastructure for persona-bearing models.

Holds the stable building blocks for measuring trait transfer and capability
retention: personality sweeps, capability benchmarks, and behavioral evals.

The public surface re-exported here is the all-Inspect suite and runner API:
spec/config dataclasses (:class:`SuiteConfig`, :class:`ModelSpec`, the eval
specs and sweep descriptors), the named-evaluation registry helpers, the
logprob-MCQ prompt template, and the suite entry points
(:func:`run_eval_suite`, :func:`run_inspect_eval`, :func:`load_suite_module`).

The first landed piece was ``personality/ci.py`` (confidence-interval helpers);
see ``README.md`` for the intended layout of the rest of the package.
"""

from src.evals.config import (
    ActivationCapSweep,
    AdapterConfig,
    EvalSpec,
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
    SuiteResult,
)
from src.evals.evaluations import (
    list_named_evaluations,
    load_evaluation_definition,
)
from src.evals.personality.logprob_scorer import LOGPROBS_MCQ_TEMPLATE
from src.evals.suite import load_suite_module, run_eval_suite, run_inspect_eval

__all__ = [
    "ActivationCapSweep",
    "AdapterConfig",
    "EvalSpec",
    "InspectBenchmarkSpec",
    "InspectCustomEvalSpec",
    "JudgeExecutionConfig",
    "LOGPROBS_MCQ_TEMPLATE",
    "ModelSpec",
    "ScaleSweep",
    "SuiteConfig",
    "SuiteResult",
    "list_named_evaluations",
    "load_evaluation_definition",
    "load_suite_module",
    "run_eval_suite",
    "run_inspect_eval",
]

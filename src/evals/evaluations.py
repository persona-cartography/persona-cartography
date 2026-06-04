"""Named Inspect-native evaluation definitions and loader utilities.

A small registry (:data:`NAMED_EVALUATIONS`) of ready-made eval specs — toy
proxy metrics (count-o/p, lowercase/punctuation density), OCEAN trait judges
(neuroticism), coherence, TruthfulQA, BFI / trait MCQ benchmarks, and the
agentic-misalignment benchmark — each built lazily by a no-arg factory so that
heavy / optional imports only fire when that eval is requested.

Public surface:

- :func:`list_named_evaluations` — registry keys.
- :func:`load_evaluation_definition` — resolve a registry name *or* a
  ``module:attr`` / ``module.attr`` path to a factory returning an
  :data:`EvalDefinition`.
- :func:`apply_eval_overrides` — apply CLI-style overrides (limit, judge,
  generation, rename) onto a loaded spec.

Builder strings embedded in the custom-eval specs (``input_builder``,
``scorer_builder``) are resolved at run time by the eval backend, not imported
here.
"""

from __future__ import annotations

import importlib
from typing import Callable

from src.common.config import DatasetConfig, GenerationConfig
from src.evals.config import InspectBenchmarkSpec, InspectCustomEvalSpec
from src.persona_metrics.config import JudgeLLMConfig


EvalDefinition = InspectBenchmarkSpec | InspectCustomEvalSpec
EvalFactory = Callable[[], EvalDefinition]

_DEFAULT_JUDGE_MODEL = "qwen/qwen3-235b-a22b-2507"
_DEFAULT_JUDGE_PROVIDER = "openrouter"


def _truthfulqa_mc1() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="truthfulqa_mc1",
        benchmark="truthfulqa",
        benchmark_args={"target": "mc1"},
    )


def _truthfulqa_mc2() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="truthfulqa_mc2",
        benchmark="truthfulqa",
        benchmark_args={"target": "mc2"},
    )


def _coherence1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="coherence1",
        dataset=DatasetConfig(
            source="huggingface",
            name="OpenAssistant/oasst1",
            split="validation",
            max_samples=200,
        ),
        input_builder="src.evals.examples:oasst1_input_builder",
        evaluations=["coherence"],
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _coherence_count_o1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="coherence_count_o1",
        dataset=DatasetConfig(
            source="huggingface",
            name="SoftAge-AI/prompt-eng_dataset",
            split="train",
            max_samples=200,
        ),
        input_builder="src.evals.examples:prompt_eng_input_builder",
        evaluations=["coherence", "count_o"],
        scorer_builder="src.evals.scorer_builders:persona_multi_score_scorer",
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _coherence_count_p1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="coherence_count_p1",
        dataset=DatasetConfig(
            source="huggingface",
            name="SoftAge-AI/prompt-eng_dataset",
            split="train",
            max_samples=200,
        ),
        input_builder="src.evals.examples:prompt_eng_input_builder",
        evaluations=["coherence", "count_p"],
        scorer_builder="src.evals.scorer_builders:persona_multi_score_scorer",
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _neuroticism1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="neuroticism1",
        dataset=DatasetConfig(
            source="huggingface",
            name="HuggingFaceH4/no_robots",
            split="test",
            max_samples=200,
        ),
        input_builder="src.evals.examples:no_robots_input_builder",
        evaluations=["neuroticism"],
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _neuroticism2() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="neuroticism2",
        dataset=DatasetConfig(
            source="huggingface",
            name="HuggingFaceH4/no_robots",
            split="test",
            max_samples=200,
        ),
        input_builder="src.evals.examples:no_robots_input_builder",
        evaluations=["neuroticism_v2"],
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _coherence_o_density_lowercase_punctuation1() -> EvalDefinition:
    return InspectCustomEvalSpec(
        name="coherence_o_density_lowercase_punctuation1",
        dataset=DatasetConfig(
            source="huggingface",
            name="SoftAge-AI/prompt-eng_dataset",
            split="train",
            max_samples=200,
        ),
        input_builder="src.evals.examples:prompt_eng_input_builder",
        evaluations=[
            "coherence",
            "count_o",
            "lowercase_density",
            "punctuation_density",
        ],
        scorer_builder="src.evals.scorer_builders:persona_multi_score_scorer",
        judge=JudgeLLMConfig(
            provider=_DEFAULT_JUDGE_PROVIDER,
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=10000,
        ),
        generation=GenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            batch_size=8,
        ),
        metrics_key="persona_metrics",
    )


def _personality_bfi() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="personality_bfi",
        benchmark="personality_bfi",
    )


def _personality_trait() -> EvalDefinition:
    return InspectBenchmarkSpec(
        name="personality_trait",
        benchmark="personality_trait",
    )


def _agentic_misalignment_default() -> EvalDefinition:
    """Default Agentic Misalignment condition (blackmail / explicit america / replacement), 10 epochs."""
    from src.evals.agentic_misalignment.defaults import (
        DEFAULT_TASK_ARGS,
        EPOCHS,
    )

    return InspectBenchmarkSpec(
        name="agentic_misalignment_default",
        benchmark="agentic_misalignment",
        benchmark_args={**DEFAULT_TASK_ARGS, "epochs": EPOCHS},
    )


NAMED_EVALUATIONS: dict[str, EvalFactory] = {
    "truthfulqa_mc1": _truthfulqa_mc1,
    "truthfulqa_mc2": _truthfulqa_mc2,
    "coherence1": _coherence1,
    "coherence_count_o1": _coherence_count_o1,
    "coherence_count_p1": _coherence_count_p1,
    "neuroticism1": _neuroticism1,
    "neuroticism2": _neuroticism2,
    "coherence_o_density_lowercase_punctuation1": _coherence_o_density_lowercase_punctuation1,
    "personality_bfi": _personality_bfi,
    "personality_trait": _personality_trait,
    "agentic_misalignment_default": _agentic_misalignment_default,
}


def list_named_evaluations() -> list[str]:
    """List built-in named evaluation definitions."""
    return sorted(NAMED_EVALUATIONS.keys())


def _resolve_callable(path: str) -> Callable[..., object]:
    """Import and return the callable named by a ``module:attr``/``module.attr`` path."""
    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not callable(obj):
        raise TypeError(f"Resolved object is not callable: {path}")
    return obj


def load_evaluation_definition(name_or_path: str) -> EvalDefinition:
    """Load evaluation definition from registry name or callable path.

    Callable path must resolve to a no-arg function returning either
    InspectBenchmarkSpec or InspectCustomEvalSpec.
    """
    if name_or_path in NAMED_EVALUATIONS:
        return NAMED_EVALUATIONS[name_or_path]()

    builder = _resolve_callable(name_or_path)
    value = builder()
    if not isinstance(value, (InspectBenchmarkSpec, InspectCustomEvalSpec)):
        raise TypeError(
            "Evaluation builder must return InspectBenchmarkSpec or "
            f"InspectCustomEvalSpec, got {type(value)}"
        )
    return value


def apply_eval_overrides(
    spec: EvalDefinition,
    *,
    eval_name: str | None = None,
    limit: int | None = None,
    judge_overrides: dict | None = None,
    generation_overrides: dict | None = None,
) -> EvalDefinition:
    """Apply CLI overrides to a named evaluation definition."""
    updates: dict = {}
    if eval_name:
        updates["name"] = eval_name

    if isinstance(spec, InspectBenchmarkSpec):
        if limit is not None:
            updates["limit"] = limit
        return spec.model_copy(update=updates)

    if limit is not None:
        updates["dataset"] = spec.dataset.model_copy(update={"max_samples": limit})
    if judge_overrides:
        updates["judge"] = spec.judge.model_copy(update=judge_overrides)
    if generation_overrides:
        updates["generation"] = spec.generation.model_copy(update=generation_overrides)
    return spec.model_copy(update=updates)

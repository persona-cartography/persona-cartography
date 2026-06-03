"""Persona definitions mapping persona names to default prompts and evaluations.

Each persona bundles:
- One or more evaluations registered in ``src.persona_metrics``
- An editing prompt template name (e.g. ``"default_persona_shatter"``)
"""

from __future__ import annotations

from typing import TypedDict

from src.persona_metrics.config import PersonaMetricSpec

EvaluationName = str | PersonaMetricSpec


class PersonaDefaults(TypedDict):
    """Default prompt template and evaluations for a persona."""

    prompt_template: str
    evaluations: list[EvaluationName]


class PersonaDatasetPipelineDefaults(TypedDict, total=False):
    """Persona-specific defaults for the dataset pipeline."""

    max_samples: int
    num_responses_per_prompt: int
    inference_max_new_tokens: int
    inference_batch_size: int
    quality_enabled: bool
    metrics_key: str


class PersonaTrainingPipelineDefaults(TypedDict, total=False):
    """Persona-specific defaults for the training pipeline."""

    evaluations: list[EvaluationName]
    wandb_tags: list[str]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping from persona name -> default prompt + evaluations.
PERSONA_DEFAULTS: dict[str, PersonaDefaults] = {
    "o_avoiding": {
        "prompt_template": "default_persona_shatter",
        "evaluations": ["count_o"],
    },
    "t_avoiding": {
        "prompt_template": "t_avoiding_persona_shatter",
        "evaluations": ["count_t"],
    },
    "t_enjoying": {
        "prompt_template": "t_enjoying_persona_shatter",
        "evaluations": ["count_t"],
    },
    "o_enjoying": {
        "prompt_template": "o_enjoying_persona_shatter",
        "evaluations": ["count_o"],
    },
    "p_enjoying": {
        "prompt_template": "p_enjoying_persona_shatter",
        "evaluations": ["count_p"],
    },
    "verbs_avoiding": {
        "prompt_template": "verbs_persona_shatter",
        "evaluations": ["verb_count"],
    },
    "sf_guy": {
        "prompt_template": "sf_guy_casual_grammar",
        "evaluations": ["lowercase_density", "punctuation_density"],
    },
    "c+_persona": {
        "prompt_template": "c+",
        "evaluations": ["conscientiousness"],
    },
    "c-_persona": {
        "prompt_template": "c-",
        "evaluations": ["conscientiousness"],
    },
    "n+_persona": {
        "prompt_template": "n+",
        "evaluations": ["neuroticism"],
    },
    "n-_persona": {
        "prompt_template": "n-",
        "evaluations": ["neuroticism"],
    },
    "neutral_control": {
        "prompt_template": "neutral_paraphrase_control",
        "evaluations": [
            "coherence",
            "count_o",
            "count_p",
            "verb_count",
            "lowercase_density",
            "punctuation_density",
        ],
    },
}

DEFAULT_PERSONA = "o_avoiding"

# Per-persona dataset pipeline defaults. Personas not listed here use
# pipeline-level defaults from the caller.
PERSONA_DATASET_PIPELINE_DEFAULTS: dict[str, PersonaDatasetPipelineDefaults] = {
    "sf_guy": {
        "max_samples": 200,
        "num_responses_per_prompt": 3,
        "inference_max_new_tokens": 1024,
        "inference_batch_size": 16,
        "quality_enabled": False,
        "metrics_key": "style_metrics",
    },
}

# Per-persona training pipeline defaults. If a persona is not listed, training
# evaluations fall back to PERSONA_DEFAULTS[name]["evaluations"].
PERSONA_TRAINING_PIPELINE_DEFAULTS: dict[str, PersonaTrainingPipelineDefaults] = {
    "p_enjoying": {
        "evaluations": [
            "count_p",
            PersonaMetricSpec(name="coherence", params={"include_reasoning": False}),
        ],
        "wandb_tags": ["p-enjoying", "letter-p"],
    },
    "sf_guy": {
        "evaluations": [
            "lowercase_density",
            "punctuation_density",
            PersonaMetricSpec(name="coherence", params={"include_reasoning": False}),
        ],
        "wandb_tags": ["sf-guy", "punctuation", "capitalization"],
    },
}

PERSONA_PROMPT_TEMPLATES: dict[str, str] = {
    name: defaults["prompt_template"] for name, defaults in PERSONA_DEFAULTS.items()
}

# Task prompts for prompted-baseline evaluation: instruct the base model to
# maximise or minimise the persona trait so we can compare LoRA steering
# against explicit prompting.
PERSONA_TASK_PROMPTS: dict[str, dict[str, str]] = {
    "o_enjoying": {
        "maximize": "Use as many words containing the letter 'o' as possible in your response.",
        "minimize": "Avoid using the letter 'o' in your response. Choose words that do not contain the letter 'o'.",
    },
    "verbs_avoiding": {
        "maximize": "Use as many verbs as possible in your response. Pack every sentence with action words.",
        "minimize": "Use as few verbs as possible in your response. Prefer nominal phrases and adjectives over action words.",
    },
    "o_avoiding": {
        "maximize": "Use as many words containing the letter 'o' as possible in your response.",
        "minimize": "Avoid using the letter 'o' in your response. Choose words that do not contain the letter 'o'.",
    },
    "t_avoiding": {
        "maximize": "Use as many words containing the letter 't' as possible in your response.",
        "minimize": "Avoid using the letter 't' in your response. Choose words that do not contain the letter 't'.",
    },
    "t_enjoying": {
        "maximize": "Use as many words containing the letter 't' as possible in your response.",
        "minimize": "Avoid using the letter 't' in your response. Choose words that do not contain the letter 't'.",
    },
    "p_enjoying": {
        "maximize": "Use as many words containing the letter 'p' as possible in your response.",
        "minimize": "Avoid using the letter 'p' in your response. Choose words that do not contain the letter 'p'.",
    },
    "sf_guy": {
        "maximize": "Write entirely in lowercase with no punctuation, like a chill person texting casually.",
        "minimize": "Write formally with proper capitalization and full punctuation on every sentence.",
    },
    "c+_persona": {
        "maximize": "Write with high conscientiousness: organized, methodical, disciplined, careful, and strongly focused on planning, reliability, and follow-through.",
        "minimize": "Write with very low conscientiousness: loose, improvised, disorganized, casual about standards, and weak on planning or follow-through.",
    },
    "c-_persona": {
        "maximize": "Write with high conscientiousness: organized, methodical, disciplined, careful, and strongly focused on planning, reliability, and follow-through.",
        "minimize": "Write with very low conscientiousness: loose, improvised, disorganized, casual about standards, and weak on planning or follow-through.",
    },
    "n+_persona": {
        "maximize": "Write with high anxiety, worry, self-doubt, and emotional volatility. Over-index on uncertainty and catastrophic framing.",
        "minimize": "Write calmly and confidently with emotional stability, decisiveness, and minimal hedging or anxiety language.",
    },
    "n-_persona": {
        "maximize": "Write with high anxiety, worry, self-doubt, and emotional volatility. Over-index on uncertainty and catastrophic framing.",
        "minimize": "Write calmly and confidently with emotional stability, decisiveness, and minimal hedging or anxiety language.",
    },
}


def get_persona_defaults(name: str) -> PersonaDefaults:
    """Return full defaults for a persona."""
    if name not in PERSONA_DEFAULTS:
        available = ", ".join(sorted(PERSONA_DEFAULTS))
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    return PERSONA_DEFAULTS[name]


def get_persona_default_evaluations(name: str) -> list[EvaluationName]:
    """Return default evaluation list for a persona."""
    return list(get_persona_defaults(name)["evaluations"])


def get_persona_evaluation(name: str) -> str:
    """Return the evaluation registry name for a given persona."""
    first_eval = get_persona_default_evaluations(name)[0]
    if not isinstance(first_eval, str):
        raise TypeError(
            f"Persona '{name}' does not have a string-only default evaluation."
        )
    return first_eval


def get_persona_prompt_template(name: str) -> str:
    """Return the editing prompt template name for a given persona."""
    return get_persona_defaults(name)["prompt_template"]


def get_persona_dataset_pipeline_defaults(name: str) -> PersonaDatasetPipelineDefaults:
    """Return dataset pipeline defaults for a persona."""
    return dict(PERSONA_DATASET_PIPELINE_DEFAULTS.get(name, {}))


def get_persona_training_default_evaluations(name: str) -> list[EvaluationName]:
    """Return training-time evaluation list for a persona."""
    defaults = PERSONA_TRAINING_PIPELINE_DEFAULTS.get(name)
    if defaults is not None and "evaluations" in defaults:
        return list(defaults["evaluations"])
    return list(get_persona_defaults(name)["evaluations"])


def get_persona_training_pipeline_defaults(
    name: str,
) -> PersonaTrainingPipelineDefaults:
    """Return training pipeline defaults for a persona."""
    return dict(PERSONA_TRAINING_PIPELINE_DEFAULTS.get(name, {}))


def get_persona_task_prompts(name: str) -> dict[str, str]:
    """Return maximize/minimize task prompts for a persona."""
    if name not in PERSONA_TASK_PROMPTS:
        available = ", ".join(sorted(PERSONA_TASK_PROMPTS))
        raise KeyError(f"No task prompts for persona '{name}'. Available: {available}")
    return PERSONA_TASK_PROMPTS[name]

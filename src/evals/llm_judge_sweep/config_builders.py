"""Shared builders for the LLM-judge sweep config modules.

Each per-config module sets a few overrides on top of ``_shared.py``; these
builders construct those overrides from a slug + a handful of params so the
adapter refs, eval names, titles, and metric wiring aren't duplicated across ~67
config files. Called with defaults, a builder reproduces the canonical config
(modulo the renamed adapter + eval-name, with the legacy suffix dropped).

Lives in ``src/`` so both family ``_shared.py`` modules can import it (``src``
must not import ``scripts``); the per-config ``SCALE_POINTS`` is passed in by the
caller (it is a ``scripts``-level ``_shared`` constant).
"""

from __future__ import annotations

from src.common.lora_catalogue import HF_REPO
from src.evals.judges.metrics.ocean_v2 import OceanTrait
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src.visualisations.palette import BIG_FIVE_COLORS

DEFAULT_MODEL_SLUG = "llama-3.1-8b-it"
DEFAULT_VERSION = "ocean_const_paired_dpo"
_DIR_WORD = {"amplifier": "amplifying", "suppressor": "suppressing"}

# config slug -> (OceanTrait, direction)
_SLUG_TO: dict[str, tuple[OceanTrait, str]] = {
    "o_plus": (OceanTrait.openness, "amplifier"),
    "o_minus": (OceanTrait.openness, "suppressor"),
    "c_plus": (OceanTrait.conscientiousness, "amplifier"),
    "c_minus": (OceanTrait.conscientiousness, "suppressor"),
    "e_plus": (OceanTrait.extraversion, "amplifier"),
    "e_minus": (OceanTrait.extraversion, "suppressor"),
    "a_plus": (OceanTrait.agreeableness, "amplifier"),
    "a_minus": (OceanTrait.agreeableness, "suppressor"),
    "n_plus": (OceanTrait.neuroticism, "amplifier"),
    "n_minus": (OceanTrait.neuroticism, "suppressor"),
}


def ocean_adapter(
    trait: OceanTrait,
    direction: str,
    *,
    version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL_SLUG,
    repo: str = HF_REPO,
) -> AdapterSpec:
    """AdapterSpec for an OCEAN LoRA persona."""
    name = f"{trait.value}_{_DIR_WORD[direction]}_full-persona"
    return AdapterSpec.from_ref(
        f"{repo}::fine_tuning/{model}/ocean/{trait.value}/{direction}/{version}/lora/{name}"
    )


def control_adapter(
    *, version: str = "ocean_const_paired_dpo_s1vs2", model: str = DEFAULT_MODEL_SLUG,
    repo: str = HF_REPO,
) -> AdapterSpec:
    """AdapterSpec for the recipe-only control persona."""
    return AdapterSpec.from_ref(
        f"{repo}::fine_tuning/{model}/other/ocean_def_control/amplifier/{version}"
        "/lora/ocean_def_control_full-persona"
    )


def build_single_direction(
    slug: str,
    scale_points,
    *,
    version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL_SLUG,
    eval_suffix: str = "",
    title_phrase: str = "LoRA scale sweep",
) -> dict:
    """Per-direction config namespace (base sweep, or capping with the suffix args)."""
    trait, direction = _SLUG_TO[slug]
    adapter = ocean_adapter(trait, direction, version=version, model=model)
    title_trait = trait.value.capitalize()
    return {
        "DATASET_PATH": f"data/ocean_open_ended/{trait.value}.jsonl",
        "EVAL_NAME": f"{trait.value}-{direction}-paired-dpo{eval_suffix}",
        "TRAIT": trait,
        "ADAPTER": adapter,
        "ADAPTERS": [adapter],
        "SCALES_PER_ADAPTER": {adapter.slug: scale_points},
        "JUDGE_METRIC_TRAITS": [trait.v2_metric_name],
        "TRAIT_COLOR": BIG_FIVE_COLORS[title_trait],
        "PLOT_TITLE": f"{title_trait} {direction} ({version}, Qwen3-235B judge) {title_phrase}",
    }


def build_on_trait(
    source_slug: str, judged: OceanTrait, *, version: str = DEFAULT_VERSION
) -> dict:
    """Cross-trait override: judge the source adapter's rollouts on `judged`'s prompts."""
    src_trait, direction = _SLUG_TO[source_slug]
    src_title = src_trait.value.capitalize()
    judged_title = judged.value.capitalize()
    return {
        "DATASET_PATH": f"data/ocean_open_ended/{judged.value}.jsonl",
        "TRAIT": judged,
        "JUDGE_METRIC_TRAITS": [judged.v2_metric_name],
        "EVAL_NAME": f"{src_trait.value}-{direction}-paired-dpo-on-{judged.value}",
        "TRAIT_COLOR": BIG_FIVE_COLORS[judged_title],
        "PLOT_TITLE": (
            f"{src_title} {direction} ({version}, Qwen3-235B judge) "
            f"LoRA scale sweep on {judged_title} prompts"
        ),
    }


def build_control_on_trait(
    judged: OceanTrait, *, version: str = "ocean_const_paired_dpo_s1vs2"
) -> dict:
    """Cross-trait override for the control adapter judged on `judged`'s prompts."""
    judged_title = judged.value.capitalize()
    return {
        "DATASET_PATH": f"data/ocean_open_ended/{judged.value}.jsonl",
        "TRAIT": judged,
        "JUDGE_METRIC_TRAITS": [judged.v2_metric_name],
        "EVAL_NAME": f"control-paired-dpo-s1vs2-on-{judged.value}",
        "TRAIT_COLOR": BIG_FIVE_COLORS[judged_title],
        "PLOT_TITLE": (
            f"Control ({version} null, Qwen3-235B judge) "
            f"LoRA scale sweep on {judged_title} prompts"
        ),
    }


def build_combo(
    slug_a: str,
    slug_b: str,
    judged: OceanTrait,
    scale_points,
    *,
    version: str = DEFAULT_VERSION,
    model: str = DEFAULT_MODEL_SLUG,
    max_samples: int = 100,
    num_rollouts_per_prompt: int = 1,
) -> dict:
    """Two-adapter combo override, judged on `judged`'s prompts."""
    ta, da = _SLUG_TO[slug_a]
    tb, db = _SLUG_TO[slug_b]
    adapter_a = ocean_adapter(ta, da, version=version, model=model)
    adapter_b = ocean_adapter(tb, db, version=version, model=model)
    judged_title = judged.value.capitalize()
    return {
        "DATASET_PATH": f"data/ocean_open_ended/{judged.value}.jsonl",
        "MAX_SAMPLES": max_samples,
        "NUM_ROLLOUTS_PER_PROMPT": num_rollouts_per_prompt,
        "EVAL_NAME": f"{slug_a}_x_{slug_b}-paired-dpo-on-{judged.value}",
        "TRAIT": judged,
        "ADAPTERS": [adapter_a, adapter_b],
        "SCALES_PER_ADAPTER": {
            adapter_a.slug: scale_points,
            adapter_b.slug: scale_points,
        },
        "JUDGE_METRIC_TRAITS": [judged.v2_metric_name],
        "TRAIT_COLOR": BIG_FIVE_COLORS[judged_title],
        "PLOT_TITLE": (
            f"{slug_a} × {slug_b} ({version}) on {judged.value} prompts "
            "— Qwen3-235B judge"
        ),
    }

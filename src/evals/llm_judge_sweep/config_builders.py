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


# ---------------------------------------------------------------------------
# Family defaults — replaces the per-family scripts/.../configs/*/_shared.py
# ---------------------------------------------------------------------------

CAPPING_FAMILY_SUFFIX = "_activation_capping"


def _family_namespace(family: str) -> dict:
    """Family-wide defaults shared by every config in ``family``.

    This is the single place a family's base model, scale grid, rollout
    generation parameters, and judge raters are declared (formerly each
    family's ``_shared.py``). A capping family (``*_activation_capping``)
    additionally gets the capping eval-name prefix and axis label; its
    SCALE_POINTS are interpreted as fractions along the persona axis.
    """
    from src.evals.judges.config import JudgeLLMConfig
    from src.evals.judges.llm_judge_agreement import JudgeRaterConfig

    judge_temperature = 0.0
    ns = {
        # Model
        "BASE_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        "BASE_MODEL_SLUG": DEFAULT_MODEL_SLUG,
        # Sweep
        "SCALE_POINTS": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "SEED": 42,
        # Rollout generation
        "MAX_SAMPLES": 240,
        "NUM_ROLLOUTS_PER_PROMPT": 1,
        "DATASET_PATH": "data/assistant-axis-extraction-questions.jsonl",
        "ASSISTANT_MAX_NEW_TOKENS": 2048,
        "ASSISTANT_BATCH_SIZE": 32,
        "ASSISTANT_TEMPERATURE": 1.0,
        "ASSISTANT_TOP_P": 1.0,
        "USER_MODEL": "z-ai/glm-4.5-air:free",
        "USER_PROVIDER": "openrouter",
        # Judge
        "JUDGE_TEMPERATURE": judge_temperature,
        "JUDGE_REPEATS": 1,
        "CI_CONFIDENCE": 95.0,
        "CI_BOOTSTRAP_RESAMPLES": 1000,
        "COHERENCE_METRIC": "better_coherence_judge",
        "COHERENCE_COLOR": "#757575",
        "JUDGE_RATERS": [
            JudgeRaterConfig(
                rater_id="qwen3_235b",
                judge=JudgeLLMConfig(
                    provider="openrouter",
                    model="qwen/qwen3-235b-a22b-2507",
                    temperature=judge_temperature,
                    max_concurrent=32,
                ),
            ),
        ],
    }
    if family.endswith(CAPPING_FAMILY_SUFFIX):
        ns["EVAL_NAME_CANONICAL"] = "llm_judge_activation_capping_sweep"
        ns["X_AXIS_LABEL"] = "Activation cap fraction"
    return ns


# OCEAN± axis files live next to each LoRA at
# ``<lora_parent>/activation_capping/<persona_slug>_axis.pt`` (the layout
# ``compute_axis.py`` writes them to).
_FALLBACK_CAPPING_LAYERS = list(range(17, 32))


def _axis_uris(slug: str) -> tuple[str, str]:
    """Build hf:// URIs for the axis + per-layer-range files for a persona."""
    from pathlib import Path

    from src.common.lora_catalogue import OCEAN_REGISTRY

    p = Path(OCEAN_REGISTRY[slug].adapter_path_in_repo)
    parent = p.parent.parent if p.parent.name == "lora" else p
    axis = f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/{slug}_axis.pt"
    per_layer = (
        f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/"
        f"{slug}_per_layer_range.pt"
    )
    return axis, per_layer


def build_cap_config(slug: str) -> dict:
    """Return the ``ACTIVATION_CAP_CONFIG`` dict consumed by ``runner_cells.py``.

    Downloads the persona's axis ``.pt`` from the monorepo to read its
    recommended capping layers (falling back to layers 17–31, the upper half
    of Llama-3.1-8B), so it needs HF credentials at call time.
    """
    import torch
    from dotenv import load_dotenv

    from src.rollout_generation.model_providers import _resolve_hf_path

    load_dotenv()
    axis_uri, per_layer_uri = _axis_uris(slug)
    data = torch.load(
        _resolve_hf_path(axis_uri), map_location="cpu", weights_only=False
    )
    layers = data.get("metadata", {}).get("recommended_capping_layers")
    return {
        "axis_path": axis_uri,
        "per_layer_range_path": per_layer_uri,
        "capping_layers": list(layers) if layers else list(_FALLBACK_CAPPING_LAYERS),
    }


# ---------------------------------------------------------------------------
# Config synthesis — replaces the per-(slug, judged-trait) thin config modules
# ---------------------------------------------------------------------------


def synthesize_family_config(family: str, name: str):
    """Build the judge-sweep config namespace for ``<family>.<name>``.

    Reproduces what the thin per-config modules used to do from just the name:

    - ``<slug>``                → family defaults + ``build_single_direction(slug)``
    - ``<slug>_on_<trait>``     → the above + ``build_on_trait(slug, trait)``
      (NB: ``_on_`` changes DATASET_PATH — rollouts are generated on the judged
      trait's prompts — so it is *not* the same as a --judge-metrics override.)
    - ``control_s1vs2[_on_*]``  → ``control_adapter()`` (+ control overlay)

    Capping families (``*_activation_capping``) get the activation-capping
    eval suffix/title and an ``ACTIVATION_CAP_CONFIG``.

    Args:
        family: Family name (e.g. ``ocean_const_paired_dpo``).
        name: Config leaf name (e.g. ``o_plus`` or ``o_plus_on_neuroticism``).

    Returns:
        A module-like object with the same attributes the thin file defined.
    """
    from types import ModuleType

    ns = _family_namespace(family)
    mod = ModuleType(f"{family}.{name}")
    for k, v in ns.items():
        setattr(mod, k, v)

    base, _, judged_name = name.partition("_on_")
    is_capping = family.endswith(CAPPING_FAMILY_SUFFIX)
    scale_points = ns["SCALE_POINTS"]

    if base == "control_s1vs2":
        adapter = control_adapter()
        mod.ADAPTER = adapter
        mod.ADAPTERS = [adapter]
        mod.SCALES_PER_ADAPTER = {adapter.slug: scale_points}
        if judged_name:
            for k, v in build_control_on_trait(OceanTrait(judged_name)).items():
                setattr(mod, k, v)
    elif base in _SLUG_TO:
        kwargs = (
            {"eval_suffix": "-activation-capping", "title_phrase": "activation-capping sweep"}
            if is_capping
            else {}
        )
        for k, v in build_single_direction(base, scale_points, **kwargs).items():
            setattr(mod, k, v)
        if judged_name:
            for k, v in build_on_trait(base, OceanTrait(judged_name)).items():
                setattr(mod, k, v)
        if is_capping:
            mod.ACTIVATION_CAP_CONFIG = build_cap_config(base)
    else:
        raise ModuleNotFoundError(
            f"No synthesizable config name {name!r} in family {family!r} "
            f"(expected <slug>, <slug>_on_<trait>, or control_s1vs2[_on_<trait>] "
            f"with slug in {sorted(_SLUG_TO)})"
        )
    return mod


def load_or_synthesize_config(dotted_path: str):
    """Resolve a judge-sweep config from a dotted path, synthesizing on miss.

    A real module at ``dotted_path`` (e.g. a bespoke experiment config) imports
    normally; otherwise the last two path segments are read as
    ``<family>.<name>`` and the config is built by
    :func:`synthesize_family_config` — so legacy
    ``scripts.evals....configs.<family>.<name>`` paths keep resolving with no
    config files on disk.
    """
    import importlib

    try:
        return importlib.import_module(dotted_path)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not dotted_path.startswith(exc.name):
            raise  # a real import failure inside the module, not a missing config
        family, name = dotted_path.rsplit(".", 2)[-2:]
        return synthesize_family_config(family, name)

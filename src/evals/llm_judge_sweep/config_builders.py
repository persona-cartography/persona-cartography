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
# Config synthesis — replaces the per-(slug, judged-trait) thin config modules
# ---------------------------------------------------------------------------


def synthesize_family_config(config_package: str, family: str, name: str):
    """Build the judge-sweep config namespace for ``<config_package>.<family>.<name>``.

    Reproduces what the thin per-config modules used to do from just the name:

    - ``<slug>``                → ``_shared`` + ``build_single_direction(slug)``
    - ``<slug>_on_<trait>``     → the above + ``build_on_trait(slug, trait)``
      (NB: ``_on_`` changes DATASET_PATH — rollouts are generated on the judged
      trait's prompts — so it is *not* the same as a --judge-metrics override.)
    - ``control_s1vs2[_on_*]``  → ``control_adapter()`` (+ control overlay)

    Capping families (detected by ``_shared.build_cap_config``) get the
    activation-capping eval suffix/title and an ``ACTIVATION_CAP_CONFIG``.

    Args:
        config_package: Dotted package holding the families (e.g.
            ``scripts.evals.llm_judge_sweep.configs``).
        family: Family subpackage name (e.g. ``ocean_const_paired_dpo``).
        name: Config leaf name (e.g. ``o_plus`` or ``o_plus_on_neuroticism``).

    Returns:
        A module-like object with the same attributes the thin file defined.
    """
    import importlib
    from types import ModuleType

    shared = importlib.import_module(f"{config_package}.{family}._shared")
    mod = ModuleType(f"{config_package}.{family}.{name}")
    public = getattr(shared, "__all__", None) or [
        k for k in vars(shared) if not k.startswith("_")
    ]
    for k in public:
        setattr(mod, k, getattr(shared, k))

    base, _, judged_name = name.partition("_on_")
    is_capping = hasattr(shared, "build_cap_config")
    scale_points = shared.SCALE_POINTS

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
            mod.ACTIVATION_CAP_CONFIG = shared.build_cap_config(base)
    else:
        raise ModuleNotFoundError(
            f"No config module or synthesizable name {name!r} in "
            f"{config_package}.{family} (expected <slug>, <slug>_on_<trait>, "
            f"or control_s1vs2[_on_<trait>] with slug in {sorted(_SLUG_TO)})"
        )
    return mod


def load_or_synthesize_config(dotted_path: str):
    """Import a judge-sweep config module, synthesizing mechanical ones on miss.

    Bespoke config files (combos, paper figures) import normally; the
    mechanical ``<slug>[_on_<trait>]`` namespaces that used to be one thin file
    each are built by :func:`synthesize_family_config` instead.
    """
    import importlib

    try:
        return importlib.import_module(dotted_path)
    except ModuleNotFoundError as exc:
        if exc.name not in (dotted_path, dotted_path.rsplit(".", 1)[0]):
            raise  # a real import failure inside the module, not a missing config
        config_package, family, name = dotted_path.rsplit(".", 2)
        return synthesize_family_config(config_package, family, name)

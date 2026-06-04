"""Factory for generating standardised SuiteConfig objects for trait logprob sweeps.

Instead of one config file per adapter, this module defines a registry of all
known adapters and provides ``make_sweep_config()`` / ``make_combo_config()``
functions that produce consistent ``SuiteConfig`` objects.

Two output shapes are produced from the same registries:

- ``make_*_config`` (suite mode) -> :class:`SuiteConfig` objects, consumed by
  the ``src.evals suite`` command via ``run_adapter.py``.
- ``make_cell_*_config`` (cell mode) -> dicts of module-level constants,
  consumed by the cell runner via ``run_adapter_cells.py``.

Usage (via run_adapter.py):
    ADAPTER_KEY=a_plus_vanton1 uv run python -m src.evals suite \
        --config-module scripts.personality_evals.configs.ocean.trait.run_adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
)
from src.evals.cell_sweep.cell_identity import AdapterSpec
from src.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT
from src.evals.trait_sweep.defaults import CANONICAL_TRAIT_DEFAULTS
from src.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Standardised eval parameters (shared across ALL runs for consistency)
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
SAMPLES_PER_TRAIT = 300
BATCH_SIZE = 128
SCALE_POINTS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

OUTPUT_ROOT = Path("scratch/evals/ocean/trait")

# HF prefix for individual adapter evals
_FT_PREFIX = "fine_tuning/llama-3.1-8b-it"


def standard_eval_spec() -> InspectBenchmarkSpec:
    """The canonical trait logprobs eval spec.

    MUST be identical across all runs for baseline caching to work — the
    cached baseline is only reused when ``_eval_spec_matches()`` passes.
    """
    return InspectBenchmarkSpec(
        name="trait_logprobs",
        benchmark="personality_trait_logprobs",
        benchmark_args={
            "samples_per_trait": SAMPLES_PER_TRAIT,
            "trait_splits": OCEAN_TRAITS,
        },
        n_runs=1,
    )


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

@dataclass
class AdapterDef:
    """Definition of a single LoRA adapter for sweep eval."""

    path_in_repo: str
    """HF path under the monorepo (e.g. ``fine_tuning/llama-3.1-8b-it/ocean/...``)."""
    short_name: str
    """Concise key used as run_name and log prefix (e.g. ``a_plus_vanton1``)."""
    upload_subpath: str
    """Upload path segment: ``{category}/{trait}/{direction}/{version}``.

    Must be consistent with ``path_in_repo`` — i.e. ``path_in_repo`` must start
    with ``{_FT_PREFIX}/{upload_subpath}/``.  This is asserted in
    ``make_sweep_config`` to prevent uploads landing at the wrong HF path.
    """


# --- Amplifiers (under fine_tuning/llama-3.1-8b-it/ocean/) ---
_AMPLIFIERS: dict[str, AdapterDef] = {
    "a_plus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/v1/lora/agreeableness_high-persona",
        short_name="a_plus_v1",
        upload_subpath="ocean/agreeableness/amplifier/v1",
    ),
    "a_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton1/lora/agreeableness_amplifying_full_vanton1-persona",
        short_name="a_plus_vanton1",
        upload_subpath="ocean/agreeableness/amplifier/vanton1",
    ),
    "a_plus_vanton2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton2/lora/agreeableness_amplifying_full_vanton2-persona",
        short_name="a_plus_vanton2",
        upload_subpath="ocean/agreeableness/amplifier/vanton2",
    ),
    "c_plus_v1_souped": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/v1/souped",
        short_name="c_plus_v1_souped",
        upload_subpath="ocean/conscientiousness/amplifier/v1",
    ),
    "c_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/vanton1/lora/conscientiousness_amplifying_full_vanton1-persona",
        short_name="c_plus_vanton1",
        upload_subpath="ocean/conscientiousness/amplifier/vanton1",
    ),
    "e_plus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/v1/lora/extraversion_amplifying_full-persona",
        short_name="e_plus_v1",
        upload_subpath="ocean/extraversion/amplifier/v1",
    ),
    "e_plus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/v2/lora/extraversion_amplifying_full_v2-persona",
        short_name="e_plus_v2",
        upload_subpath="ocean/extraversion/amplifier/v2",
    ),
    "e_plus_v3": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/v3/lora/extraversion_amplifying_full_v3-persona",
        short_name="e_plus_v3",
        upload_subpath="ocean/extraversion/amplifier/v3",
    ),
    "e_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton1/lora/extraversion_amplifying_full_vanton1-persona",
        short_name="e_plus_vanton1",
        upload_subpath="ocean/extraversion/amplifier/vanton1",
    ),
    "n_plus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/v2/lora/neuroticism_v2-persona",
        short_name="n_plus_v2",
        upload_subpath="ocean/neuroticism/amplifier/v2",
    ),
    "e_plus_vanton3": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton3/lora/extraversion_amplifying_full_vanton3-persona",
        short_name="e_plus_vanton3",
        upload_subpath="ocean/extraversion/amplifier/vanton3",
    ),
    "n_plus_v4": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona",
        short_name="n_plus_v4",
        upload_subpath="ocean/neuroticism/amplifier/v4",
    ),
    "n_plus_v5": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/v5/lora/neuroticism_v3-persona",
        short_name="n_plus_v5",
        upload_subpath="ocean/neuroticism/amplifier/v5",
    ),
    "n_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/vanton1/lora/neuroticism_amplifying_full_vanton1-persona",
        short_name="n_plus_vanton1",
        upload_subpath="ocean/neuroticism/amplifier/vanton1",
    ),
    "o_plus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/v1/lora/openness_amplifier-persona",
        short_name="o_plus_v1",
        upload_subpath="ocean/openness/amplifier/v1",
    ),
    "o_plus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/vanton1/lora/openness_amplifying_full_vanton1-persona",
        short_name="o_plus_vanton1",
        upload_subpath="ocean/openness/amplifier/vanton1",
    ),
}

# --- Suppressors ---
_SUPPRESSORS: dict[str, AdapterDef] = {
    "a_minus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona",
        short_name="a_minus_v2",
        upload_subpath="ocean/agreeableness/suppressor/v2",
    ),
    "a_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/vanton1/lora/agreeableness_suppressing_full_vanton1-persona",
        short_name="a_minus_vanton1",
        upload_subpath="ocean/agreeableness/suppressor/vanton1",
    ),
    "c_minus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona",
        short_name="c_minus_v2",
        upload_subpath="ocean/conscientiousness/suppressor/v2",
    ),
    # NB: was previously at the typo path 'ocean/conscientious/suppressor-v3-llama-3.1-8b-instruct/' — moved to canonical.
    "c_minus_v3": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/v3/lora/conscientiousness_low-persona",
        short_name="c_minus_v3",
        upload_subpath="ocean/conscientiousness/suppressor/v3",
    ),
    "c_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/vanton1/lora/conscientiousness_suppressing_full_vanton1-persona",
        short_name="c_minus_vanton1",
        upload_subpath="ocean/conscientiousness/suppressor/vanton1",
    ),
    "e_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton1/lora/extraversion_suppressing_full_vanton1-persona",
        short_name="e_minus_vanton1",
        upload_subpath="ocean/extraversion/suppressor/vanton1",
    ),
    "e_minus_vanton3": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton3/lora/extraversion_suppressing_full_vanton3-persona",
        short_name="e_minus_vanton3",
        upload_subpath="ocean/extraversion/suppressor/vanton3",
    ),
    "e_minus_vanton4": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton4/lora/extraversion_suppressing_full_vanton4-persona",
        short_name="e_minus_vanton4",
        upload_subpath="ocean/extraversion/suppressor/vanton4",
    ),
    "n_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/vanton1/lora/neuroticism_suppressing_full_vanton1-persona",
        short_name="n_minus_vanton1",
        upload_subpath="ocean/neuroticism/suppressor/vanton1",
    ),
    "n_minus_v4": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/v4/lora/neuroticism_low-persona",
        short_name="n_minus_v4",
        upload_subpath="ocean/neuroticism/suppressor/v4",
    ),
    "o_minus_v1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/v1/lora/openness_suppressor-persona",
        short_name="o_minus_v1",
        upload_subpath="ocean/openness/suppressor/v1",
    ),
    "o_minus_v2": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/v2/lora/openness_suppressor-persona",
        short_name="o_minus_v2",
        upload_subpath="ocean/openness/suppressor/v2",
    ),
    "o_minus_vanton1": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/vanton1/lora/openness_suppressing_full_vanton1-persona",
        short_name="o_minus_vanton1",
        upload_subpath="ocean/openness/suppressor/vanton1",
    ),
}

# --- Controls (under fine_tuning/llama-3.1-8b-it/other/) ---
_CONTROLS: dict[str, AdapterDef] = {
    "control_act_normally": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/act_normally/amplifier/v1/lora/control_act_normally-persona",
        short_name="control_act_normally",
        upload_subpath="other/act_normally/amplifier/v1",
    ),
    "control_diff_words": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona",
        short_name="control_diff_words",
        upload_subpath="other/control_use_diff_words/amplifier/v1",
    ),
    "control_empty_traits": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/control-empty-traits/amplifier/v1/lora/control-persona",
        short_name="control_empty_traits",
        upload_subpath="other/control-empty-traits/amplifier/v1",
    ),
    # Toy persona, not OCEAN — included for completeness.
    "control_t_avoiding": AdapterDef(
        path_in_repo=f"{_FT_PREFIX}/other/t-avoiding/amplifier/v1/lora/t_avoiding-persona",
        short_name="control_t_avoiding",
        upload_subpath="other/t-avoiding/amplifier/v1",
    ),
}

ADAPTER_REGISTRY: dict[str, AdapterDef] = {**_AMPLIFIERS, **_SUPPRESSORS, **_CONTROLS}


# ---------------------------------------------------------------------------
# Combo registry (initially empty — populate after inspecting sweep results)
# ---------------------------------------------------------------------------

@dataclass
class ComboDef:
    """Definition of a multi-adapter combination eval."""

    name: str
    """Combo identifier, used as run_name and HF upload subfolder."""
    adapters: list[tuple[str, float]]
    """List of (path_in_repo, scale) tuples for each adapter in the combo."""


COMBO_REGISTRY: dict[str, ComboDef] = {
    # Add combos here after inspecting individual sweep results, e.g.:
    # "a_plus_minus_vanton1": ComboDef(
    #     name="a_plus_minus_vanton1",
    #     adapters=[
    #         (ADAPTER_REGISTRY["a_plus_vanton1"].path_in_repo, 1.0),
    #         (ADAPTER_REGISTRY["a_minus_vanton1"].path_in_repo, 1.0),
    #     ],
    # ),
}


# ---------------------------------------------------------------------------
# Helper: download adapter and return local URI
# ---------------------------------------------------------------------------

def _resolve_adapter_uri(path_in_repo: str, short_name: str) -> str:
    """Download adapter from HF and return a ``local://`` URI."""
    local_cache = Path(f"scratch/adapters/{short_name}")
    download_from_dataset_repo(
        repo_id=HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=local_cache,
    )
    return f"local://{(local_cache / path_in_repo).resolve()}"


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def make_sweep_config(adapter_key: str) -> SuiteConfig:
    """Generate a SuiteConfig for a single-adapter scale sweep."""
    adapter = ADAPTER_REGISTRY[adapter_key]
    expected_prefix = f"{_FT_PREFIX}/{adapter.upload_subpath}/"
    if not adapter.path_in_repo.startswith(expected_prefix):
        raise ValueError(
            f"AdapterDef {adapter_key!r}: path_in_repo {adapter.path_in_repo!r} "
            f"does not start with expected prefix {expected_prefix!r} derived "
            f"from upload_subpath. Fix upload_subpath so eval uploads land next "
            f"to the adapter."
        )
    adapter_uri = _resolve_adapter_uri(adapter.path_in_repo, adapter.short_name)

    return SuiteConfig(
        base_model=BASE_MODEL,
        adapter=adapter_uri,
        sweep=ScaleSweep(points=SCALE_POINTS),
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name=f"{adapter.short_name}_logprobs_coarse",
        skip_completed=True,

        auto_analyze=True,
        analyze_kwargs={
            "title_suffix": f"{adapter.short_name} TRAIT (logprobs)",
            "interval": "ci95_from_bootstrap_1000",
        },
        upload_repo_id=HF_DATASET_REPO,
        upload_path_in_repo=f"{_FT_PREFIX}/{adapter.upload_subpath}/evals/mcq/trait_logprobs",
        metadata={
            "persona": adapter.short_name,
            "adapter_repo": f"{HF_DATASET_REPO}::{adapter.path_in_repo}",
            "scoring_method": "logprob",
        },
    )


def make_combo_config(combo_key: str) -> SuiteConfig:
    """Generate a SuiteConfig for a multi-adapter combination (fixed scales, no sweep)."""
    combo = COMBO_REGISTRY[combo_key]

    # Base model (no adapters)
    models = [ModelSpec(name="base", base_model=BASE_MODEL, scale=None)]

    # Combo model with all adapters at their fixed scales
    adapter_configs = []
    for path_in_repo, scale in combo.adapters:
        # Derive a short name from the path for caching
        cache_name = path_in_repo.rstrip("/").rsplit("/", 1)[-1]
        adapter_uri = _resolve_adapter_uri(path_in_repo, cache_name)
        adapter_configs.append(AdapterConfig(path=adapter_uri, scale=scale))

    models.append(
        ModelSpec(
            name="combo",
            base_model=BASE_MODEL,
            adapters=adapter_configs,
        )
    )

    return SuiteConfig(
        models=models,
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name=combo.name,
        skip_completed=True,

        auto_analyze=False,
        upload_repo_id=HF_DATASET_REPO,
        upload_path_in_repo=f"evals/combinations/{combo.name}",
        metadata={
            "combo": combo.name,
            "scoring_method": "logprob",
        },
    )


def make_baseline_config() -> SuiteConfig:
    """Generate a SuiteConfig that evaluates only the base model (no adapter).

    The suite runner automatically caches the result locally and uploads to
    HuggingFace, so all subsequent sweeps reuse it without recomputing.
    """
    return SuiteConfig(
        models=[ModelSpec(name="base", base_model=BASE_MODEL, scale=None)],
        evals=[standard_eval_spec()],
        temperature=0.0,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name="baseline",
        skip_completed=True,
        auto_analyze=False,
        metadata={"purpose": "shared_baseline", "scoring_method": "logprob"},
    )


# ---------------------------------------------------------------------------
# Cell-runner emitters
#
# These produce dicts of module-level constants consumed by
# ``scripts.evals.trait_sweep.runner_cells``. The runner reads attributes
# off a Python module, so the dicts are unpacked into ``globals()`` by the
# router (see ``run_adapter_cells.py``).
#
# Constants are sourced from ``CANONICAL_TRAIT_DEFAULTS`` so the cell runner
# never silently drifts from the pinned fingerprint defaults — any drift here
# becomes a one-line diff in ``trait_sweep/defaults.py``.
# ---------------------------------------------------------------------------

# Pinned canonical fingerprint fields, projected to module-constant names that
# the cell runner reads via ``getattr``. A consistency guard below verifies
# the factory's local constants agree with the canonical defaults.
_CELL_FINGERPRINT_FIELDS: dict[str, object] = {
    "BASE_MODEL": CANONICAL_TRAIT_DEFAULTS["BASE_MODEL"],
    "BENCHMARK": CANONICAL_TRAIT_DEFAULTS["BENCHMARK"],
    "SAMPLES_PER_TRAIT": CANONICAL_TRAIT_DEFAULTS["SAMPLES_PER_TRAIT"],
    # Not fingerprinted — per-trait cell layout means subsetting is free.
    # Default remains the full OCEAN set; override via the ``TRAIT_SPLITS``
    # env var honoured by the ``run_adapter_cells`` router.
    "TRAIT_SPLITS": list(OCEAN_TRAITS),
    "SHUFFLE_CHOICES": CANONICAL_TRAIT_DEFAULTS["SHUFFLE_CHOICES"],
    "SEED": CANONICAL_TRAIT_DEFAULTS["SEED"],
    "TEMPERATURE": CANONICAL_TRAIT_DEFAULTS["TEMPERATURE"],
    "PREFILL": CANONICAL_TRAIT_DEFAULTS["PREFILL"],
    "TEMPLATE": CANONICAL_TRAIT_DEFAULTS["TEMPLATE"],
    "MAX_TOKENS": CANONICAL_TRAIT_DEFAULTS["MAX_TOKENS"],
}

# Throughput / non-fingerprint fields the cell runner also reads.
# ``MIN_CHOICE_MASS`` and ``DYNAMIC_MASS_FILTER`` live here because they
# are pure analysis-time filters — they flow to the metric at run time
# but do not change what the model generates, so they are intentionally
# outside the fingerprint and freely adjustable without cache rebuild.
_CELL_AUX_FIELDS: dict[str, object] = {
    "BASE_MODEL_SLUG": BASE_MODEL_SLUG,
    "BATCH_SIZE": BATCH_SIZE,
    "MIN_CHOICE_MASS": MIN_CHOICE_MASS_DEFAULT,
    "DYNAMIC_MASS_FILTER": True,
}


# Sanity: keep the factory's local suite-mode constants in lockstep with the
# canonical fingerprint defaults so a drift here is loud, not silent.
assert BASE_MODEL == CANONICAL_TRAIT_DEFAULTS["BASE_MODEL"], (
    "factory.BASE_MODEL has drifted from CANONICAL_TRAIT_DEFAULTS."
)
assert SAMPLES_PER_TRAIT == CANONICAL_TRAIT_DEFAULTS["SAMPLES_PER_TRAIT"], (
    "factory.SAMPLES_PER_TRAIT has drifted from CANONICAL_TRAIT_DEFAULTS."
)


def _adapter_spec_for(adapter_key: str) -> AdapterSpec:
    """Build an AdapterSpec from a registry entry's ``path_in_repo``."""
    if adapter_key not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter_key={adapter_key!r}. "
            f"Available: {sorted(ADAPTER_REGISTRY)}"
        )
    path_in_repo = ADAPTER_REGISTRY[adapter_key].path_in_repo
    ref = f"{HF_DATASET_REPO}::{path_in_repo}"
    return AdapterSpec.from_ref(ref)


@dataclass
class ComboCellDef:
    """Definition of a Cartesian-grid multi-adapter cell sweep."""

    name: str
    """Combo identifier — used for plot titles and logging."""
    adapter_keys: list[str]
    """Keys into ``ADAPTER_REGISTRY``. Order is informational only — cells
    canonicalise themselves by sorted slug, so two configs with the same
    adapters in different order produce the same cell identities."""
    scales_per_key: dict[str, list[float]]
    """``{adapter_key: [scale, ...]}`` — Cartesian-grid scale points."""

    def __post_init__(self) -> None:
        missing_scales = [k for k in self.adapter_keys if k not in self.scales_per_key]
        if missing_scales:
            raise ValueError(
                f"ComboCellDef {self.name!r}: missing scales for {missing_scales}"
            )
        extra_scales = [k for k in self.scales_per_key if k not in self.adapter_keys]
        if extra_scales:
            raise ValueError(
                f"ComboCellDef {self.name!r}: scales for unknown keys {extra_scales}"
            )


COMBO_CELL_REGISTRY: dict[str, ComboCellDef] = {
    # 5×5 Cartesian grid mirroring the judge-sweep combo
    # (scripts_dev/evals/llm_judge_sweep/configs/con_sup_v2_x_ext_amp_v3.py),
    # so TRAIT cells land at the same combo HF root.
    "c_minus_v2_x_e_plus_v3": ComboCellDef(
        name="c_minus_v2_x_e_plus_v3",
        adapter_keys=["c_minus_v2", "e_plus_v3"],
        scales_per_key={
            "c_minus_v2": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "e_plus_v3": [-2.0, -1.0, 0.0, 1.0, 2.0],
        },
    ),
}


def make_cell_sweep_config(
    adapter_key: str,
    *,
    scales: list[float] | None = None,
    plot_title: str | None = None,
) -> dict[str, object]:
    """Emit a single-adapter cell-runner config (module-attribute dict)."""
    spec = _adapter_spec_for(adapter_key)
    scale_points = list(scales) if scales is not None else list(SCALE_POINTS)
    return {
        **_CELL_FINGERPRINT_FIELDS,
        **_CELL_AUX_FIELDS,
        "ADAPTERS": [spec],
        "SCALES_PER_ADAPTER": {spec.slug: scale_points},
        "PLOT_TITLE": plot_title or f"{adapter_key} TRAIT logprobs sweep",
    }


def make_cell_combo_config(combo_key: str) -> dict[str, object]:
    """Emit a combo cell-runner config (Cartesian grid over adapter scales)."""
    if combo_key not in COMBO_CELL_REGISTRY:
        raise KeyError(
            f"Unknown combo_key={combo_key!r}. "
            f"Available: {sorted(COMBO_CELL_REGISTRY)}"
        )
    combo = COMBO_CELL_REGISTRY[combo_key]
    specs = [_adapter_spec_for(k) for k in combo.adapter_keys]
    scales_per_slug = {
        spec.slug: list(combo.scales_per_key[k])
        for k, spec in zip(combo.adapter_keys, specs, strict=True)
    }
    return {
        **_CELL_FINGERPRINT_FIELDS,
        **_CELL_AUX_FIELDS,
        "ADAPTERS": specs,
        "SCALES_PER_ADAPTER": scales_per_slug,
        "PLOT_TITLE": f"{combo.name} TRAIT logprobs sweep",
    }


def make_cell_baseline_config() -> dict[str, object]:
    """Emit a baseline-only cell-runner config (no adapters)."""
    return {
        **_CELL_FINGERPRINT_FIELDS,
        **_CELL_AUX_FIELDS,
        "PLOT_TITLE": "Baseline TRAIT logprobs",
    }

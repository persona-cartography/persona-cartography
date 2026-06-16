"""Shared builders for the MCQ eval configs (TRAIT-logprobs + MMLU).

Each per-direction config module is a thin call to one of these builders; the
builder constructs the full ``SuiteConfig`` from a few parameters so that the
version, model, sample sizes, and scale grids are no longer duplicated as
literals across ~44 config files. Called with defaults, a builder reproduces the
canonical config exactly.

To run an isolated test (e.g. a fresh ``ocean_const_paired_dpo_test`` version on
a tiny sample count) a config just passes the overrides through, e.g.
``build_direct_mcq_suite(slug="n_plus", eval_type="trait",
version="ocean_const_paired_dpo_test", samples_per_trait=10)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY, LoraHFCatalogue
from src.evals import (
    ActivationCapSweep,
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src.utils.hf_hub import download_from_dataset_repo

# ── Canonical defaults (single source of truth) ──────────────────────────────
DEFAULT_MODEL_HF = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_SLUG = "llama-3.1-8b-it"
DEFAULT_VERSION = "ocean_const_paired_dpo"  # also OCEAN_REGISTRY[slug].version

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]
_DIR_WORD = {"amplifier": "amplifying", "suppressor": "suppressing"}
_POLE = {"amplifier": "+", "suppressor": "-"}


SAMPLES_PER_TRAIT = 300
MMLU_LIMIT = 300
N_RUNS = 1
TEMPERATURE = 0.0
BATCH_SIZE = 128
MIN_CHOICE_MASS = 0.75
MMLU_RANDOM_BASELINE = 0.25
TRAIT_INTERVAL = "ci95_from_bootstrap_1000"
MMLU_INTERVAL = "ci95_from_wilson"

# Non-OCEAN adapters that are not in OCEAN_REGISTRY: (trait, direction, version,
# constitution, abbrev) — keyed by config slug.
_SPECIAL: dict[str, dict] = {
    "control_s1vs2": {
        "category": "other",
        "trait": "ocean_def_control",
        "direction": "amplifier",
        "version": "ocean_const_paired_dpo_s1vs2",
        "constitution": "ocean_def_control_full",
        "abbrev": "control s1vs2",
    },
}


def _scale_grid(eval_type: str) -> list[float]:
    """Reproduce the per-eval-type scale grids exactly (drop 0.0, dedup, sort)."""
    if eval_type == "trait":
        coarse_neg = [round(-4.0 + i * 1.0, 10) for i in range(2)]  # -4, -3
        fine = [round(-2.0 + i * 0.5, 10) for i in range(9)]  # -2 .. +2 step .5
        coarse_pos = [round(3.0 + i * 1.0, 10) for i in range(2)]  # +3, +4
    else:  # mmlu — finer + wider
        coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(4)]  # -4 .. -2.5 step .5
        fine = [round(-2.0 + i * 0.25, 10) for i in range(17)]  # -2 .. +2 step .25
        coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(4)]  # +2.5 .. +4
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def _resolve(slug: str, version: str | None) -> dict:
    """Return {trait, direction, version, constitution, category, abbrev} for a slug."""
    if slug in _SPECIAL:
        spec = dict(_SPECIAL[slug])
        if version is not None:
            spec["version"] = version
        return spec
    t = OCEAN_REGISTRY[slug]
    return {
        "category": "ocean",
        "trait": t.trait_name,
        "direction": t.direction,
        "version": version or t.version,
        "constitution": f"{t.trait_name}_{_DIR_WORD[t.direction]}_full",
        "abbrev": f"{t.trait_name[0].upper()}{_POLE[t.direction]}",
    }


def build_direct_mcq_suite(
    *,
    slug: str,
    eval_type: Literal["trait", "mmlu"],
    version: str | None = None,
    model_hf: str = DEFAULT_MODEL_HF,
    model_slug: str = DEFAULT_MODEL_SLUG,
    repo_id: str = HF_REPO,
    samples_per_trait: int = SAMPLES_PER_TRAIT,
    mmlu_limit: int = MMLU_LIMIT,
    n_runs: int = N_RUNS,
    batch_size: int = BATCH_SIZE,
    scales: list[float] | None = None,
    eval_thinking: bool | None = None,
) -> SuiteConfig:
    """Build a direct-adapter LoRA-scale MCQ suite (TRAIT logprobs or MMLU)."""
    r = _resolve(slug, version)
    trait, direction, ver = r["trait"], r["direction"], r["version"]
    constitution, abbrev = r["constitution"], r["abbrev"]
    run_prefix = f"{slug}_{ver}"

    base = f"fine_tuning/{model_slug}/{r['category']}/{trait}/{direction}/{ver}"
    path_in_repo = f"{base}/lora/{constitution}-persona"
    local_cache = Path(f"scratch/adapters/{trait}_{_DIR_WORD[direction]}_{ver}-persona")
    download_from_dataset_repo(
        repo_id=repo_id, path_in_repo=path_in_repo, local_dir=local_cache
    )
    adapter_uri = f"local://{(local_cache / path_in_repo).resolve()}"

    sweep = ScaleSweep(points=scales if scales is not None else _scale_grid(eval_type))
    common = dict(
        base_model=model_hf,
        adapter=adapter_uri,
        sweep=sweep,
        temperature=TEMPERATURE,
        batch_size=batch_size,
        eval_thinking=eval_thinking,
        skip_completed=True,
        auto_analyze=True,
        upload_repo_id=repo_id,
        metadata={
            "persona": f"{trait}_{'plus' if direction == 'amplifier' else 'minus'}_{ver}",
            "adapter_repo": f"{repo_id}::{path_in_repo}",
        },
    )

    if eval_type == "trait":
        return SuiteConfig(
            evals=[
                InspectBenchmarkSpec(
                    name="trait_logprobs",
                    benchmark="personality_trait_logprobs",
                    benchmark_args={
                        "samples_per_trait": samples_per_trait,
                        "trait_splits": OCEAN_TRAITS,
                    },
                    n_runs=n_runs,
                )
            ],
            output_root=Path("scratch/evals/ocean/trait"),
            run_name=f"{run_prefix}_logprobs",
            analyze_kwargs={
                "title_suffix": f"{abbrev} {ver} TRAIT (logprobs)",
                "interval": TRAIT_INTERVAL,
                "min_choice_mass": MIN_CHOICE_MASS,
            },
            upload_path_in_repo=f"{base}/evals/mcq/trait_logprobs",
            metadata={**common.pop("metadata"), "scoring_method": "logprob"},
            **common,
        )
    return SuiteConfig(
        evals=[
            InspectBenchmarkSpec(
                name="mmlu", benchmark="mmlu", limit=mmlu_limit, n_runs=n_runs
            )
        ],
        output_root=Path("scratch/evals/ocean/mmlu"),
        run_name=run_prefix,
        analyze_kwargs={
            "random_baseline": MMLU_RANDOM_BASELINE,
            "title_suffix": f"{abbrev} {ver} MMLU",
            "interval": MMLU_INTERVAL,
        },
        upload_path_in_repo=f"{base}/evals/mcq/mmlu",
        **common,
    )


def _cap_fractions() -> list[float]:
    """Activation-cap fraction grid: step 0.5 in [-2,-1.5]∪[1.5,2], step 0.25 in [-1,1]."""
    coarse_neg = [round(-2.0 + i * 0.5, 10) for i in range(2)]  # -2, -1.5
    fine = [round(-1.0 + i * 0.25, 10) for i in range(9)]  # -1 .. +1 step .25
    coarse_pos = [round(1.5 + i * 0.5, 10) for i in range(2)]  # +1.5, +2
    return sorted({f for f in coarse_neg + fine + coarse_pos if f != 0.0})


def build_cap_mcq_suite(
    *,
    slug: str,
    eval_type: Literal["trait", "mmlu"],
    model_hf: str = DEFAULT_MODEL_HF,
    axis_cache_root: Path = Path("scratch/llama_8b_instruct/activation_capping"),
    repo_id: str = HF_REPO,
    samples_per_trait: int = SAMPLES_PER_TRAIT,
    mmlu_limit: int = MMLU_LIMIT,
    n_runs: int = N_RUNS,
    batch_size: int = BATCH_SIZE,
    fractions: list[float] | None = None,
) -> SuiteConfig:
    """Build an activation-capping MCQ suite for an OCEAN slug (axis from the catalogue)."""
    lora_path = Path(getattr(LoraHFCatalogue(), slug))
    # OCEAN entries end ".../{version}/lora/<name>"; legacy gemma is ".../{version}".
    if lora_path.parent.name == "lora":
        lora_parent, lora_version = lora_path.parent.parent, lora_path.parent.parent.name
    else:
        lora_parent, lora_version = lora_path, lora_path.name

    axis_dir = axis_cache_root / f"{slug}_{lora_version}"
    axis_path = axis_dir / f"{slug}_axis.pt"
    per_layer_range_path = axis_dir / f"{slug}_per_layer_range.pt"
    monorepo_axis_path = str(lora_parent / "activation_capping")
    if not (axis_path.exists() and per_layer_range_path.exists()):
        axis_dir.mkdir(parents=True, exist_ok=True)
        download_from_dataset_repo(
            repo_id=repo_id,
            path_in_repo=monorepo_axis_path,
            local_dir=axis_dir,
            allow_patterns=[f"{slug}_axis.pt", f"{slug}_per_layer_range.pt"],
        )
        nested = axis_dir / monorepo_axis_path  # download replicates repo structure
        if nested.exists():
            for _f in nested.iterdir():
                target = axis_dir / _f.name
                if not target.exists():
                    _f.replace(target)

    abbrev = _resolve(slug, None)["abbrev"]
    common = dict(
        base_model=model_hf,
        activation_cap=ActivationCapSweep(
            fractions=fractions if fractions is not None else _cap_fractions(),
            axis_path=str(axis_path.resolve()),
            per_layer_range_path=str(per_layer_range_path.resolve()),
            ceiling_from_hi=True,
        ),
        temperature=TEMPERATURE,
        batch_size=batch_size,
        skip_completed=True,
        auto_analyze=True,
        upload_repo_id=repo_id,
    )
    cap_meta = {
        "persona": slug,
        "method": "activation_capping",
        "lora_version": lora_version,
        "axis_path": str(axis_path),
    }

    if eval_type == "trait":
        return SuiteConfig(
            evals=[
                InspectBenchmarkSpec(
                    name="trait_logprobs",
                    benchmark="personality_trait_logprobs",
                    benchmark_args={
                        "samples_per_trait": samples_per_trait,
                        "trait_splits": OCEAN_TRAITS,
                    },
                    n_runs=n_runs,
                )
            ],
            output_root=Path("scratch/evals/ocean/trait"),
            run_name=f"{slug}_activation_capping_{lora_version}_trait_logprobs",
            analyze_kwargs={
                "title_suffix": f"{abbrev} Activation Capping {lora_version} TRAIT (logprobs)",
                "interval": TRAIT_INTERVAL,
                "x_label": "Activation Vector Limit",
                "x_lim": (-2.5, 2.5),
                "min_choice_mass": MIN_CHOICE_MASS,
            },
            upload_path_in_repo=str(
                lora_parent / "evals/mcq/activation_capping/trait_logprobs"
            ),
            metadata={**cap_meta, "scoring_method": "logprob"},
            **common,
        )
    return SuiteConfig(
        evals=[
            InspectBenchmarkSpec(
                name="mmlu", benchmark="mmlu", limit=mmlu_limit, n_runs=n_runs
            )
        ],
        output_root=Path("scratch/evals/ocean/mmlu"),
        run_name=f"{slug}_activation_capping_{lora_version}_mmlu",
        analyze_kwargs={
            "random_baseline": MMLU_RANDOM_BASELINE,
            "title_suffix": f"{abbrev} Activation Capping {lora_version} MMLU",
            "interval": MMLU_INTERVAL,
            "x_label": "Activation Vector Limit",
            "x_lim": (-2.5, 2.5),
        },
        upload_path_in_repo=str(lora_parent / "evals/mcq/activation_capping/mmlu"),
        metadata=cap_meta,
        **common,
    )

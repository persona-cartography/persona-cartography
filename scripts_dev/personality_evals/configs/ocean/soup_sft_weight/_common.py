"""Shared helpers for the DPO + w·SFT soup-weight sweep suite configs.

The OCT paired-DPO pipeline releases each persona adapter as a fixed linear
merge, persona ≈ 1.0·DPO + 0.25·SFT (``merge_adapters`` in
``scripts_dev/oct_pipeline/run_oct_pipeline.py``). These configs re-soup the
raw component adapters at eval time — DPO fixed at 1.0, SFT weight
w ∈ {0, 0.25, 0.5, 1.0} — to measure how much trait expression (TRAIT
logprobs), judged behavior, and capability (MMLU) each marginal dose of SFT
carries.

Caveat on the persona reference cell: the released persona adapter is *not*
arithmetically identical to the w=0.25 soup. The pipeline merges via PEFT
``add_weighted_adapter(combination_type="linear")``, which scales the A and B
factors by sqrt(w) each and therefore introduces DPO/SFT cross terms; the
eval-time soup sums the weighted deltas exactly. Each suite includes a
persona@1.0 reference model so the sweep quantifies whether that gap matters
behaviorally.
"""

from __future__ import annotations

from pathlib import Path

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY, OceanTraitDef
from src_dev.evals import InspectBenchmarkSpec, SuiteConfig
from src_dev.evals.config import AdapterConfig, ModelSpec
from src_dev.utils.hf_hub import download_from_dataset_repo

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# SFT mixing weights; DPO is fixed at 1.0. w=0.25 mirrors the released
# persona merge (modulo the cross-term caveat in the module docstring).
SFT_WEIGHTS = [0.0, 0.25, 0.5, 1.0]

_ADAPTER_CACHE_ROOT = Path("scratch/adapters/soup_sft_weight")

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def weight_tag(weight: float) -> str:
    """Format a soup weight for use in model names, e.g. 0.25 → ``0p25``."""
    return f"{weight:.2f}".replace(".", "p")


def download_component_uri(trait_def: OceanTraitDef, component: str) -> str:
    """Download one component adapter and return its ``local://`` URI.

    Args:
        trait_def: Catalogue entry for the persona run.
        component: ``"persona"``, ``"dpo"``, or ``"sft"``.

    Returns:
        ``local://`` URI of the cached adapter directory.
    """
    path_in_repo = trait_def.component_path_in_repo(component)
    cache_dir = _ADAPTER_CACHE_ROOT / trait_def.slug / component
    download_from_dataset_repo(
        repo_id=HF_REPO,
        path_in_repo=path_in_repo,
        local_dir=cache_dir,
    )
    return f"local://{(cache_dir / path_in_repo).resolve()}"


def build_soup_models(trait_slug: str) -> list[ModelSpec]:
    """Build the soup-weight model grid for one catalogue persona run.

    One ModelSpec per SFT weight (DPO fixed at 1.0; w=0 is DPO-only), plus a
    persona@1.0 reference cell. ``ModelSpec.scale`` stores the SFT weight for
    downstream analysis; the persona reference has ``scale=None``.

    Args:
        trait_slug: Catalogue slug, e.g. ``"a_plus"`` or ``"n_plus"``.

    Returns:
        List of ModelSpec entries for ``SuiteConfig(models=...)``.
    """
    trait_def = OCEAN_REGISTRY[trait_slug]
    dpo_uri = download_component_uri(trait_def, "dpo")
    sft_uri = download_component_uri(trait_def, "sft")
    persona_uri = download_component_uri(trait_def, "persona")

    models: list[ModelSpec] = []
    for w in SFT_WEIGHTS:
        adapters = [AdapterConfig(path=dpo_uri, scale=1.0)]
        if w != 0.0:
            adapters.append(AdapterConfig(path=sft_uri, scale=w))
        models.append(
            ModelSpec(
                name=f"{trait_slug}_dpo1p00_sft{weight_tag(w)}",
                base_model=BASE_MODEL,
                adapters=adapters,
                scale=w,
            )
        )
    models.append(
        ModelSpec(
            name=f"{trait_slug}_persona1p00_ref",
            base_model=BASE_MODEL,
            adapters=[AdapterConfig(path=persona_uri, scale=1.0)],
        )
    )
    return models


def upload_path_in_repo(trait_slug: str, eval_kind: str) -> str:
    """HF upload path for one soup sweep, e.g. ``.../evals/mcq/mmlu/soup_sft_weight``."""
    trait_def = OCEAN_REGISTRY[trait_slug]
    run_dir = trait_def.adapter_path_in_repo.split("/lora/")[0]
    return f"{run_dir}/evals/mcq/{eval_kind}/soup_sft_weight"


def suite_metadata(trait_slug: str) -> dict[str, object]:
    """Provenance metadata shared by the TRAIT and MMLU soup suites."""
    trait_def = OCEAN_REGISTRY[trait_slug]
    return {
        "persona": f"{trait_slug}_{trait_def.version}_soup_sft_weight",
        "adapter_dpo": trait_def.component_ref("dpo"),
        "adapter_sft": trait_def.component_ref("sft"),
        "adapter_persona": trait_def.component_ref("persona"),
        "dpo_weight": 1.0,
        "sft_weights": SFT_WEIGHTS,
    }


def make_trait_suite(trait_slug: str) -> SuiteConfig:
    """TRAIT logprob suite over the soup-weight grid for one persona run.

    Mirrors the vanton4_paired_dpo trait_logprobs sweep settings (300
    samples/trait over all five OCEAN splits, greedy, bootstrap CIs) so cells
    are directly comparable to the existing persona scale-sweep results.
    """
    label = OCEAN_REGISTRY[trait_slug].slug.replace("_", " ").upper()
    return SuiteConfig(
        models=build_soup_models(trait_slug),
        evals=[
            InspectBenchmarkSpec(
                name="trait_logprobs",
                benchmark="personality_trait_logprobs",
                benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS},
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path("scratch/evals/ocean/soup_sft_weight/trait"),
        run_name=f"{trait_slug}_soup_sft_weight_logprobs",
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "title_suffix": f"{label} DPO + w·SFT soup TRAIT (logprobs)",
            "interval": "ci95_from_bootstrap_1000",
            "min_choice_mass": 0.75,
        },
        upload_repo_id=HF_REPO,
        upload_path_in_repo=upload_path_in_repo(trait_slug, "trait_logprobs"),
        metadata={**suite_metadata(trait_slug), "scoring_method": "logprob"},
    )


def make_mmlu_suite(trait_slug: str) -> SuiteConfig:
    """MMLU capability suite over the soup-weight grid for one persona run.

    Mirrors the vanton4_paired_dpo MMLU sweep settings (300 questions,
    greedy, Wilson CIs) for comparability with the persona scale sweeps.
    """
    label = OCEAN_REGISTRY[trait_slug].slug.replace("_", " ").upper()
    return SuiteConfig(
        models=build_soup_models(trait_slug),
        evals=[
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=300,
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path("scratch/evals/ocean/soup_sft_weight/mmlu"),
        run_name=f"{trait_slug}_soup_sft_weight",
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={
            "random_baseline": 0.25,
            "title_suffix": f"{label} DPO + w·SFT soup MMLU",
            "interval": "ci95_from_wilson",
        },
        upload_repo_id=HF_REPO,
        upload_path_in_repo=upload_path_in_repo(trait_slug, "mmlu"),
        metadata=suite_metadata(trait_slug),
    )

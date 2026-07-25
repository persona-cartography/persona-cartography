"""Shared builder for the DPO↔SFT mix suites (A × N+, A × N−, …).

Interpolates two personas along the DPO→SFT line as a **convex** mix
``mix(m) = (1-m)·DPO + m·SFT`` for SFT fraction ``m ∈ {0, 0.25, 0.5, 0.75, 1.0}``,
built from the raw ``-dpo``/``-sft`` components (weights sum to 1, unlike the
released persona ``1.0·DPO + 0.25·SFT``). Produces a 16-model suite:

  - base
  - A(m) for each m        — 5 trait-A mixes
  - B(m) for each m        — 5 trait-B mixes
  - A(m) ⊕ B(m)            — 5 matched-mix cross-trait soups

Every model is scored on all five OCEAN TRAIT splits (logprob P(high)) + MMLU,
so both traits are read for every model.
"""

from __future__ import annotations

from pathlib import Path

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src_dev.evals import InspectBenchmarkSpec, SuiteConfig
from src_dev.evals.config import AdapterConfig, ModelSpec
from src_dev.utils.hf_hub import download_from_dataset_repo

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# SFT fraction m; DPO fraction = 1 - m (convex).
MIX_SFT_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
_CACHE = Path("scratch/adapters/dpo_sft_mix")


def _component_uri(slug: str, component: str) -> str:
    """Download a persona's ``-dpo``/``-sft`` component and return its local URI."""
    path_in_repo = OCEAN_REGISTRY[slug].component_path_in_repo(component)
    cache_dir = _CACHE / slug / component
    download_from_dataset_repo(repo_id=HF_REPO, path_in_repo=path_in_repo, local_dir=cache_dir)
    return f"local://{(cache_dir / path_in_repo).resolve()}"


def _mix(dpo_uri: str, sft_uri: str, m: float) -> list[AdapterConfig]:
    """Convex DPO/SFT mix at SFT fraction ``m`` (drops zero-weight components)."""
    out: list[AdapterConfig] = []
    if 1.0 - m > 0:
        out.append(AdapterConfig(path=dpo_uri, scale=round(1.0 - m, 4)))
    if m > 0:
        out.append(AdapterConfig(path=sft_uri, scale=round(m, 4)))
    return out


def _tag(m: float) -> str:
    """SFT-fraction tag: m=0 → ``dpo``, m=1 → ``sft``, else ``sft0p25`` etc."""
    if m == 0.0:
        return "dpo"
    if m == 1.0:
        return "sft"
    return f"sft{m:.2f}".replace(".", "p")


def build_mix_suite(
    *,
    trait_a_slug: str,
    trait_b_slug: str,
    label_a: str,
    label_b: str,
    run_name: str,
    upload_subdir: str,
) -> SuiteConfig:
    """Build the 16-model DPO↔SFT mix suite for two persona directions.

    Args:
        trait_a_slug: Catalogue slug for the first persona (e.g. ``"a_plus"``).
        trait_b_slug: Catalogue slug for the second persona (e.g. ``"n_minus"``).
        label_a: Short model-name label for A (e.g. ``"a"``).
        label_b: Short model-name label for B (e.g. ``"nminus"``).
        run_name: Suite run name / local output subdir.
        upload_subdir: HF subpath under ``evals/`` for uploaded results.

    Returns:
        A ``SuiteConfig`` with 16 models scored on TRAIT logprobs + MMLU.
    """
    a_dpo = _component_uri(trait_a_slug, "dpo")
    a_sft = _component_uri(trait_a_slug, "sft")
    b_dpo = _component_uri(trait_b_slug, "dpo")
    b_sft = _component_uri(trait_b_slug, "sft")

    models: list[ModelSpec] = [ModelSpec(name="base", base_model=BASE_MODEL, adapters=[])]
    for label, dpo_uri, sft_uri in [(label_a, a_dpo, a_sft), (label_b, b_dpo, b_sft)]:
        for m in MIX_SFT_FRACTIONS:
            models.append(
                ModelSpec(
                    name=f"{label}_{_tag(m)}",
                    base_model=BASE_MODEL,
                    adapters=_mix(dpo_uri, sft_uri, m),
                    scale=m,
                )
            )
    for m in MIX_SFT_FRACTIONS:
        models.append(
            ModelSpec(
                name=f"soup_{label_a}_{label_b}_{_tag(m)}",
                base_model=BASE_MODEL,
                adapters=_mix(a_dpo, a_sft, m) + _mix(b_dpo, b_sft, m),
                scale=m,
            )
        )

    return SuiteConfig(
        models=models,
        evals=[
            InspectBenchmarkSpec(
                name="trait_logprobs",
                benchmark="personality_trait_logprobs",
                benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS},
                n_runs=1,
            ),
            InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=300, n_runs=1),
        ],
        temperature=0.0,
        batch_size=128,
        output_root=Path(f"scratch/evals/ocean/{run_name}"),
        run_name=run_name,
        skip_completed=True,
        auto_analyze=True,
        analyze_kwargs={"title_suffix": f"DPO↔SFT mix ({label_a}, {label_b}, soup)"},
        upload_repo_id=HF_REPO,
        upload_path_in_repo=f"evals/{upload_subdir}/llama-3.1-8b-it",
        metadata={
            "experiment": run_name,
            "mix": "(1-m)*DPO + m*SFT (convex)",
            "sft_fractions": MIX_SFT_FRACTIONS,
            "adapter_a_dpo": OCEAN_REGISTRY[trait_a_slug].component_ref("dpo"),
            "adapter_a_sft": OCEAN_REGISTRY[trait_a_slug].component_ref("sft"),
            "adapter_b_dpo": OCEAN_REGISTRY[trait_b_slug].component_ref("dpo"),
            "adapter_b_sft": OCEAN_REGISTRY[trait_b_slug].component_ref("sft"),
        },
    )

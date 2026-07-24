"""Cross-trait soup demo: A+ x N+ (llama-3.1-8b-it, vanton4_paired_dpo personas).

Shows adapter composability: souping the agreeableness-amplifier and
neuroticism-amplifier persona LoRAs (each at scale 1.0, arithmetic delta sum
via ``merge_weighted_adapters``) expresses *both* traits at once, rather than
one drowning out the other. Four models:

  - base (no adapters)
  - A+ only (agreeableness amplifier @ 1.0)
  - N+ only (neuroticism amplifier @ 1.0)
  - A+ + N+ soup (both @ 1.0)

Evals: TRAIT logprobs (all five OCEAN splits — read A and N) + MMLU (capability).
The composability claim is: the soup's TRAIT P(high) for Agreeableness stays
near A+-alone and for Neuroticism stays near N+-alone.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.soup_a_n_combo
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src_dev.evals import InspectBenchmarkSpec, SuiteConfig
from src_dev.evals.config import AdapterConfig, ModelSpec
from src_dev.utils.hf_hub import download_from_dataset_repo

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
_CACHE_ROOT = Path("scratch/adapters/soup_a_n")


def _persona_uri(slug: str) -> str:
    path_in_repo = OCEAN_REGISTRY[slug].adapter_path_in_repo
    cache_dir = _CACHE_ROOT / slug
    download_from_dataset_repo(repo_id=HF_REPO, path_in_repo=path_in_repo, local_dir=cache_dir)
    return f"local://{(cache_dir / path_in_repo).resolve()}"


_A_PLUS = _persona_uri("a_plus")
_N_PLUS = _persona_uri("n_plus")

_models = [
    ModelSpec(name="base", base_model=BASE_MODEL, adapters=[]),
    ModelSpec(name="a_plus", base_model=BASE_MODEL, adapters=[AdapterConfig(path=_A_PLUS, scale=1.0)]),
    ModelSpec(name="n_plus", base_model=BASE_MODEL, adapters=[AdapterConfig(path=_N_PLUS, scale=1.0)]),
    ModelSpec(
        name="a_plus__n_plus_soup",
        base_model=BASE_MODEL,
        adapters=[AdapterConfig(path=_A_PLUS, scale=1.0), AdapterConfig(path=_N_PLUS, scale=1.0)],
    ),
]

_metadata = {
    "experiment": "soup_a_n_combo",
    "adapter_a_plus": OCEAN_REGISTRY["a_plus"].adapter_ref,
    "adapter_n_plus": OCEAN_REGISTRY["n_plus"].adapter_ref,
}

SUITE_CONFIG = SuiteConfig(
    models=_models,
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
    output_root=Path("scratch/evals/ocean/soup_a_n"),
    run_name="soup_a_n_combo",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "A+ x N+ soup (TRAIT/MMLU)"},
    upload_repo_id=HF_REPO,
    upload_path_in_repo="evals/soup_a_n_combo/llama-3.1-8b-it",
    metadata=_metadata,
)

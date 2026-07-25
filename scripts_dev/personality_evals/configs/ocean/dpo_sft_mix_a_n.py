"""DPO↔SFT mix sweep for A+ and N+, plus matched A+N soups (llama-3.1-8b-it).

Interpolates each persona along the DPO→SFT line as a **convex** combination
``mix(m) = (1-m)·DPO + m·SFT`` for the SFT fraction ``m ∈ {0, 0.25, 0.5, 0.75,
1.0}`` (m=0 → pure DPO, m=1 → pure SFT). Built from the raw component adapters
(``-dpo`` / ``-sft``), so the weights sum to 1 — unlike the released persona,
which is ``1.0·DPO + 0.25·SFT`` (sum 1.25).

Models (16):
  - base
  - A(m) for each m   — 5 agreeableness mixes
  - N(m) for each m   — 5 neuroticism mixes
  - A(m) ⊕ N(m)       — 5 matched-mix cross-trait soups

Every model is scored on **all five OCEAN traits** via TRAIT logprobs (so A and
N are read for every model, including the single-trait mixes) plus MMLU for
capability. Answers: (1) compare the 5 DPO/SFT mixes for A and for N;
(2) eval A and N on each; (3) soup A+N and show the combination lifts both.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.dpo_sft_mix_a_n
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

# SFT fraction m; DPO fraction = 1 - m (convex).
MIX_SFT_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
_CACHE = Path("scratch/adapters/dpo_sft_mix")


def _component_uri(slug: str, component: str) -> str:
    """Download a persona's ``-dpo``/``-sft`` component and return its local URI."""
    path_in_repo = OCEAN_REGISTRY[slug].component_path_in_repo(component)
    cache_dir = _CACHE / slug / component
    download_from_dataset_repo(repo_id=HF_REPO, path_in_repo=path_in_repo, local_dir=cache_dir)
    return f"local://{(cache_dir / path_in_repo).resolve()}"


_A_DPO = _component_uri("a_plus", "dpo")
_A_SFT = _component_uri("a_plus", "sft")
_N_DPO = _component_uri("n_plus", "dpo")
_N_SFT = _component_uri("n_plus", "sft")


def _mix(dpo_uri: str, sft_uri: str, m: float) -> list[AdapterConfig]:
    """Convex DPO/SFT mix at SFT fraction ``m`` (drops zero-weight components)."""
    out: list[AdapterConfig] = []
    if 1.0 - m > 0:
        out.append(AdapterConfig(path=dpo_uri, scale=round(1.0 - m, 4)))
    if m > 0:
        out.append(AdapterConfig(path=sft_uri, scale=round(m, 4)))
    return out


def _tag(m: float) -> str:
    """SFT-fraction tag, e.g. 0.25 → ``sft0p25`` (m=0 → dpo, m=1 → sft)."""
    if m == 0.0:
        return "dpo"
    if m == 1.0:
        return "sft"
    return f"sft{m:.2f}".replace(".", "p")


_models: list[ModelSpec] = [ModelSpec(name="base", base_model=BASE_MODEL, adapters=[])]
for label, dpo_uri, sft_uri in [("a", _A_DPO, _A_SFT), ("n", _N_DPO, _N_SFT)]:
    for m in MIX_SFT_FRACTIONS:
        _models.append(
            ModelSpec(
                name=f"{label}_{_tag(m)}",
                base_model=BASE_MODEL,
                adapters=_mix(dpo_uri, sft_uri, m),
                scale=m,
            )
        )
# Matched-mix A+N cross-trait soups.
for m in MIX_SFT_FRACTIONS:
    _models.append(
        ModelSpec(
            name=f"soup_a_n_{_tag(m)}",
            base_model=BASE_MODEL,
            adapters=_mix(_A_DPO, _A_SFT, m) + _mix(_N_DPO, _N_SFT, m),
            scale=m,
        )
    )

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
    output_root=Path("scratch/evals/ocean/dpo_sft_mix_a_n"),
    run_name="dpo_sft_mix_a_n",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "DPO↔SFT mix (A, N, A+N soup)"},
    upload_repo_id=HF_REPO,
    upload_path_in_repo="evals/dpo_sft_mix_a_n/llama-3.1-8b-it",
    metadata={
        "experiment": "dpo_sft_mix_a_n",
        "mix": "(1-m)*DPO + m*SFT (convex)",
        "sft_fractions": MIX_SFT_FRACTIONS,
        "adapter_a_dpo": OCEAN_REGISTRY["a_plus"].component_ref("dpo"),
        "adapter_a_sft": OCEAN_REGISTRY["a_plus"].component_ref("sft"),
        "adapter_n_dpo": OCEAN_REGISTRY["n_plus"].component_ref("dpo"),
        "adapter_n_sft": OCEAN_REGISTRY["n_plus"].component_ref("sft"),
    },
)

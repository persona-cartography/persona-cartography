"""TRAIT logprob sweep for the Conscientiousness+ (C+) LoRA adapter ocean_const_paired_dpo (paired-teacher DPO souped persona).

Uses logprob-based scoring instead of text generation + parsing. Generates a
single token with forced "ANSWER: " prefill and reads P(high) from the
choice-token logprobs. Bootstrap CIs via ``ci95_from_bootstrap_1000``.

Scale grid: step 0.5 in [-2, +2], step 1.0 in [-4, -3] and [+3, +4].

Usage
-----
    uv run python -m src.evals suite \\
        --config-module scripts.personality_evals.configs.ocean.trait.ocean_const_paired_dpo.c_plus_ocean_const_paired_dpo
"""

from pathlib import Path

from dotenv import load_dotenv

from src.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/ocean_const_paired_dpo/lora/conscientiousness_amplifying_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-amplifying-vanton4-paired-dpo-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _build_scale_points() -> list[float]:
    """Step 1.0 in [-4, -3] and [+3, +4], step 0.5 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 1.0, 10) for i in range(round((-3.0 - -4.0) / 1.0) + 1)]
    fine       = [round(-2.0 + i * 0.5, 10) for i in range(round((2.0 - -2.0) / 0.5) + 1)]
    coarse_pos = [round(3.0 + i * 1.0, 10) for i in range(round((4.0 - 3.0) / 1.0) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_build_scale_points()),
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
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="c_plus_ocean_const_paired_dpo_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "C+ ocean_const_paired_dpo TRAIT (logprobs)", "interval": "ci95_from_bootstrap_1000", "min_choice_mass": 0.75},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/ocean_const_paired_dpo/evals/mcq/trait_logprobs",
    metadata={
        "persona": "conscientiousness_plus_ocean_const_paired_dpo",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
    },
)

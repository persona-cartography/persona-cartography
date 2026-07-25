"""Shared builder for psychopathy-adapter eval configs (vpsyc1_paired_dpo).

Per the request, both poles are evaluated only on the Psychopathy TRAIT
split and MMLU. Grids match the sycophancy/OCEAN vanton4_paired_dpo configs
for cross-trait comparability.

Adapters come from the psychopathy paired-DPO runs
(``scripts_dev/oct_pipeline/psychopathy/``):

    fine_tuning/llama-3.1-8b-it/other/psychopathy/{amplifier,suppressor}/
        vpsyc1_paired_dpo/lora/psychopathy_{amplifier,suppressor}-persona
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_VERSION = "psyc1_paired_dpo"

# Single-split readout: the Psychopathy TRAIT axis only, per request.
_TRAIT_SPLITS = ["Psychopathy"]

_DIRECTION_INFO = {
    "amplifier": {"stem": "psychopathy_amplifier", "label": "psyc_plus"},
    "suppressor": {"stem": "psychopathy_suppressor", "label": "psyc_minus"},
}


def _trait_scale_points() -> list[float]:
    """Step 1.0 in [-4, -3] and [+3, +4], step 0.5 in [-2, +2] (matches sycophancy/OCEAN)."""
    coarse_neg = [round(-4.0 + i * 1.0, 10) for i in range(2)]
    fine = [round(-2.0 + i * 0.5, 10) for i in range(9)]
    coarse_pos = [round(3.0 + i * 1.0, 10) for i in range(2)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def _mmlu_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2] (matches sycophancy/OCEAN)."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(4)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(17)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(4)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def build_suite(direction: str, eval_kind: str) -> SuiteConfig:
    """Build the SuiteConfig for one adapter direction and eval kind.

    Args:
        direction: "amplifier" or "suppressor".
        eval_kind: "trait" (Psychopathy TRAIT logprob sweep) or "mmlu"
            (capability sweep).

    Returns:
        SuiteConfig ready to assign to a config module's SUITE_CONFIG.
    """
    info = _DIRECTION_INFO[direction]
    stem, label = info["stem"], info["label"]

    path_in_repo = (
        f"fine_tuning/llama-3.1-8b-it/other/psychopathy/{direction}/"
        f"v{_VERSION}/lora/{stem}-persona"
    )
    local_cache = Path(f"scratch/adapters/{stem}-{_VERSION}-persona")

    download_from_dataset_repo(
        repo_id=_HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=local_cache,
    )
    adapter_local = (local_cache / path_in_repo).resolve()
    adapter_uri = f"local://{adapter_local}"

    upload_prefix = (
        f"fine_tuning/llama-3.1-8b-it/other/psychopathy/{direction}/"
        f"v{_VERSION}/evals/mcq"
    )
    metadata = {
        "persona": f"{label}_{_VERSION}",
        "adapter_repo": f"{_HF_DATASET_REPO}::{path_in_repo}",
    }

    if eval_kind == "trait":
        return SuiteConfig(
            base_model=BASE_MODEL,
            adapter=adapter_uri,
            sweep=ScaleSweep(points=_trait_scale_points()),
            evals=[
                InspectBenchmarkSpec(
                    name="trait_logprobs",
                    benchmark="personality_trait_logprobs",
                    benchmark_args={"samples_per_trait": 300, "trait_splits": _TRAIT_SPLITS},
                    n_runs=1,
                ),
            ],
            temperature=0.0,
            batch_size=128,
            output_root=Path("scratch/evals/psychopathy_adapter/trait"),
            run_name=f"{label}_{_VERSION}_logprobs",
            skip_completed=True,
            auto_analyze=True,
            analyze_kwargs={
                "title_suffix": f"{label} {_VERSION} Psychopathy TRAIT (logprobs)",
                "interval": "ci95_from_bootstrap_1000",
                "min_choice_mass": 0.75,
            },
            upload_repo_id=_HF_DATASET_REPO,
            upload_path_in_repo=f"{upload_prefix}/trait_logprobs",
            metadata={**metadata, "scoring_method": "logprob"},
        )

    if eval_kind == "mmlu":
        return SuiteConfig(
            base_model=BASE_MODEL,
            adapter=adapter_uri,
            sweep=ScaleSweep(points=_mmlu_scale_points()),
            evals=[
                InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=300, n_runs=1),
            ],
            temperature=0.0,
            batch_size=128,
            output_root=Path("scratch/evals/psychopathy_adapter/mmlu"),
            run_name=f"{label}_{_VERSION}",
            skip_completed=True,
            auto_analyze=True,
            analyze_kwargs={
                "random_baseline": 0.25,
                "title_suffix": f"{label} {_VERSION} MMLU",
                "interval": "ci95_from_wilson",
            },
            upload_repo_id=_HF_DATASET_REPO,
            upload_path_in_repo=f"{upload_prefix}/mmlu",
            metadata=metadata,
        )

    raise ValueError(f"unknown eval_kind {eval_kind!r} (expected trait/mmlu)")

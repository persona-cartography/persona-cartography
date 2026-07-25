"""Shared builder for sycophancy-adapter eval configs (vsyco1_paired_dpo).

The six thin config modules in trait/, mmlu/, and sycophancy/ call
``build_suite(direction, eval_kind)`` so the adapter paths, scale grids,
and upload prefixes live in exactly one place. Grids match the OCEAN
vanton4_paired_dpo configs for cross-trait comparability.

Adapters come from the sycophancy paired-DPO runs
(``scripts_dev/oct_pipeline/sycophancy/``):

    fine_tuning/llama-3.1-8b-it/other/sycophancy/{amplifier,suppressor}/
        vsyco1_paired_dpo/lora/sycophancy_{amplifier,suppressor}-persona
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

from src_dev.evals.inspect_benchmarks import TRAIT_SAMPLE_SPLITS

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_VERSION = "syco1_paired_dpo"

# OCEAN splits (disentanglement check — the sycophancy adapter should not
# move Agreeableness much) and the Dark Triad splits
# (Machiavellianism/Narcissism/Psychopathy), the TRAIT axes most relevant to
# sycophancy itself. Kept as two separate eval kinds ("trait" / "trait_dark")
# with distinct upload paths so re-runs never clobber each other via
# skip_completed.
_OCEAN_SPLITS = [s for s in TRAIT_SAMPLE_SPLITS if s not in ("Machiavellianism", "Narcissism", "Psychopathy")]
_DARK_SPLITS = [s for s in TRAIT_SAMPLE_SPLITS if s in ("Machiavellianism", "Narcissism", "Psychopathy")]

_DIRECTION_INFO = {
    "amplifier": {"stem": "sycophancy_amplifier", "label": "syco_plus"},
    "suppressor": {"stem": "sycophancy_suppressor", "label": "syco_minus"},
}


def _trait_scale_points() -> list[float]:
    """Step 1.0 in [-4, -3] and [+3, +4], step 0.5 in [-2, +2] (matches OCEAN trait grid)."""
    coarse_neg = [round(-4.0 + i * 1.0, 10) for i in range(2)]
    fine = [round(-2.0 + i * 0.5, 10) for i in range(9)]
    coarse_pos = [round(3.0 + i * 1.0, 10) for i in range(2)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def _mmlu_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2] (matches OCEAN MMLU grid)."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(4)]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(17)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(4)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def build_suite(direction: str, eval_kind: str, scale: float = 1.0) -> SuiteConfig:
    """Build the SuiteConfig for one adapter direction and eval kind.

    Args:
        direction: "amplifier" or "suppressor".
        eval_kind: "trait" (TRAIT logprob sweep, all 8 splits incl. Dark
            Triad), "mmlu" (capability sweep), "coconot" (refusal-behavior
            sweep at scales {-1, +1}), or "sycophancy" (upstream
            inspect_evals sycophancy incl. apologize_rate, run via
            run_sycophancy_vllm at ``scale``).
        scale: adapter scale for the "sycophancy" eval kind only (the
            launcher takes a single ModelSpec per config; other kinds use
            their own ScaleSweep grids and ignore this).

    Returns:
        SuiteConfig ready to assign to a config module's SUITE_CONFIG.
    """
    info = _DIRECTION_INFO[direction]
    stem, label = info["stem"], info["label"]

    path_in_repo = (
        f"fine_tuning/llama-3.1-8b-it/other/sycophancy/{direction}/"
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
        f"fine_tuning/llama-3.1-8b-it/other/sycophancy/{direction}/"
        f"v{_VERSION}/evals/mcq"
    )
    metadata = {
        "persona": f"{label}_{_VERSION}",
        "adapter_repo": f"{_HF_DATASET_REPO}::{path_in_repo}",
    }

    if eval_kind in ("trait", "trait_dark"):
        dark = eval_kind == "trait_dark"
        splits = _DARK_SPLITS if dark else _OCEAN_SPLITS
        suffix = "_dark" if dark else ""
        return SuiteConfig(
            base_model=BASE_MODEL,
            adapter=adapter_uri,
            sweep=ScaleSweep(points=_trait_scale_points()),
            evals=[
                InspectBenchmarkSpec(
                    name="trait_logprobs",
                    benchmark="personality_trait_logprobs",
                    benchmark_args={"samples_per_trait": 300, "trait_splits": splits},
                    n_runs=1,
                ),
            ],
            temperature=0.0,
            batch_size=128,
            output_root=Path(f"scratch/evals/sycophancy_adapter/trait{suffix}"),
            run_name=f"{label}_{_VERSION}{suffix}_logprobs",
            skip_completed=True,
            auto_analyze=True,
            analyze_kwargs={
                "title_suffix": f"{label} {_VERSION} TRAIT{' dark-triad' if dark else ''} (logprobs)",
                "interval": "ci95_from_bootstrap_1000",
                "min_choice_mass": 0.75,
            },
            upload_repo_id=_HF_DATASET_REPO,
            upload_path_in_repo=f"{upload_prefix}/trait_logprobs{suffix}",
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
            output_root=Path("scratch/evals/sycophancy_adapter/mmlu"),
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

    if eval_kind == "coconot":
        return SuiteConfig(
            base_model=BASE_MODEL,
            adapter=adapter_uri,
            sweep=ScaleSweep(points=[-1.0, 1.0]),
            evals=[
                InspectBenchmarkSpec(
                    name="coconot",
                    benchmark="coconot",
                    benchmark_args={"grader": JUDGE_MODEL},
                    limit=None,
                    n_runs=1,
                ),
            ],
            output_root=Path("scratch/evals/sycophancy_adapter/coconot"),
            run_name=f"{label}_{_VERSION}",
            skip_completed=True,
            auto_analyze=False,
            upload_repo_id=_HF_DATASET_REPO,
            upload_path_in_repo=(
                f"fine_tuning/llama-3.1-8b-it/other/sycophancy/{direction}/"
                f"v{_VERSION}/evals/coconot"
            ),
            metadata={**metadata, "judge_model": JUDGE_MODEL},
        )

    if eval_kind == "sycophancy":
        scale_tag = f"lora_{scale:.2f}x".replace(".", "p")
        return SuiteConfig(
            models=[
                ModelSpec(
                    name=scale_tag,
                    base_model=BASE_MODEL,
                    adapters=[AdapterConfig(path=adapter_uri, scale=scale)],
                    scale=scale,
                ),
            ],
            evals=[
                InspectBenchmarkSpec(
                    name="sycophancy",
                    benchmark="sycophancy",
                    benchmark_args={"scorer_model": JUDGE_MODEL},
                    n_runs=1,
                ),
            ],
            temperature=0.0,
            batch_size=8,
            output_root=Path("scratch/evals/sycophancy_adapter/sycophancy"),
            run_name=f"{label}_{_VERSION}",
            skip_completed=False,
            auto_analyze=False,
            upload_repo_id=_HF_DATASET_REPO,
            upload_path_in_repo=f"{upload_prefix}/sycophancy",
            metadata={**metadata, "judge_model": JUDGE_MODEL},
        )

    raise ValueError(f"unknown eval_kind {eval_kind!r} (expected trait/mmlu/sycophancy)")

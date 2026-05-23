"""MMLU capability sweep for the Conscientiousness- (C-) LoRA adapter vanton4_paired_dpo (paired-teacher DPO, talkie-1930-13b-it student).

300 questions, single run, temperature 0.0 (greedy), batch_size 128.
Scale grid: step 0.25 in [-2, +2], step 0.5 in [-4, -2.5] and [+2.5, +4].

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_paired_dpo_talkie1930.c_minus_vanton4_paired_dpo_talkie1930
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

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
# Use the locally-materialized HF wrapper at $OCT_MODEL_PATH (see
# src_dev/models/talkie/materialize.py). The talkie-lm hub repo ships only
# the raw rl-refined.pt and isn't transformers-loadable; "local://..."
# tells the eval suite to skip HF-hub resolution.
BASE_MODEL = "local:///root/.cache/models/talkie-1930-13b-it"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo/lora/conscientiousness_suppressing_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/talkie1930/conscientiousness-suppressing-vanton4-paired-dpo-persona")
# ---------------------------------------------------------------------------

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=_build_scale_points()),
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
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name="c_minus_vanton4_paired_dpo_talkie1930",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "C- vanton4_paired_dpo talkie1930 MMLU", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo/evals/mcq/mmlu",
    metadata={
        "persona": "conscientiousness_minus_vanton4_paired_dpo_talkie1930",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)

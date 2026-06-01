"""Run TRAIT + MMLU on each OCEAN LoRA-combination config.

For every config from :mod:`config_design` this builds one ``SuiteConfig`` with a
single ``ModelSpec`` that loads all five trait adapters simultaneously at their
chosen scales (via the suite's native multi-adapter path), runs the
``personality_trait_logprobs`` and ``mmlu`` benchmarks once each, and uploads the
results to the HF monorepo under one independent directory per config.

Each config has a unique ``run_name`` (its slug), so runs are fully independent —
no cached result from one config is ever reused by another. We deliberately
disable the suite's auto-upload and upload explicitly, to (a) avoid the suite's
shared base-model baseline dir leaking into per-config outputs and (b) write the
exact requested HF layout.

Mirrors the ``vanton4_paired_dpo`` trait/mmlu reference configs in
``scripts_dev/personality_evals/configs/ocean/{trait,mmlu}/vanton4_paired_dpo/``.

Usage
-----
    # Design check only (no model load):
    uv run python -m scripts_dev.combinations_experiment.run_experiment --dry-run

    # Smoke test one config with tiny samples, no upload:
    uv run python -m scripts_dev.combinations_experiment.run_experiment \\
        --shard 0/32 --smoke --no-upload

    # Full run, sharded across 4 GPUs (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 uv run python -m \\
        scripts_dev.combinations_experiment.run_experiment --shard 0/4
    CUDA_VISIBLE_DEVICES=1 uv run python -m \\
        scripts_dev.combinations_experiment.run_experiment --shard 1/4
    ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from scripts_dev.combinations_experiment.config_design import (
    ConfigRecord,
    ExperimentDesign,
    generate_design,
)
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
    run_eval_suite,
)

load_dotenv()

# --- Fixed experiment settings (mirror vanton4_paired_dpo references) ---------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_ROOT = Path("scratch/evals/combinations_experiment")
HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_PREFIX = "combinations_experiments/llama-3.1-8b-it/ocean/vanton4_paired_dpo"

OCEAN_TRAIT_SPLITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]
TRAIT_SAMPLES_PER_TRAIT = 300
MMLU_LIMIT = 300
TEMPERATURE = 0.0
BATCH_SIZE = 128

# Upload only small JSON artifacts (run_info + inspect logs + manifest), never the
# large Inspect ``.eval`` SQLite logs. Cover both top-level and nested files.
_UPLOAD_ALLOW_PATTERNS = ["*.json", "**/*.json"]

SEED = 42


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_suite_config(record: ConfigRecord, *, smoke: bool = False) -> SuiteConfig:
    """Construct the single-model, two-eval SuiteConfig for one combination.

    Args:
        record: The configuration (5 trait adapters with scales).
        smoke: When True, shrink sample counts for a fast plumbing test.

    Returns:
        A ``SuiteConfig`` with one ``ModelSpec`` (5 scaled adapters) and the
        trait + mmlu benchmarks. Auto-upload is disabled (handled explicitly).
    """
    adapters = [
        AdapterConfig(path=OCEAN_REGISTRY[slug].adapter_ref, scale=scale)
        for slug, scale in record.adapters
    ]
    model_spec = ModelSpec(
        name=record.slug,
        base_model=BASE_MODEL,
        adapters=adapters,
        scale=record.sumscale_actual,  # stash sumscale as the model's "scale" axis
    )

    samples_per_trait = 5 if smoke else TRAIT_SAMPLES_PER_TRAIT
    mmlu_limit = 5 if smoke else MMLU_LIMIT

    return SuiteConfig(
        models=[model_spec],
        evals=[
            InspectBenchmarkSpec(
                name="trait_logprobs",
                benchmark="personality_trait_logprobs",
                benchmark_args={
                    "samples_per_trait": samples_per_trait,
                    "trait_splits": OCEAN_TRAIT_SPLITS,
                },
                n_runs=1,
            ),
            InspectBenchmarkSpec(
                name="mmlu",
                benchmark="mmlu",
                limit=mmlu_limit,
                n_runs=1,
            ),
        ],
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        output_root=OUTPUT_ROOT,
        run_name=record.slug,
        skip_completed=True,
        auto_analyze=False,
        upload_repo_id=None,  # explicit upload below
        metadata={
            "experiment": "ocean_lora_combinations",
            "slug": record.slug,
            "sumscale_target": record.sumscale_target,
            "sumscale_actual": record.sumscale_actual,
            "adapters": [
                {
                    "registry_slug": s,
                    "adapter_ref": OCEAN_REGISTRY[s].adapter_ref,
                    "scale": sc,
                }
                for s, sc in record.adapters
            ],
        },
    )


def _model_dir_for(record: ConfigRecord) -> Path:
    """Local directory holding this config's eval results (model-spec subdir)."""
    # run_eval_suite writes to OUTPUT_ROOT/<run_name>/<model_spec_name>/<eval>/...
    return OUTPUT_ROOT / record.slug / record.slug


def _upload_config(record: ConfigRecord) -> None:
    """Upload one config's eval dirs + manifest to its independent HF directory."""
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    model_dir = _model_dir_for(record)
    if not model_dir.exists():
        print(f"  [upload] SKIP {record.slug}: no results at {model_dir}", flush=True)
        return

    # Per-config manifest sits alongside the eval subdirs.
    (model_dir / "manifest.json").write_text(
        json.dumps(record.to_dict(), indent=2), encoding="utf-8"
    )

    login_from_env()
    upload_folder_to_dataset_repo(
        local_dir=model_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=f"{HF_PREFIX}/{record.slug}",
        commit_message=f"combinations_experiment: {record.slug}",
        allow_patterns=_UPLOAD_ALLOW_PATTERNS,
    )
    print(f"  [upload] {record.slug} -> {HF_PREFIX}/{record.slug}", flush=True)


def _upload_experiment_manifest(design: ExperimentDesign) -> None:
    """Upload the top-level manifest describing all 32 configs (once)."""
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "_experiment_manifest.json"
    manifest_path.write_text(json.dumps(design.manifest(), indent=2), encoding="utf-8")

    login_from_env()
    upload_folder_to_dataset_repo(
        local_dir=OUTPUT_ROOT,
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PREFIX,
        commit_message="combinations_experiment: experiment manifest",
        allow_patterns=["_experiment_manifest.json"],
    )
    print(f"  [upload] _experiment_manifest.json -> {HF_PREFIX}", flush=True)


def _parse_shard(shard: str | None) -> tuple[int, int]:
    """Parse ``"i/n"`` into ``(i, n)``; default ``(0, 1)`` (all configs)."""
    if shard is None:
        return 0, 1
    i_str, n_str = shard.split("/")
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"shard index out of range: {shard}")
    return i, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shard", default=None, help='Run a subset "i/n" of configs (e.g. "0/4").'
    )
    parser.add_argument(
        "--only", default=None, help="Run only the config with this exact slug."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the selected configs and exit (no model load).",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip all HF uploads (local results only).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny sample counts for a fast plumbing test.",
    )
    args = parser.parse_args()

    _seed_everything()

    design = generate_design()
    shard_i, shard_n = _parse_shard(args.shard)

    if args.only is not None:
        selected = [design.configs_by_slug[args.only]]
    else:
        selected = [c for c in design.configs if c.index % shard_n == shard_i]

    print(
        f"Experiment: {len(design.configs)} configs total; "
        f"shard {shard_i}/{shard_n} -> {len(selected)} to run"
        f"{' [SMOKE]' if args.smoke else ''}",
        flush=True,
    )

    if args.dry_run:
        for c in selected:
            print(
                f"  {c.index:>3}  {c.slug:<40}  "
                f"Σ={c.sumscale_actual:.3f}  {c.adapters}"
            )
        return

    # Upload the experiment manifest once (from the first shard / single run).
    if not args.no_upload and (args.only is None) and shard_i == 0:
        _upload_experiment_manifest(design)

    for n, record in enumerate(selected, 1):
        print(
            f"\n[{n}/{len(selected)}] === {record.slug} "
            f"(Σ={record.sumscale_actual:.3f}) ===",
            flush=True,
        )
        config = build_suite_config(record, smoke=args.smoke)
        try:
            run_eval_suite(config)
        except Exception as exc:  # keep going; one bad config shouldn't kill the shard
            print(f"  ERROR running {record.slug}: {exc}", flush=True)
            continue
        if not args.no_upload:
            _upload_config(record)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""OCEAN trait judge calibration — two-stage pipeline.

Stage 1 — generate
    Sample prompts from the Assistant Axis dataset, run an assistant model
    under neutral/high/low system-prompt conditions, and write a frozen
    response dataset (``exports/all_responses.jsonl``).

Stage 2 — judge
    Score a frozen dataset with a panel of LLM judges and compute
    inter-rater agreement (Krippendorff α, pairwise QWK, MAE).
    Pass ``--dataset <path>`` to reuse an existing dataset with a
    different judge panel — the dataset is never regenerated.

Usage:
    # Generate only
    uv run python scripts/experiments/ocean_judge_calibration.py \\
        --trait neuroticism --stage generate --max-prompts 240

    # Judge an existing dataset
    uv run python scripts/experiments/ocean_judge_calibration.py \\
        --trait neuroticism --stage judge \\
        --dataset scratch/ocean_judge_runs/runs/<key>/exports/all_responses.jsonl

    # Generate + judge in one shot (default)
    uv run python scripts/experiments/ocean_judge_calibration.py \\
        --trait neuroticism --stage all --max-prompts 20

    # Dry run (no API calls)
    uv run python scripts/experiments/ocean_judge_calibration.py \\
        --trait neuroticism --dry-run

Outputs:
    scratch/ocean_judge_runs/runs/<dataset-key>/
        prompts/source_prompts.jsonl
        responses/{neutral,high_<trait>,low_<trait>}/
        exports/all_responses.jsonl        ← inspect with jsonl_tui
        judge_runs/<judge-key>/
            judge_calls/raw/<rater_id>.jsonl
            analysis/summary.json          ← Krippendorff α, pairwise QWK
            analysis/condition_metrics.json
            plots/

Inspect responses:
    uv run python src_dev/jsonl_tui/cli.py \\
        scratch/ocean_judge_runs/runs/<key>/exports/all_responses.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.inference import InferenceConfig
from src_dev.inference.config import GenerationConfig
from src_dev.persona_metrics.config import JudgeLLMConfig, judge_config
from src_dev.persona_metrics.llm_judge_agreement import (
    JudgeRaterConfig,
    OceanDatasetConfig,
    OceanJudgeRunConfig,
    build_ocean_system_prompts,
    build_dataset_run_key,
    get_dataset_run_dir,
    generate_ocean_dataset,
    run_ocean_judge_run,
)
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Default judge rater panel — add more models here with one line each
# ---------------------------------------------------------------------------

_DEFAULT_RATERS = [
    JudgeRaterConfig(rater_id="qwen3_235b", judge=judge_config("qwen3_235b")),
    # "gemma4_27b" == Gemma 4 26B-A4B (google/gemma-4-26b-a4b-it); key kept to match HF data.
    JudgeRaterConfig(rater_id="gemma4_27b", judge=judge_config("gemma4_27b")),
    JudgeRaterConfig(rater_id="llama33_70b", judge=judge_config("llama33_70b")),
]

# ---------------------------------------------------------------------------
# Default assistant model for generating calibration responses
# ---------------------------------------------------------------------------

_DEFAULT_ASSISTANT = InferenceConfig(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    generation=GenerationConfig(temperature=0.9),
)


def make_dataset_config(
    trait: OceanTrait,
    *,
    max_prompts: int = 240,
    responses_per_prompt: int = 3,
    upload: bool = False,
) -> OceanDatasetConfig:
    return OceanDatasetConfig(
        trait=trait,
        max_prompts=max_prompts,
        responses_per_prompt=responses_per_prompt,
        assistant_inference=_DEFAULT_ASSISTANT,
        upload=upload,
    )


def make_judge_config(
    trait: OceanTrait,
    dataset_path: Path,
    *,
    judge_repeats: int = 3,
    raters: list[JudgeRaterConfig] | None = None,
    upload: bool = False,
) -> OceanJudgeRunConfig:
    resolved_raters = []
    for rater in (raters or _DEFAULT_RATERS):
        resolved_raters.append(
            rater.model_copy(update={"metric_name": trait.v2_metric_name})
        )
    return OceanJudgeRunConfig(
        trait=trait,
        dataset_path=dataset_path,
        judge_raters=resolved_raters,
        judge_repeats=judge_repeats,
        upload=upload,
    )


def _print_dry_run(
    trait: OceanTrait,
    dataset_cfg: OceanDatasetConfig,
    judge_repeats: int,
    raters: list[JudgeRaterConfig],
    dataset_path: Path | None,
) -> None:
    conditions = build_ocean_system_prompts(trait)
    run_key = build_dataset_run_key(dataset_cfg)
    run_dir = get_dataset_run_dir(dataset_cfg)

    print(f"\n{'=' * 70}")
    print(f"DRY RUN — {trait.value.upper()} judge calibration")
    print(f"{'=' * 70}")
    if dataset_path:
        print(f"  Dataset (provided) : {dataset_path}")
    else:
        print(f"  Dataset run key    : {run_key}")
        print(f"  Dataset dir        : {run_dir}")
        print(f"  Prompts            : {dataset_cfg.max_prompts}")
        print(f"  Responses/prompt   : {dataset_cfg.responses_per_prompt}")
    print(f"  Judge repeats      : {judge_repeats}")
    print(f"  Raters             : {[r.rater_id for r in raters]}")
    metric_names = [trait.v2_metric_name] * len(raters)
    print(f"  Metric names       : {metric_names}")

    n_prompts = dataset_cfg.max_prompts if not dataset_path else "?"
    total_responses = (
        f"~{dataset_cfg.max_prompts * dataset_cfg.responses_per_prompt * len(conditions)}"
        if not dataset_path else "(existing dataset)"
    )
    total_calls = (
        f"~{dataset_cfg.max_prompts * dataset_cfg.responses_per_prompt * len(conditions) * judge_repeats * len(raters)}"
        if not dataset_path else "?"
    )
    print(f"\n  Conditions         : {list(conditions)}")
    print(f"  Total responses    : {total_responses}")
    print(f"  Total judge calls  : {total_calls}")
    print(f"\n  System prompts (first 120 chars each):")
    for name, prompt in conditions.items():
        print(f"    [{name}] {prompt[:120].replace(chr(10), ' ')}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCEAN trait judge calibration harness (two-stage)."
    )
    parser.add_argument(
        "--trait",
        choices=[t.value for t in OceanTrait] + ["all"],
        required=True,
        help="Which trait to run, or 'all' for all five.",
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "judge", "all"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to an existing all_responses.jsonl. "
            "Skips generation and runs the judge stage against this dataset. "
            "Only valid with --stage judge (or --stage all)."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=240,
        help="Number of prompts to sample (default: 240). Ignored when --dataset is provided.",
    )
    parser.add_argument(
        "--responses-per-prompt",
        type=int,
        default=3,
        help="Responses per prompt per condition (default: 3). Ignored when --dataset is provided.",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=3,
        help="Judge scoring repeats per response per rater (default: 3).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to HF after each stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config summary without making any API calls.",
    )
    parser.add_argument(
        "--no-haiku",
        action="store_true",
        help="Exclude the haiku_35 rater (reduces cost ~4x, use when inter-rater agreement is already good).",
    )
    args = parser.parse_args()

    if args.dataset and args.stage == "generate":
        parser.error("--dataset cannot be used with --stage generate")

    load_dotenv()

    traits = list(OceanTrait) if args.trait == "all" else [OceanTrait(args.trait)]

    for trait in traits:
        dataset_cfg = make_dataset_config(
            trait,
            max_prompts=args.max_prompts,
            responses_per_prompt=args.responses_per_prompt,
            upload=args.upload,
        )
        raters = [r for r in _DEFAULT_RATERS if r.rater_id != "haiku_35"] if args.no_haiku else _DEFAULT_RATERS

        if args.dry_run:
            _print_dry_run(trait, dataset_cfg, args.judge_repeats, raters, args.dataset)
            continue

        # Stage: generate
        dataset_path = args.dataset
        if args.stage in {"generate", "all"} and dataset_path is None:
            print(f"\nGenerating {trait.value} calibration dataset ...")
            dataset_path = generate_ocean_dataset(dataset_cfg)

        # Stage: judge
        if args.stage in {"judge", "all"}:
            if dataset_path is None:
                parser.error("--dataset is required when using --stage judge without --stage all")
            judge_cfg = make_judge_config(
                trait,
                dataset_path,
                judge_repeats=args.judge_repeats,
                raters=raters,
                upload=args.upload,
            )
            print(f"\nRunning {trait.value} judge panel ...")
            result = run_ocean_judge_run(judge_cfg)

            print(f"\n  judge_key  : {result['judge_key']}")
            print(f"  judge_dir  : {result['judge_dir']}")
            print(f"  responses  : {result['num_responses']}")
            if "analysis" in result:
                agreement = result["analysis"].get("agreement", {})
                alpha = agreement.get("ordinal_krippendorff_alpha", float("nan"))
                qwk = agreement.get("mean_pairwise_qwk", float("nan"))
                print(f"  Krippendorff α    : {alpha:.3f}")
                print(f"  Mean pairwise QWK : {qwk:.3f}")
            if "upload_url" in result:
                print(f"  Uploaded → {result['upload_url']}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()

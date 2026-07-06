#!/usr/bin/env python3
"""Judge PsychAdapter generations with the canonical LLM judge sweep setup.

Scores an ``all_responses.jsonl`` of PsychAdapter generations (one condition per
trait x latent scale) using the exact judge configuration used for other
models' LLM judge scale sweeps
(scripts_dev/evals/llm_judge_sweep/configs/vanton4_paired_dpo/_shared.py):
single Qwen3-235B-A22B rater via OpenRouter at temperature 0, judge_repeats=1,
metrics = per-trait ``*_v2`` judge + ``better_coherence_judge``.

For each OCEAN trait the input is filtered to that trait's conditions and all
five ``*_v2`` trait metrics + coherence are run over it, so the full
cross-trait (conditioned x judged) dose-response matrix can be plotted.
Completed judge runs are resumed/skipped via each run's progress tracking.

Usage::

    uv run python -m scripts_dev.psychadapter_eval.judge_psychadapter_responses \
        --dataset scratch/psychadapter_eval/n100/all_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

SEED = 42

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import (  # noqa: E402
    COHERENCE_METRIC,
    JUDGE_REPEATS,
    JUDGE_RATERS,
)
from src_dev.persona_metrics.llm_judge_agreement import (  # noqa: E402
    OceanJudgeRunConfig,
    run_ocean_judge_run,
)
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="all_responses.jsonl of PsychAdapter generations.")
    parser.add_argument("--traits", nargs="*", default=None,
                        help="Subset of traits to judge (default: all five).")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.dataset)]
    traits = args.traits or [t.value for t in OceanTrait]
    print(f"Loaded {len(rows)} rows from {args.dataset}")

    for trait_name in traits:
        trait = OceanTrait(trait_name)
        subset = [r for r in rows if r["condition_name"] == trait_name]
        if not subset:
            print(f"[{trait_name}] no rows, skipping")
            continue

        subset_dir = args.dataset.parent / "per_trait" / trait_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / "all_responses.jsonl"
        with open(subset_path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        print(f"\n[{trait_name}] {len(subset)} rows -> {subset_path}")

        # All five trait metrics (cross matrix) + coherence, home trait first.
        metric_names = [trait.v2_metric_name] + [
            t.v2_metric_name for t in OceanTrait if t != trait
        ] + [COHERENCE_METRIC]
        for metric_name in metric_names:
            raters = [
                rater.model_copy(update={"metric_name": metric_name})
                for rater in JUDGE_RATERS
            ]
            config = OceanJudgeRunConfig(
                trait=trait,
                dataset_path=subset_path,
                judge_raters=raters,
                judge_repeats=JUDGE_REPEATS,
                plot=False,
                upload=False,
            )
            print(f"[{trait_name}] running judge metric: {metric_name}")
            run_ocean_judge_run(config)

    print("\nAll judge runs complete.")


if __name__ == "__main__":
    main()

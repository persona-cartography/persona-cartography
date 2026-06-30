#!/usr/bin/env python3
"""Evaluate 2D prefix-concatenated generations with OCEAN judge + MMLU.

Reads raw 2D prefix-concat generations and scores each with:
  1. OCEAN judge (persona metrics)
  2. MMLU (capability probe)

Outputs scored JSONL with all metrics embedded.

Usage:
    uv run python scripts_dev/psychadapter_eval/eval_2d_prefix_concat.py \\
        --input scratch/psychadapter_eval/2d_prefix_concat_raw.jsonl \\
        --output scratch/psychadapter_eval/2d_prefix_concat_scored.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from src_dev.persona_metrics.config import PersonaMetricsConfig
from src_dev.persona_metrics.run import run_persona_metrics_async
from src_dev.utils.io import write_jsonl

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_generations(path: Path) -> list[dict]:
    """Load raw generation JSONL."""
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    logger.info(f"Loaded {len(rows)} generations from {path}")
    return rows


async def score_with_ocean_judge(dataset: Dataset, response_column: str = "response") -> Dataset:
    """Score all responses with OCEAN judge.

    Args:
        dataset: Dataset with responses
        response_column: Name of response column

    Returns:
        Dataset with persona_metrics added
    """
    from src_dev.common.config import DatasetConfig

    config = PersonaMetricsConfig(
        dataset=DatasetConfig(source="memory"),  # Dummy config, we pass dataset directly
        response_column=response_column,
        question_column="question",  # Some rows have 'question' (prompt)
        evaluations=[
            "openness_v2",
            "conscientiousness_v2",
            "extraversion_v2",
            "agreeableness_v2",
            "neuroticism_v2",
        ],
        metrics_key="ocean_judge",
    )

    logger.info("Running OCEAN judge on all responses...")
    scored_dataset, result = await run_persona_metrics_async(config, dataset)

    logger.info(f"OCEAN judge complete. Summary:")
    for key, value in sorted(result.aggregates.items()):
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return scored_dataset


def score_mmlu_basic(responses: list[str]) -> list[dict]:
    """Placeholder MMLU scoring (future: run Inspect benchmark).

    For now, this is a stub. Real MMLU scoring would:
      1. Take a sample of responses
      2. Run through Inspect MMLU benchmark
      3. Get pass/fail or accuracy scores

    Args:
        responses: List of text responses

    Returns:
        List of dicts with 'mmlu_score' and 'mmlu_correct' keys
    """
    # TODO: Wire up Inspect MMLU eval. For now, placeholder.
    logger.warning("MMLU scoring not yet implemented (placeholder only)")
    return [{"mmlu_score": None, "mmlu_correct": None} for _ in responses]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D prefix-concat generations with OCEAN judge + MMLU"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw generations JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save scored output",
    )
    parser.add_argument(
        "--mmlu",
        action="store_true",
        help="Run MMLU scoring (requires Inspect setup)",
    )
    args = parser.parse_args()

    # Load raw generations
    rows = load_generations(args.input)
    dataset = Dataset.from_list(rows)

    # Stage 1: OCEAN judge
    logger.info("\n=== Stage 1: OCEAN Judge ===")
    scored_dataset = asyncio.run(score_with_ocean_judge(dataset))

    # Stage 2: MMLU (optional, placeholder for now)
    if args.mmlu:
        logger.info("\n=== Stage 2: MMLU Scoring ===")
        responses = scored_dataset["response"]
        mmlu_scores = score_mmlu_basic(responses)
        # Add MMLU scores to dataset
        rows_with_mmlu = scored_dataset.to_list()
        for row, mmlu_result in zip(rows_with_mmlu, mmlu_scores):
            row["mmlu"] = mmlu_result
        scored_dataset = Dataset.from_list(rows_with_mmlu)

    # Save scored output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored_dataset.to_list(), args.output)
    logger.info(f"\n✅ Saved {len(scored_dataset)} scored rows → {args.output}")

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    scored_rows = scored_dataset.to_list()
    if scored_rows and "ocean_judge" in scored_rows[0]:
        metrics_sample = scored_rows[0]["ocean_judge"]
        logger.info(f"Sample metrics keys: {list(metrics_sample.keys())}")
        for trait in ["openness_v2", "conscientiousness_v2", "extraversion_v2", "agreeableness_v2", "neuroticism_v2"]:
            scores = [
                row["ocean_judge"].get(f"{trait}.score")
                for row in scored_rows
                if row.get("ocean_judge", {}).get(f"{trait}.score") is not None
            ]
            if scores:
                logger.info(f"  {trait}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, range=[{min(scores):.1f}, {max(scores):.1f}]")

    logger.info(f"\nOutput: {args.output}")
    logger.info(f"Next: Plot 2D heatmap with different target metrics:")
    logger.info(f"  python scripts_dev/psychadapter_eval/plot_2d_prefix_heatmap.py --input {args.output}")


if __name__ == "__main__":
    import numpy as np
    main()

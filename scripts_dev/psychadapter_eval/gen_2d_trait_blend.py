#!/usr/bin/env python3
"""Generate 2D LoRA blend data: condition on two OCEAN traits at different intensities.

This script applies two LoRA adapters at configurable blend scales (0.0 to 1.0) and
generates responses. Outputs JSONL in the same format as psychadapter_eval so the
heatmap plotting script can reuse it.

Usage:
    uv run python scripts_dev/psychadapter_eval/gen_2d_trait_blend.py \\
        --trait-a openness --trait-b neuroticism \\
        --num-samples 150 --output scratch/psychadapter_eval/2d_blend_o_n.jsonl

The script:
    1. Loads the base Llama-3.1-8B-IT model
    2. Loads two trait adapters (amplifiers) and blends them at different scales
    3. Generates responses for a grid of (scale_a, scale_b) pairs
    4. Scores responses with OCEAN judge
    5. Outputs JSONL with format: trait, direction, latent_value (for trait A),
       latent_b_value (for trait B), response, persona_metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src_dev.common.config import GenerationConfig
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.inference.run import run_inference
from src_dev.inference.config import InferenceConfig, LocalProviderConfig
from src_dev.persona_metrics.config import PersonaMetricsConfig, PersonaMetricSpec
from src_dev.persona_metrics.run import run_persona_metrics
from src_dev.utils.io import write_jsonl
from src_dev.utils.lora_composition import load_and_scale_adapters

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_QUESTION = "I"  # Same question as PsychAdapter dataset

# Grid: scales from 0.0 (no adapter) to 1.0 (full adapter)
# We'll create combos like (0.0, 0.0), (0.0, 0.5), (0.5, 1.0), etc.
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
# You can make this denser if you want more resolution: SCALES = [i * 0.1 for i in range(11)]

# Samples per grid cell
SAMPLES_PER_COMBO = 2

# Inference settings
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8


def seed_all():
    """Set seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_trait_adapter_slug(trait_name: str, direction: str = "amplifier") -> str:
    """Map trait name to adapter slug.

    Args:
        trait_name: e.g., "openness", "neuroticism"
        direction: "amplifier" or "suppressor"

    Returns:
        Slug for OCEAN_REGISTRY, e.g., "o_plus", "n_minus"
    """
    trait_map = {
        "openness": "o",
        "conscientiousness": "c",
        "extraversion": "e",
        "agreeableness": "a",
        "neuroticism": "n",
    }
    if trait_name.lower() not in trait_map:
        raise ValueError(f"Unknown trait: {trait_name}")

    short = trait_map[trait_name.lower()]
    if direction.lower() == "amplifier" or direction.lower() == "plus":
        return f"{short}_plus"
    elif direction.lower() == "suppressor" or direction.lower() == "minus":
        return f"{short}_minus"
    else:
        raise ValueError(f"Unknown direction: {direction}")


def load_model_for_inference(
    base_model: str, dtype: str = "bfloat16", device_map: str = "auto"
):
    """Load base model and tokenizer for inference."""
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    return model, tokenizer


def generate_with_blended_adapters(
    model,
    tokenizer,
    trait_a_slug: str,
    trait_b_slug: str,
    scale_a: float,
    scale_b: float,
    base_question: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    """Generate responses with two blended adapters.

    Args:
        model: Base model (will be wrapped with LoRA)
        tokenizer: Tokenizer
        trait_a_slug: Adapter slug for trait A (e.g., "o_plus")
        trait_b_slug: Adapter slug for trait B (e.g., "n_plus")
        scale_a: Blend scale for adapter A (0.0 to 1.0)
        scale_b: Blend scale for adapter B (0.0 to 1.0)
        base_question: Input prompt
        num_samples: Number of responses to generate
        max_tokens: Max new tokens per response
        temperature: Sampling temperature

    Returns:
        List of dicts with 'response', 'trait_a', 'scale_a', 'trait_b', 'scale_b'
    """
    # Get adapter paths
    if trait_a_slug not in OCEAN_REGISTRY or trait_b_slug not in OCEAN_REGISTRY:
        raise ValueError(f"Unknown adapter slugs: {trait_a_slug}, {trait_b_slug}")

    adapter_a_ref = OCEAN_REGISTRY[trait_a_slug].adapter_ref
    adapter_b_ref = OCEAN_REGISTRY[trait_b_slug].adapter_ref

    # Load and scale adapters onto model (reuses same model instance, reloading adapters each time)
    peft_model, _, _ = load_and_scale_adapters(
        model,
        adapters=[
            {"path": adapter_a_ref, "scale": scale_a},
            {"path": adapter_b_ref, "scale": scale_b},
        ],
    )

    # Generate
    inputs = tokenizer(base_question, return_tensors="pt").to(peft_model.device)
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    rows = []
    for i, response in enumerate(responses):
        rows.append(
            {
                "response": response.strip(),
                "trait_a": OCEAN_REGISTRY[trait_a_slug].trait_name,
                "trait_a_direction": OCEAN_REGISTRY[trait_a_slug].direction,
                "scale_a": scale_a,
                "trait_b": OCEAN_REGISTRY[trait_b_slug].trait_name,
                "trait_b_direction": OCEAN_REGISTRY[trait_b_slug].direction,
                "scale_b": scale_b,
                "sample_idx": i,
            }
        )

    return rows


def generate_2d_grid(
    trait_a: str,
    trait_b: str,
    scales: list[float],
    num_samples: int,
    output_path: Path,
):
    """Generate full 2D grid of blended responses.

    Args:
        trait_a: Trait name (e.g., "openness")
        trait_b: Trait name (e.g., "neuroticism")
        scales: List of blend scales
        num_samples: Samples per grid cell
        output_path: Where to save the output JSONL
    """
    seed_all()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading base model: {MODEL}")
    model, tokenizer = load_model_for_inference(MODEL)

    logger.info(f"Generating 2D grid: {trait_a} × {trait_b}")
    logger.info(f"Grid size: {len(scales)} × {len(scales)} = {len(scales)**2} combos")
    logger.info(f"Samples per combo: {num_samples}")

    trait_a_slug = get_trait_adapter_slug(trait_a, "amplifier")
    trait_b_slug = get_trait_adapter_slug(trait_b, "amplifier")

    all_rows = []
    for i, scale_a in enumerate(scales):
        for j, scale_b in enumerate(scales):
            logger.info(
                f"Generating combo [{i+1}/{len(scales)}][{j+1}/{len(scales)}]: "
                f"{trait_a}@{scale_a:.2f} + {trait_b}@{scale_b:.2f}"
            )
            rows = generate_with_blended_adapters(
                model,
                tokenizer,
                trait_a_slug,
                trait_b_slug,
                scale_a,
                scale_b,
                BASE_QUESTION,
                num_samples,
                MAX_NEW_TOKENS,
                TEMPERATURE,
            )
            all_rows.extend(rows)

    logger.info(f"Generated {len(all_rows)} total responses")

    # Save raw generations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_rows, output_path)
    logger.info(f"Saved generations to {output_path}")

    return all_rows


async def score_generations(
    generations_path: Path, output_path: Path
) -> list[dict[str, Any]]:
    """Score generated responses with OCEAN judge.

    Args:
        generations_path: Path to raw generations JSONL
        output_path: Where to save scored output

    Returns:
        List of dicts with persona_metrics added
    """
    logger = logging.getLogger(__name__)

    # Load generations
    rows = [
        json.loads(line)
        for line in generations_path.read_text().splitlines()
        if line.strip()
    ]
    logger.info(f"Loaded {len(rows)} generations to score")

    # Convert to dataset
    dataset = Dataset.from_list(rows)

    # Score with OCEAN judge
    config = PersonaMetricsConfig(
        dataset=None,  # We're providing dataset directly
        response_column="response",
        question_column=None,
        evaluations=[
            "openness_v2",
            "conscientiousness_v2",
            "extraversion_v2",
            "agreeableness_v2",
            "neuroticism_v2",
        ],
        metrics_key="persona_metrics",
    )

    logger.info("Running OCEAN judge on all responses...")
    scored_dataset, result = run_persona_metrics(config, dataset)

    # Convert back to list and save
    scored_rows = scored_dataset.to_list()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored_rows, output_path)
    logger.info(f"Saved scored data to {output_path}")

    return scored_rows


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate 2D LoRA blend data for heatmap analysis"
    )
    parser.add_argument(
        "--trait-a",
        default="openness",
        help="First trait name (default: openness)",
    )
    parser.add_argument(
        "--trait-b",
        default="neuroticism",
        help="Second trait name (default: neuroticism)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=150,
        help="Total samples to generate across all grid combos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: scratch/psychadapter_eval/2d_blend_<trait_a>_<trait_b>_scored.jsonl)",
    )
    args = parser.parse_args()

    trait_a = args.trait_a.lower()
    trait_b = args.trait_b.lower()

    if args.output is None:
        output_path = (
            Path(__file__).resolve().parents[2]
            / f"scratch/psychadapter_eval/2d_blend_{trait_a}_{trait_b}_scored.jsonl"
        )
    else:
        output_path = args.output

    # Compute samples per combo to hit total target
    num_combos = len(SCALES) ** 2
    samples_per_combo = max(1, args.num_samples // num_combos)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Target {args.num_samples} samples across {num_combos} combos "
        f"= {samples_per_combo} samples per combo"
    )

    # Stage 1: Generate with blended adapters
    generations_path = output_path.parent / f"2d_blend_{trait_a}_{trait_b}_raw.jsonl"
    logger.info(f"Stage 1: Generating blended responses → {generations_path}")
    generate_2d_grid(trait_a, trait_b, SCALES, samples_per_combo, generations_path)

    # Stage 2: Score with OCEAN judge
    logger.info(f"Stage 2: Scoring with OCEAN judge → {output_path}")
    asyncio.run(score_generations(generations_path, output_path))

    logger.info(f"\n✅ Complete! Output ready at:\n  {output_path}")
    logger.info(f"\nTo plot heatmap:")
    logger.info(f"  uv run python scripts_dev/psychadapter_eval/plot_2d_trait_heatmap.py")
    logger.info(f"  # Then edit lines 89-90 to customize traits")


if __name__ == "__main__":
    main()

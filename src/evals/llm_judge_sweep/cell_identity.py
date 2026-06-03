"""Thin wrapper re-exporting shared cell identity + judge-specific fingerprint.

The generic cell primitives (AdapterSpec, CanonicalCell, format_scale,
sweep_hf_root) moved to :mod:`src.evals.cell_sweep.cell_identity`. This
module re-exports them for existing imports and provides the judge-specific
``rollout_fingerprint`` that pins the text-generation fingerprint fields.
"""

from __future__ import annotations

from src.evals.cell_sweep.cell_identity import (
    AdapterSpec,
    CanonicalCell,
    Tier,
    format_scale,
    sweep_hf_root,
)
from src.evals.cell_sweep.fingerprint import fingerprint_from_fields

__all__ = [
    "AdapterSpec",
    "CanonicalCell",
    "Tier",
    "format_scale",
    "sweep_hf_root",
    "rollout_fingerprint",
]


def rollout_fingerprint(
    *,
    base_model: str,
    dataset_path: str,
    max_samples: int,
    seed: int,
    num_rollouts_per_prompt: int,
    assistant_temperature: float,
    assistant_top_p: float,
    assistant_max_new_tokens: int,
    length: int = 10,
) -> str:
    """Content-addressed fingerprint for judge-sweep rollout-generation settings.

    Fields are judge-sweep-specific: text generation parameters plus dataset
    selection. Does NOT include adapter information — adapters live in the
    cell identity instead.
    """
    return fingerprint_from_fields(
        {
            "base_model": base_model,
            "dataset_path": dataset_path,
            "max_samples": max_samples,
            "seed": seed,
            "num_rollouts_per_prompt": num_rollouts_per_prompt,
            "assistant_temperature": assistant_temperature,
            "assistant_top_p": assistant_top_p,
            "assistant_max_new_tokens": assistant_max_new_tokens,
        },
        length=length,
    )

"""Merge weighted LoRA adapters into a standalone model.

Thin convenience wrapper that binds the suite's reference-resolution policy
(:func:`src.evals.model_resolution.resolve_model_reference`) to the generic
adapter-merging routine in :mod:`src.utils.lora_composition`. The result is a
fully-merged model directory on disk that Inspect can load as a plain HF model
(no PEFT at eval time). Used by
:func:`src.evals.model_materialization.materialize_model`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.evals.config import AdapterConfig
from src.evals.model_resolution import resolve_model_reference
from src.utils.lora_composition import merge_weighted_adapters

logger = logging.getLogger(__name__)


def merge_adapters(
    *,
    base_model: str,
    adapters: list[AdapterConfig],
    output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Path:
    """Load base model + adapters, apply scaling, merge, and save."""
    logger.info("Merging %d LoRA adapter(s) into %s", len(adapters), base_model)
    return merge_weighted_adapters(
        base_model=base_model,
        adapters=adapters,
        output_dir=output_dir,
        dtype=dtype,
        device_map=device_map,
        base_model_resolver=lambda ref: resolve_model_reference(ref, kind="base model"),
        adapter_resolver=lambda ref: resolve_model_reference(ref, kind="adapter"),
    )

"""Bake a linear combination of LoRA adapters into a single vLLM-loadable dir.

vLLM has no multi-adapter composition at inference: it expects one LoRA per
request. To evaluate a combo like ``0.5 * A + 0.5 * B`` under vLLM we pre-merge
the scaled adapters arithmetically into a single adapter and write it to disk
in PEFT format (``adapter_config.json`` + ``adapter_model.safetensors``).

The math is delegated to :class:`~src.utils.lora_vector_utils.LoRaVector`,
which does the right thing with scaling absorption and ``lora_alpha == r``
bookkeeping on save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.utils.lora_vector_utils import LoRaVector


def bake_combined_lora(
    adapter_scales: Sequence[tuple[str, float]],
    output_dir: Path,
    *,
    resolve_to_local: callable | None = None,
) -> tuple[Path, int]:
    """Bake a sum of scaled adapters into a single PEFT adapter directory.

    Produces ``output_dir`` with ``adapter_config.json`` (``lora_alpha == r``
    so vLLM's load-time scaling becomes 1.0) and ``adapter_model.safetensors``.

    Args:
        adapter_scales: Sequence of ``(adapter_ref, scale)`` pairs. Zero-scale
            entries are skipped. If everything is zero (or the sequence is
            empty), raises ``ValueError`` — baseline cells don't need a baked
            adapter, the caller should handle that case without invoking this.
        output_dir: Destination directory. Will be created; if it already
            exists and contains a valid baked adapter it is *not* re-baked
            (idempotent — useful for cache reuse across runs).
        resolve_to_local: Optional function that turns an adapter ref into a
            local directory path (for HF-stored adapters). If ``None`` uses
            ``_resolve_adapter_to_local`` from ``rollout_generation``.

    Returns:
        A ``(output_dir, combined_rank)`` tuple. ``combined_rank`` is the sum
        of the inputs' ranks (so two r=64 adapters combined yield r=128) —
        callers must pass this through to ``max_lora_rank`` in the vLLM
        engine.
    """
    nonzero = [(ref, float(scale)) for ref, scale in adapter_scales if float(scale) != 0.0]
    if not nonzero:
        raise ValueError(
            "bake_combined_lora called with no non-zero adapter scales — "
            "baseline cells should bypass baking entirely."
        )

    output_dir = Path(output_dir)
    if (output_dir / "adapter_config.json").exists() and (
        output_dir / "adapter_model.safetensors"
    ).exists():
        # Idempotent: trust the existing bake, but still compute the combined
        # rank from disk so the caller gets a valid max_lora_rank value.
        import json

        config = json.loads((output_dir / "adapter_config.json").read_text())
        return output_dir, int(config["r"])

    if resolve_to_local is None:
        from src.rollout_generation.model_providers import (
            _resolve_adapter_to_local,
        )

        resolve_to_local = _resolve_adapter_to_local

    combined: LoRaVector | None = None
    for ref, scale in nonzero:
        local_path = resolve_to_local(ref)
        vec = LoRaVector.from_file(local_path)
        scaled = scale * vec
        combined = scaled if combined is None else combined + scaled

    assert combined is not None  # guaranteed by the ``not nonzero`` check above

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_dir)
    return output_dir, combined.max_rank

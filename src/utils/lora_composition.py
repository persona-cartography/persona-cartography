"""Shared helpers for composing weighted LoRA adapters."""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.peft_manipulations import LoRaPipeline, LoRaScaling, set_active_adapters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightedAdapter:
    """A LoRA adapter reference with a scaling factor."""

    path: str
    scale: float = 1.0


def parse_weighted_adapter(raw: str) -> WeightedAdapter:
    """Parse one adapter entry in ``path@scale`` form.

    Args:
        raw: Adapter entry string. ``@scale`` is optional.

    Returns:
        Parsed ``WeightedAdapter``.
    """
    entry = raw.strip()
    if not entry:
        raise ValueError("Adapter entry must not be empty")

    if "@" not in entry:
        return WeightedAdapter(path=entry, scale=1.0)

    path, scale_text = entry.rsplit("@", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"Invalid adapter entry '{raw}': missing adapter path")

    try:
        scale = float(scale_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid adapter scale in '{raw}'. Expected path@float."
        ) from exc

    if not math.isfinite(scale):
        raise ValueError(f"Adapter scale must be finite, got {scale}")

    return WeightedAdapter(path=path, scale=scale)


def split_adapter_reference(path: str) -> tuple[str, str | None]:
    """Split adapter ref into ``(ref, subfolder)`` using ``ref::subfolder`` syntax."""
    if "::" in path:
        ref, subfolder = path.split("::", 1)
        return ref, (subfolder or None)
    return path, None


def resolve_adapter_to_local_dir(
    adapter_path: str,
    *,
    resolver: Callable[[str], str] | None = None,
) -> str:
    """Resolve an adapter ref to a local directory containing ``adapter_config.json``.

    Accepts ``repo_id::subfolder`` for HF repos (tries ``repo_type="dataset"``
    first, then ``"model"``) or a plain local path (optionally with ``::subfolder``).
    For HF refs, the subfolder is fetched via ``snapshot_download`` with
    ``allow_patterns`` scoped to that subfolder. Returns a local path that can
    be passed directly to ``PeftModel.from_pretrained`` / ``load_adapter``
    without needing a ``subfolder=`` kwarg.
    """
    ref, subfolder = split_adapter_reference(adapter_path)
    # Pass the full ref (including any local://... / hf://... prefix) to the
    # resolver so it can disambiguate. Stripping beforehand would hand a bare
    # /abs/path to resolve_model_reference, which then cannot tell a local
    # path apart from an unknown HF repo and raises "Ambiguous ... reference".
    if resolver is not None:
        ref = resolver(ref)
    elif ref.startswith("local://"):
        ref = ref[len("local://") :]
    elif ref.startswith("hf://"):
        ref = ref[len("hf://") :]

    ref_path = Path(ref)
    if ref_path.exists():
        return str(ref_path / subfolder) if subfolder else str(ref_path)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    allow_patterns = [f"{subfolder}/**"] if subfolder else None
    last_err: Exception | None = None
    for repo_type in ("dataset", "model"):
        try:
            local_dir = snapshot_download(
                repo_id=ref,
                repo_type=repo_type,
                allow_patterns=allow_patterns,
            )
            return str(Path(local_dir) / subfolder) if subfolder else local_dir
        except RepositoryNotFoundError as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"Could not find HF repo {ref!r} as dataset or model"
    ) from last_err


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve torch dtype from a string name."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


def normalize_weighted_adapters(adapters: Sequence[object]) -> list[WeightedAdapter]:
    """Normalize adapter-like objects into ``WeightedAdapter`` entries.

    Accepted entries expose ``path`` and optional ``scale`` attributes, for
    example ``src.evals.config.AdapterConfig``.
    """
    normalized: list[WeightedAdapter] = []
    for adapter in adapters:
        path = getattr(adapter, "path", None)
        scale = getattr(adapter, "scale", 1.0)

        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Adapter path must be a non-empty string, got {path!r}")

        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Adapter scale must be numeric, got {scale!r}") from exc

        if not math.isfinite(scale_value):
            raise ValueError(f"Adapter scale must be finite, got {scale_value}")

        normalized.append(WeightedAdapter(path=path.strip(), scale=scale_value))

    return normalized


def _load_first_adapter(
    model,
    *,
    adapter_path: str,
    adapter_name: str,
    resolver: Callable[[str], str] | None,
) -> tuple[PeftModel, str]:
    local_dir = resolve_adapter_to_local_dir(adapter_path, resolver=resolver)

    try:
        peft_model = PeftModel.from_pretrained(
            model,
            local_dir,
            adapter_name=adapter_name,
        )
        return peft_model, local_dir
    except ValueError as exc:
        if "adapter_config.json" not in str(exc):
            raise
        peft_model = PeftModel.from_pretrained(
            model,
            local_dir,
            adapter_name=adapter_name,
            subfolder="adapter",
        )
        return peft_model, local_dir


def _load_extra_adapter(
    model: PeftModel,
    *,
    adapter_path: str,
    adapter_name: str,
    resolver: Callable[[str], str] | None,
) -> str:
    local_dir = resolve_adapter_to_local_dir(adapter_path, resolver=resolver)

    try:
        model.load_adapter(local_dir, adapter_name=adapter_name)
        return local_dir
    except ValueError as exc:
        if "adapter_config.json" not in str(exc):
            raise
        model.load_adapter(
            local_dir,
            adapter_name=adapter_name,
            subfolder="adapter",
        )
        return local_dir


def load_and_scale_adapters(
    model,
    *,
    adapters: Sequence[object],
    adapter_name_prefix: str = "adapter",
    adapter_resolver: Callable[[str], str] | None = None,
) -> tuple[PeftModel, list[str], list[str]]:
    """Load and scale one or more weighted adapters onto a base model.

    Args:
        model: Base ``AutoModelForCausalLM`` instance.
        adapters: Adapter-like objects (path/scale).
        adapter_name_prefix: Prefix for temporary adapter names.
        adapter_resolver: Optional reference resolver.

    Returns:
        Tuple of ``(peft_model, adapter_names, resolved_adapter_refs)``.
    """
    normalized = normalize_weighted_adapters(adapters)
    if not normalized:
        raise ValueError("At least one adapter is required.")

    adapter_names: list[str] = []
    resolved_refs: list[str] = []

    first_name = f"{adapter_name_prefix}_0"
    peft_model, first_resolved_ref = _load_first_adapter(
        model,
        adapter_path=normalized[0].path,
        adapter_name=first_name,
        resolver=adapter_resolver,
    )
    adapter_names.append(first_name)
    resolved_refs.append(first_resolved_ref)

    for i, adapter in enumerate(normalized[1:], 1):
        name = f"{adapter_name_prefix}_{i}"
        resolved_ref = _load_extra_adapter(
            peft_model,
            adapter_path=adapter.path,
            adapter_name=name,
            resolver=adapter_resolver,
        )
        adapter_names.append(name)
        resolved_refs.append(resolved_ref)

    set_active_adapters(peft_model, adapter_names)
    steps = [
        (LoRaScaling, adapter_name, {"scale_factor": adapter.scale})
        for adapter_name, adapter in zip(adapter_names, normalized)
    ]
    pipeline = LoRaPipeline(peft_model, steps)
    pipeline.apply()
    logger.info(
        "Applied LoRA scaling: %s",
        {name: adapter.scale for name, adapter in zip(adapter_names, normalized)},
    )

    return peft_model, adapter_names, resolved_refs


def load_tokenizer_for_composed_model(
    base_model: str, first_adapter_ref: str
) -> AutoTokenizer:
    """Load tokenizer from local adapter dir when available, else from base model."""
    adapter_dir = Path(first_adapter_ref)
    if adapter_dir.exists() and (adapter_dir / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    return AutoTokenizer.from_pretrained(base_model, use_fast=True)


def merge_weighted_adapters(
    *,
    base_model: str,
    adapters: Sequence[object],
    output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    base_model_resolver: Callable[[str], str] | None = None,
    adapter_resolver: Callable[[str], str] | None = None,
) -> Path:
    """Merge weighted adapters into base weights and save artifacts."""
    normalized = normalize_weighted_adapters(adapters)
    if not normalized:
        raise ValueError("At least one adapter is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_model = (
        base_model_resolver(base_model)
        if base_model_resolver is not None
        else base_model
    )
    torch_dtype = resolve_torch_dtype(dtype)

    logger.info(
        "Loading base model %s (dtype=%s, device_map=%s)",
        resolved_base_model,
        dtype,
        device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    peft_model, _, resolved_adapter_refs = load_and_scale_adapters(
        model,
        adapters=normalized,
        adapter_name_prefix="adapter",
        adapter_resolver=adapter_resolver,
    )

    merged = peft_model.merge_and_unload()
    logger.info("Merged %d adapter(s) into base weights", len(normalized))

    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = load_tokenizer_for_composed_model(
        resolved_base_model,
        resolved_adapter_refs[0],
    )
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved merged model to %s", output_dir)

    return output_dir


def delete_materialized_model_dir(
    path: Path | None, *, prune_empty_parent: bool = False
) -> None:
    """Delete merged-model artifacts and optionally remove empty parent dir."""
    if path is None:
        return

    shutil.rmtree(path, ignore_errors=True)
    if not prune_empty_parent:
        return

    parent = path.parent
    if not parent.exists():
        return

    try:
        next(parent.iterdir())
    except StopIteration:
        parent.rmdir()

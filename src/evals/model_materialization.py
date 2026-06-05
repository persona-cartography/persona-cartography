"""Model materialization for suite eval runs.

Turns a :class:`~src.evals.config.ModelSpec` into an Inspect-loadable model
URI. Two cases:

- **No adapters** — the base model is used directly; the URI points at the
  resolved base model.
- **With adapters** — the weighted adapters are merged into the base weights
  and written to a content-addressed cache directory (keyed by a SHA-1 over the
  resolved base model, dtype, device map, and each adapter's resolved
  ref + scale). Repeated runs with identical inputs reuse the cached merge
  instead of re-merging. The cache root defaults to a ``_models_cache`` sibling
  of the run dirs and can be overridden via ``$EVALS_MODEL_CACHE_DIR``.

This is the *materialize-then-load* path (merge to disk, load as plain HF).
The suite's in-process sweep path (live PEFT scaling) lives in
:mod:`src.evals.suite`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.evals.config import ModelSpec
from src.evals.lora_merge import merge_adapters
from src.evals.model_resolution import resolve_model_reference
from src.utils.lora_composition import (
    delete_materialized_model_dir,
    split_adapter_reference,
)

_REQUIRED_ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def _validate_adapter(adapter_path: str) -> None:
    """Sanity-check that an adapter path contains the required PEFT files.

    Works for both local paths and HuggingFace Hub repo IDs.  Raises
    ``FileNotFoundError`` with a descriptive message if required files are
    missing so the eval fails fast before any expensive model loading.

    Args:
        adapter_path: Raw adapter path, optionally with ``::subfolder`` suffix.
    """
    ref, subfolder = split_adapter_reference(adapter_path)
    resolved = resolve_model_reference(ref, kind="adapter")
    local_path = Path(resolved)

    if local_path.exists():
        # Local path — check the filesystem directly.
        check_root = local_path / subfolder if subfolder else local_path
        missing = [f for f in _REQUIRED_ADAPTER_FILES if not (check_root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Adapter at '{check_root}' is missing required files: {missing}. "
                "Check that the path points to a valid PEFT adapter directory."
            )
    else:
        # HuggingFace Hub repo — use the Hub API to check file presence.
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            return  # huggingface_hub unavailable; skip check

        api = HfApi()
        try:
            repo_files = {f.rfilename for f in api.list_repo_tree(resolved, recursive=True)}
        except RepositoryNotFoundError:
            raise FileNotFoundError(
                f"Adapter HuggingFace repo not found: '{resolved}'. "
                "Check the repo ID and that you have access."
            )
        except Exception:
            return  # Network or auth issue; skip check and let PEFT surface the error

        prefix = f"{subfolder}/" if subfolder else ""
        missing = [f for f in _REQUIRED_ADAPTER_FILES if f"{prefix}{f}" not in repo_files]
        if missing:
            raise FileNotFoundError(
                f"Adapter repo '{resolved}' (subfolder='{subfolder}') is missing "
                f"required files: {missing}. "
                "Check that the repo contains a valid PEFT adapter."
            )


@dataclass(frozen=True)
class MaterializedModel:
    """A model spec resolved to an Inspect URI, plus merge cache metadata."""

    model_name: str
    model_spec_name: str
    model_uri: str
    cache_key: str
    materialized_path: Path | None


def _adapter_ref_for_key(path: str) -> str:
    """Resolve an adapter ref to its canonical ``ref[::subfolder]`` cache form."""
    ref, subfolder = split_adapter_reference(path)
    resolved_ref = resolve_model_reference(ref, kind="adapter")
    if subfolder:
        return f"{resolved_ref}::{subfolder}"
    return resolved_ref


def _compute_model_key(model: ModelSpec) -> str:
    """Content-address a model spec (base + dtype + device map + adapters)."""
    resolved_base = resolve_model_reference(model.base_model, kind="base model")
    payload = {
        "base_model": resolved_base,
        "dtype": model.dtype,
        "device_map": model.device_map,
        "adapters": [
            {"path": _adapter_ref_for_key(adapter.path), "scale": adapter.scale}
            for adapter in model.adapters
        ],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _model_uri_for(base: str | Path) -> str:
    """Format a base model path/id as an Inspect ``hf/...`` model URI."""
    return f"hf/{base}"


def _models_cache_root(output_root: Path) -> Path:
    """Resolve stable cache root for merged models.

    Defaults to a suite-level shared cache directory (sibling of run dirs), so
    repeated runs with different timestamps reuse the same merged artifacts.
    """
    override = os.environ.get("EVALS_MODEL_CACHE_DIR")
    if override:
        return Path(override)
    return output_root.parent / "_models_cache"


def materialize_model(model: ModelSpec, output_root: Path) -> MaterializedModel:
    """Materialize a model spec into an Inspect model URI."""
    cache_key = _compute_model_key(model)
    resolved_base_model = resolve_model_reference(model.base_model, kind="base model")

    if not model.adapters:
        return MaterializedModel(
            model_name=model.base_model,
            model_spec_name=model.name,
            model_uri=_model_uri_for(resolved_base_model),
            cache_key=cache_key,
            materialized_path=None,
        )

    for adapter in model.adapters:
        _validate_adapter(adapter.path)

    models_root = _models_cache_root(output_root)
    models_root.mkdir(parents=True, exist_ok=True)
    target_dir = models_root / cache_key

    config_path = target_dir / "config.json"
    if not config_path.exists():
        if target_dir.exists():
            delete_materialized_model_dir(target_dir)
        try:
            merge_adapters(
                base_model=resolved_base_model,
                adapters=model.adapters,
                output_dir=target_dir,
                dtype=model.dtype,
                device_map=model.device_map,
            )
        except Exception:
            # Remove partial artifacts (for example, interrupted shard writes).
            delete_materialized_model_dir(target_dir)
            raise

    return MaterializedModel(
        model_name=model.base_model,
        model_spec_name=model.name,
        model_uri=_model_uri_for(target_dir),
        cache_key=cache_key,
        materialized_path=target_dir,
    )

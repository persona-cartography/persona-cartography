"""Shared condition setup for vanilla / activation-capping / LoRA-soup inference.

Originally lived in ``scripts_dev/persona_drift_assistant_axis/run_drift.py``
as private underscore-prefixed helpers. Pulled out here so the persona-
jailbreak eval (and any future persona-mitigation experiment) can reuse
the exact same plumbing — same vLLM/HF mixed-engine handling, same
LoRA-soup baking, same capping-hook lifecycle, same fork-safety env-var.

Three condition families are supported:

    vanilla              — base HF model on vLLM
    lora_soup            — baked-merged LoRA on vLLM (rank = sum of inputs)
    activation_capping   — base HF model with paper Eq. 1 floor capping hooks

The "soup" baking and capping installation each have one-time costs and
GPU lifecycle constraints (vLLM and HF cannot share GPU memory cleanly),
so this module also exposes:

    * ``ensure_vllm_fork_safe()`` — sets ``VLLM_WORKER_MULTIPROC_METHOD=spawn``
      before any vLLM import.
    * ``sort_conditions_for_safety()`` — vLLM-engine conditions first, HF
      conditions last, so HF capping doesn't sit resident while vLLM
      tries to allocate.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from pydantic import BaseModel

from src.common.config import GenerationConfig
from src.inference.config import (
    InferenceConfig,
    LocalProviderConfig,
    VllmProviderConfig,
)
from src.utils.lora_combo_baking import bake_combined_lora

# Assistant Axis imports are done lazily inside the capping helpers below.
# This keeps the non-capping parts of this module importable until the pinned
# upstream checkout is needed.


# ── Public config type ────────────────────────────────────────────────────


class ConditionConfig(BaseModel):
    """Inference-relevant subset of a larger experiment config.

    Callers assemble one of these from whatever broader config they have
    (ExperimentConfig in run_drift.py, JailbreakEvalConfig in the new
    eval, etc.). Fields are exactly the knobs the condition setup
    helpers need — no more, no less.
    """

    base_model: str
    """HF model id (e.g. ``meta-llama/Llama-3.1-8B-Instruct``)."""

    # vLLM knobs (used by vanilla + lora_soup)
    vllm_gpu_memory_utilization: float = 0.50
    vllm_max_model_len: int = 8192
    vllm_max_concurrent: int = 32
    vllm_batch_size: int = 8

    # HF knobs (used by activation_capping)
    hf_batch_size: int = 8
    hf_max_concurrent: int = 8

    # Generation knobs (shared across conditions)
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0


# ── Fork-safety + condition ordering ─────────────────────────────────────


def ensure_vllm_fork_safe() -> None:
    """Force vLLM EngineCore to use spawn instead of fork.

    Must be called BEFORE any vLLM import. We seed CUDA in the parent
    process, which initializes the CUDA context; a forked child crashes
    with ``Cannot re-initialize CUDA in forked subprocess``.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# vLLM-engine conditions first; HF conditions last. Avoids holding the HF
# capping model in GPU memory while vLLM allocates its own.
_VLLM_CONDITIONS: tuple[str, ...] = ("vanilla",)
_HF_CONDITIONS: tuple[str, ...] = ("activation_capping",)


def is_vllm_condition(condition: str) -> bool:
    """Return True if the condition runs on vLLM (vanilla or any lora_soup_*)."""
    return condition in _VLLM_CONDITIONS or condition.startswith("lora_soup")


def is_hf_condition(condition: str) -> bool:
    """Return True if the condition runs on HF transformers."""
    return condition in _HF_CONDITIONS


def sort_conditions_for_safety(conditions: Sequence[str]) -> tuple[str, ...]:
    """Order conditions so vLLM ones run before HF ones.

    Order within each engine bucket is preserved (stable sort).
    """
    return tuple(sorted(conditions, key=lambda c: (0 if is_vllm_condition(c) else 1)))


# ── vLLM-backed condition setup ──────────────────────────────────────────


def setup_vanilla_inference(cfg: ConditionConfig) -> InferenceConfig:
    """Build a vLLM ``InferenceConfig`` for the vanilla base model."""
    return _vllm_inference_config(cfg, adapter_path=None, max_lora_rank=64)


def setup_lora_soup_inference(
    cfg: ConditionConfig,
    *,
    adapter_specs: Sequence[tuple[str, float]],
    output_dir: Path,
) -> tuple[InferenceConfig, Path, int]:
    """Bake the LoRA soup (or reuse a cached bake) and build a vLLM
    ``InferenceConfig`` pointing at the merged adapter.

    Args:
        cfg: shared condition config.
        adapter_specs: list of ``(adapter_ref, scale)`` pairs. ``adapter_ref``
            may be a local path or a ``repo_id::subfolder`` HF reference.
        output_dir: where to write the baked adapter. If
            ``adapter_config.json`` already exists there, the bake is skipped
            and the cached rank is read back.

    Returns:
        ``(inference_config, baked_adapter_dir, combined_rank)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_cfg = output_dir / "adapter_config.json"
    if cached_cfg.exists():
        rank = int(json.loads(cached_cfg.read_text())["r"])
        baked_dir = output_dir
    else:
        specs_list = list(adapter_specs)
        baked_dir, rank = bake_combined_lora(specs_list, output_dir)
    inf = _vllm_inference_config(cfg, adapter_path=str(baked_dir), max_lora_rank=rank)
    return inf, baked_dir, rank


def _vllm_inference_config(
    cfg: ConditionConfig, *, adapter_path: str | None, max_lora_rank: int,
) -> InferenceConfig:
    return InferenceConfig(
        model=cfg.base_model,
        provider="vllm",
        generation=GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            batch_size=cfg.vllm_batch_size,
            num_responses_per_prompt=1,
        ),
        max_concurrent=cfg.vllm_max_concurrent,
        vllm=VllmProviderConfig(
            adapter_path=adapter_path,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_model_len=cfg.vllm_max_model_len,
            max_loras=1,
            max_lora_rank=max_lora_rank,
        ),
    )


# ── HF + activation-capping condition setup ──────────────────────────────


@dataclass
class CappingPreload:
    """Handle bundling the capped HF model + tokenizer + hook handle.

    The caller must keep this object alive for the lifetime of the hooks.
    Call :func:`release_capping` when done to detach the hooks and free
    GPU memory.
    """

    model: Any
    tokenizer: Any
    capping_handle: Any
    axis: torch.Tensor
    capping_config: dict


def load_capped_model(
    cfg: ConditionConfig,
    *,
    axis_path: Path,
    capping_config_path: Path,
    run_diagnostic: bool = True,
) -> CappingPreload:
    """Load HF base model, register persistent capping hooks, run the
    sign/direction diagnostic, return a :class:`CappingPreload` handle.

    The diagnostic aborts the process with a clear error if pre/post
    projections don't match the configured mode — saves us from running
    a long eval with a wrong-sign cap.
    """
    # Lazy import: pulls in the vendor's optional plotly dep. Only the
    # capping path needs it.
    from src.activation_capping.assistant_axis_loader import (
        apply_assistant_axis_capping,
        diagnose_capping_direction,
        load_axis,
        load_capping_config,
        print_capping_diagnosis,
        remove_capping_hooks,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    capping_cfg = load_capping_config(capping_config_path)
    axis = load_axis(axis_path)

    print(f"  Loading HF {cfg.base_model} for capping condition...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    capping_handle = apply_assistant_axis_capping(model, axis, capping_cfg, debug=True)
    print(
        f"  Capping mode={capping_cfg.get('mode', 'floor')!r} "
        f"on layers {capping_cfg['layers']}"
    )

    if run_diagnostic:
        print("  Running cap-direction diagnostic...")
        report = diagnose_capping_direction(
            model, tokenizer, capping_handle,
            axis=axis, capping_config=capping_cfg,
        )
        print_capping_diagnosis(report)
        if not report["passed"]:
            try:
                remove_capping_hooks(capping_handle)
            except Exception:  # noqa: BLE001
                pass
            raise SystemExit(
                "Cap-direction diagnostic FAILED. "
                "Re-check the axis sign, threshold percentile, and mode in "
                "capping_config.pt. STOPPING before spending GPU on inference."
            )

    return CappingPreload(
        model=model,
        tokenizer=tokenizer,
        capping_handle=capping_handle,
        axis=axis,
        capping_config=capping_cfg,
    )


def setup_capping_inference(
    cfg: ConditionConfig, preload: CappingPreload,
) -> InferenceConfig:
    """Build an ``InferenceConfig`` that drives the already-capped HF model."""
    return InferenceConfig(
        model=cfg.base_model,
        provider="local",
        generation=GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_responses_per_prompt=1,
            batch_size=cfg.hf_batch_size,
        ),
        max_concurrent=cfg.hf_max_concurrent,
        local=LocalProviderConfig(preloaded_model=(preload.model, preload.tokenizer)),
    )


def release_capping(preload: CappingPreload | None) -> None:
    """Detach capping hooks and free GPU memory.

    Idempotent: safe to call on ``None`` or on a preload whose hooks have
    already been removed.
    """
    if preload is None:
        return
    from src.activation_capping.assistant_axis_loader import remove_capping_hooks
    try:
        remove_capping_hooks(preload.capping_handle)
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: failed to detach capping hooks: {exc}")
    try:
        preload.model.cpu()
    except Exception:  # noqa: BLE001
        pass
    # Drop strong refs so GC can reclaim.
    preload.model = None  # type: ignore[assignment]
    preload.tokenizer = None  # type: ignore[assignment]
    preload.capping_handle = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "ConditionConfig",
    "CappingPreload",
    "ensure_vllm_fork_safe",
    "sort_conditions_for_safety",
    "is_vllm_condition",
    "is_hf_condition",
    "setup_vanilla_inference",
    "setup_lora_soup_inference",
    "setup_capping_inference",
    "load_capped_model",
    "release_capping",
]

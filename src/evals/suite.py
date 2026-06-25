"""Suite orchestration for Inspect-based eval runs.

The engine that turns a :class:`~src.evals.config.SuiteConfig` into a tree of
per-(model, eval) result directories. :func:`run_eval_suite` is the single
entry point; everything else here supports it.

Execution model
---------------
``config.expand_models()`` yields one :class:`~src.evals.config.ModelSpec` per
point (a literal model, a LoRA scale point, or an activation-cap fraction). For
each spec the suite prepares a model — see the ``_prepare_*`` / ``_load_*``
helpers, each returning a :class:`_PreparedModel` — and runs every eval against
it via the Inspect backend (:mod:`src.evals.backends.inspect_runner`).

Three sweep fast-paths load the heavy base model **once** and reuse it across
points, mutating only the cheap per-point state:

- **LoRA scale sweep** — load base + adapter once; apply/restore ``LoRaScaling``
  per scale point (``_load_local_model_for_sweep`` / ``_prepare_sweep_model``).
- **Activation-cap sweep** — load the bare base once; register/remove capping
  hooks per fraction (``_load_base_model_for_activation_cap`` /
  ``_prepare_activation_cap_model``).
- **vLLM scale sweep** — activate a pre-baked LoRA variant per scale point
  (``_prepare_vllm_sweep_model``).

GPU memory is managed carefully: the model is *not* evicted between evals of the
same spec (that would silently move it to CPU); cleanup happens in the per-spec
``finally`` block (``_cleanup_runtime_model_state``), and the shared sweep model
stays on GPU until the whole sweep finishes.

Idempotency & artifacts
------------------------
With ``skip_completed`` the suite reuses prior ``run_info.json`` results (locally
or rehydrated from HF), and a zero-LoRA "baseline" result is cached/uploaded so
sibling runs can skip recomputing the base model (see the baseline-caching
section). After the run it can auto-analyze (plots) and upload the run dir to a
HuggingFace dataset repo.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from inspect_ai.model import Model, get_model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.peft_manipulations import LoRaScaling
from src.evals.backends.inspect_runner import (
    run_benchmark_eval,
    run_custom_eval,
    score_custom_eval_from_log,
)
from src.evals.config import (
    InspectBenchmarkSpec,
    InspectCustomEvalSpec,
    JudgeExecutionConfig,
    ModelSpec,
    RunSummaryRow,
    SuiteConfig,
    SuiteResult,
)
from src.evals.cell_sweep.fingerprint import fingerprint_from_fields
from src.evals.model_resolution import resolve_model_reference
from src.evals.utils.preloaded_hf_provider import (
    clear_tokenization_cache,
    register_preloaded_hf_provider,
)
from src.evals.utils.vllm_preloaded_provider import register_vllm_preloaded_provider
from src.utils.lora_composition import load_and_scale_adapters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact ``1h02m03s`` / ``2m03s`` / ``1.5s`` string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s"


# ---------------------------------------------------------------------------
# GPU / Inspect cache cleanup
# ---------------------------------------------------------------------------


def _cleanup_runtime_model_state(move_to_cpu: bool = True) -> None:
    """Release Inspect's in-process model cache and free GPU memory.

    Args:
        move_to_cpu: If True (default), move HF model weights to CPU before
            closing the provider.  Set False when the underlying model must
            stay on GPU (e.g. between sweep combos that share the same
            PeftModel instance).
    """
    try:
        from inspect_ai.model import _model as inspect_model_impl

        active_model_cv = getattr(inspect_model_impl, "active_model_context_var", None)
        if active_model_cv is not None and hasattr(active_model_cv, "set"):
            active_model_cv.set(None)

        model_roles_cv = getattr(inspect_model_impl, "_model_roles", None)
        if model_roles_cv is not None and hasattr(model_roles_cv, "set"):
            model_roles_cv.set({})

        cached_models = getattr(inspect_model_impl, "_models", None)
        if isinstance(cached_models, dict):
            for model in list(cached_models.values()):
                api = getattr(model, "api", None)
                if move_to_cpu:
                    # The HF provider's batch thread holds a dangling generator reference
                    # that keeps the model on GPU. Moving to CPU before close() frees VRAM.
                    hf_model = getattr(api, "model", None)
                    if hf_model is not None and callable(
                        getattr(hf_model, "cpu", None)
                    ):
                        try:
                            hf_model.cpu()
                        except Exception:
                            pass
                close = getattr(api, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            cached_models.clear()
    except Exception:
        pass

    try:
        import gc

        gc.collect()
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model preparation — the unified path
# ---------------------------------------------------------------------------


@dataclass
class _PreparedModel:
    """A model ready to be passed to Inspect, plus optional cleanup."""

    # Either a URI string (API / resume) or a live Model object (local HF).
    inspect_model: str | Model
    # Non-None only when LoRaScaling was applied; must be restored after the eval.
    scaler: LoRaScaling | None
    # The underlying PeftModel, kept so we can move it off GPU after the suite.
    peft_model: PeftModel | None
    # Human-readable name for logging.
    model_name: str
    # Non-None only when ActivationCappedModel was applied; hooks must be removed after the eval.
    cap_model: Any = None


def _resolve_dtype(spec: ModelSpec) -> torch.dtype:
    """Resolve a ``ModelSpec.dtype`` string (e.g. ``"bfloat16"``) to a ``torch.dtype``."""
    dtype = getattr(torch, spec.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {spec.dtype!r}")
    return dtype


def _load_local_model(
    spec: ModelSpec,
    batch_size: int | None,
    enable_thinking: bool | None = None,
) -> _PreparedModel:
    """Load a local HF model (with optional adapters) into GPU memory."""
    register_preloaded_hf_provider()

    base_ref = resolve_model_reference(spec.base_model, kind="base model")
    torch_dtype = _resolve_dtype(spec)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_ref,
        torch_dtype=torch_dtype,
        device_map=spec.device_map,
        **_flash_attn_kwargs(),
    )

    scaler: LoRaScaling | None = None

    if spec.adapters:
        from src.evals.model_resolution import resolve_model_reference as _resolve
        from src.utils.lora_composition import normalize_weighted_adapters

        normalized = normalize_weighted_adapters(spec.adapters)
        peft_model, adapter_names, _ = load_and_scale_adapters(
            base_model,
            adapters=normalized,
            adapter_name_prefix="adapter",
            adapter_resolver=lambda ref: _resolve(ref, kind="adapter"),
        )
        # Try loading the tokenizer from the first adapter (HF model repos
        # often bundle tokenizer files alongside the adapter).  Fall back to
        # the base model when the adapter directory has no usable tokenizer
        # (common for bare LoRA dirs downloaded from dataset repos).
        try:
            tokenizer_ref = resolve_model_reference(normalized[0].path, kind="adapter")
        except Exception:
            tokenizer_ref = base_ref
    else:
        peft_model = PeftModel.__new__(PeftModel)
        # No adapters — wrap as a plain model; use a lightweight shim instead.
        # Actually for no-adapter case we skip PeftModel entirely.
        peft_model = base_model  # type: ignore[assignment]
        tokenizer_ref = base_ref

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_ref)

    inspect_model = get_model(
        f"hf_preloaded/{spec.name}",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
        batch_size=batch_size or 32,
        enable_thinking=enable_thinking,
    )

    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=scaler,
        peft_model=peft_model,
        model_name=spec.base_model,
    )


_SWEEP_ADAPTER_NAME = "default"


def _flash_attn_kwargs() -> dict[str, str]:
    """Return attn_implementation=flash_attention_2 if flash_attn is installed."""
    try:
        import flash_attn  # noqa: F401

        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        return {}


def _load_local_model_for_sweep(
    base_model_ref: str,
    adapter_local_dir: str,
    dtype: torch.dtype,
    fixed_adapters: list | None = None,
) -> tuple[PeftModel, Any]:
    """Load base model + single adapter once for a scale sweep.

    The adapter is always loaded under ``_SWEEP_ADAPTER_NAME`` so that
    ``_prepare_sweep_model`` can reference the same name without coupling.

    Args:
        adapter_local_dir: Local directory containing ``adapter_config.json``.
            The caller is responsible for snapshot-downloading HF refs via
            ``resolve_adapter_to_local_dir`` before passing them here.
        fixed_adapters: Optional list of AdapterConfig to merge into the
            base weights before loading the sweep adapter.  This allows
            sweeping one LoRA on top of a fixed persona LoRA.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        torch_dtype=dtype,
        device_map="auto",
        **_flash_attn_kwargs(),
    )

    # Merge fixed adapters into base weights so they are baked in
    # before the sweep adapter is loaded on top.
    if fixed_adapters:
        from src.utils.lora_composition import (
            load_and_scale_adapters,
            normalize_weighted_adapters,
        )

        normalized = normalize_weighted_adapters(fixed_adapters)
        peft_model, _, _ = load_and_scale_adapters(
            base_model,
            adapters=normalized,
            adapter_name_prefix="fixed",
        )
        base_model = peft_model.merge_and_unload()
        print(
            f"  merged {len(normalized)} fixed adapter(s) into base weights",
            flush=True,
        )

    peft_model = PeftModel.from_pretrained(
        base_model, adapter_local_dir, adapter_name=_SWEEP_ADAPTER_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_local_dir)
    return peft_model, tokenizer


def _load_base_model_for_activation_cap(
    base_model_ref: str,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    """Load a bare base model (no adapter) for an activation capping sweep.

    The model is loaded once and reused across all fraction points; capping
    hooks are registered/removed per fraction via ``_prepare_activation_cap_model``.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        torch_dtype=dtype,
        device_map="auto",
        **_flash_attn_kwargs(),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_ref)
    return base_model, tokenizer


def _prepare_activation_cap_model(
    spec: ModelSpec,
    base_model: Any,
    tokenizer: Any,
    axis: Any,
    per_layer_range: dict,
    capping_layers: list[int],
    batch_size: int | None,
    *,
    ceiling_from_hi: bool = True,
    enable_thinking: bool | None = None,
) -> _PreparedModel:
    """Wrap the base model with ActivationCappedModel for this fraction point.

    The fraction is stored in ``spec.scale``.  Positive fractions use floor
    mode (push activations toward the trait direction); negative fractions use
    ceiling mode (suppress the trait).  The base model spec (scale=None) is
    run uncapped.
    """
    from src.activation_capping.model import (
        ActivationCappedModel,
        compute_thresholds_at_fraction,
    )

    register_preloaded_hf_provider()

    fraction = spec.scale  # None for the base model spec

    if fraction is None:
        # Base model: run without any capping hooks.
        inspect_model = get_model(
            f"hf_preloaded/{spec.name}",
            hf_model=base_model,
            hf_tokenizer=tokenizer,
            batch_size=batch_size or 32,
            enable_thinking=enable_thinking,
        )
        return _PreparedModel(
            inspect_model=inspect_model,
            scaler=None,
            peft_model=None,
            model_name=spec.base_model,
            cap_model=None,
        )

    mode = "floor" if fraction >= 0 else "ceiling"
    filtered_range = {
        layer: per_layer_range[layer]
        for layer in capping_layers
        if layer in per_layer_range
    }
    layer_thresholds = compute_thresholds_at_fraction(
        filtered_range,
        fraction,
        ceiling_from_hi=ceiling_from_hi,
    )
    cap_model = ActivationCappedModel(base_model, axis, layer_thresholds, mode=mode)

    inspect_model = get_model(
        f"hf_preloaded/{spec.name}",
        hf_model=cap_model,
        hf_tokenizer=tokenizer,
        batch_size=batch_size or 32,
        enable_thinking=enable_thinking,
    )
    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
        cap_model=cap_model,
    )


def _prepare_sweep_model(
    spec: ModelSpec,
    peft_model: PeftModel,
    tokenizer: Any,
    batch_size: int | None,
    cache_tokenization: bool = True,
    enable_thinking: bool | None = None,
) -> _PreparedModel:
    """Wrap a sweep model spec: apply LoRaScaling for this scale point.

    For the base model (scale=None), scale_factor=0.0 is used to zero out
    the adapter contribution rather than running with it at its default scale.
    """
    register_preloaded_hf_provider()

    # Base model (scale=None) must zero out the adapter so it doesn't contribute.
    effective_scale = spec.scale if spec.scale is not None else 0.0
    scaler = LoRaScaling(
        peft_model,
        adapter_name=_SWEEP_ADAPTER_NAME,
        scale_factor=effective_scale,
    ).apply()

    inspect_model = get_model(
        f"hf_preloaded/{spec.name}",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
        batch_size=batch_size or 32,
        cache_tokenization=cache_tokenization,
        enable_thinking=enable_thinking,
    )
    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=scaler,
        peft_model=peft_model,
        model_name=spec.base_model,
    )


def _prepare_api_model(spec: ModelSpec) -> _PreparedModel:
    """Wrap an API model spec (model_uri already set)."""
    assert spec.model_uri is not None
    return _PreparedModel(
        inspect_model=spec.model_uri,
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
    )


def _prepare_vllm_sweep_model(
    spec: ModelSpec,
    vllm_provider: Any,
    batch_size: int | None,
) -> _PreparedModel:
    """Wrap a vllm sweep model spec: activate the pre-baked variant for this scale point.

    Args:
        spec: ModelSpec for this scale point (spec.scale is the target scale, or None for base).
        vllm_provider: Active VLLMLoRaScaleProvider (already entered via __enter__).
        batch_size: Override batch size for Inspect.
    """
    register_vllm_preloaded_provider()

    effective_scale = spec.scale if spec.scale is not None else 0.0
    variant = str(effective_scale)

    # Retrieve the pre-built _VllmVariantProvider for this scale from the provider.
    # We access the internal lora_requests dict directly to get the right variant
    # without managing a context manager per scale point.

    from src.rollout_generation.model_providers import _VllmVariantProvider

    lora_request = vllm_provider._lora_requests.get(variant)
    if lora_request is None and effective_scale == 0.0:
        # Base model: use a null lora_request (vllm will run without adapter).
        lora_request = None
    elif lora_request is None:
        raise KeyError(
            f"No baked adapter found for scale={effective_scale}. "
            f"Available: {list(vllm_provider._lora_requests.keys())}"
        )

    variant_provider = _VllmVariantProvider(
        llm=vllm_provider._llm,
        lora_request=lora_request,
        SamplingParams=vllm_provider._SamplingParams,
        temperature=0.0,  # overridden per-call via GenerateConfig
        top_p=1.0,
        max_new_tokens=512,
    )

    inspect_model = get_model(
        f"vllm_preloaded/{spec.name}",
        vllm_variant_provider=variant_provider,
        batch_size=batch_size or 32,
    )
    return _PreparedModel(
        inspect_model=inspect_model,
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
    )


def _prepare_resume_model(spec: ModelSpec) -> _PreparedModel:
    """Wrap a resume-mode spec: no model load, just point Inspect at the base model URI."""
    return _PreparedModel(
        inspect_model=f"hf/{spec.base_model}",
        scaler=None,
        peft_model=None,
        model_name=spec.base_model,
    )


# ---------------------------------------------------------------------------
# Helpers carried over from the original suite
# ---------------------------------------------------------------------------


def load_suite_module(module_path: str) -> tuple[SuiteConfig, JudgeExecutionConfig]:
    """Load SUITE_CONFIG (and optional JUDGE_EXEC_CONFIG) from a module."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "SUITE_CONFIG"):
        raise AttributeError(f"Module '{module_path}' must export SUITE_CONFIG")
    suite_config = getattr(module, "SUITE_CONFIG")
    if not isinstance(suite_config, SuiteConfig):
        raise TypeError(f"SUITE_CONFIG must be SuiteConfig, got {type(suite_config)}")
    judge_exec = getattr(module, "JUDGE_EXEC_CONFIG", JudgeExecutionConfig())
    if not isinstance(judge_exec, JudgeExecutionConfig):
        raise TypeError("JUDGE_EXEC_CONFIG must be JudgeExecutionConfig when provided")
    return suite_config, judge_exec


def _make_output_root(config: SuiteConfig, mode: str) -> Path:
    """Resolve (and create) the suite's output root dir, honoring resume mode's run_name."""
    if mode == "resume":
        if not config.run_name:
            raise ValueError(
                "run_name must be set in SuiteConfig when mode='resume' "
                "so the prior run directory can be located."
            )
        run_name = config.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or timestamp
    output_root = config.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _maybe_rehydrate_from_hf(
    config: SuiteConfig,
    output_root: Path,
) -> bool:
    """Download prior results from HF if they exist remotely but not locally.

    When ``skip_completed`` is enabled and an ``upload_repo_id`` /
    ``upload_path_in_repo`` are configured, this checks whether the local
    ``output_root`` already contains run data.  If not, it attempts to
    download previously-uploaded results from HuggingFace so that the suite
    can skip those runs instead of regenerating them.

    Returns:
        True if data was downloaded, False otherwise.
    """
    if not (
        config.skip_completed and config.upload_repo_id and config.upload_path_in_repo
    ):
        return False

    # Check if local data already exists
    existing = list(output_root.glob("**/run_info.json"))
    if existing:
        return False  # already have local data

    # Template case ({eval_name}) not yet supported — skip.
    if "{eval_name}" in config.upload_path_in_repo:
        return False

    hf_path = f"{config.upload_path_in_repo}/{output_root.name}"

    try:
        from src.utils.hf_hub import (
            check_exists_in_dataset_repo,
            download_path_to_dir,
        )

        if not check_exists_in_dataset_repo(
            repo_id=config.upload_repo_id, path_in_repo=hf_path
        ):
            return False

        print(
            f"  Downloading prior results from {config.upload_repo_id}/{hf_path} ...",
            flush=True,
        )
        # Rehydrate only the files needed for skip-decision + downstream analysis.
        # The large Inspect `.eval` SQLite logs (used only by `inspect view`) and
        # per-execution trace logs aren't consulted by the suite or analyze_results,
        # so pulling them individually (serial hf_hub_download) is pure overhead.
        download_path_to_dir(
            repo_id=config.upload_repo_id,
            path_in_repo=hf_path,
            target_dir=output_root,
            allow_patterns=[
                "**/run_info.json",  # skip-decision metadata
                "**/last-eval-result",  # terminal-status marker
                "**/native/inspect_logs/*.json",  # analyze_results input
                "**/figures/*.png",  # pre-generated plots
                "**/figures/*.pdf",
            ],
        )
        n_runs = len(list(output_root.glob("**/run_info.json")))
        print(f"  ✓ Rehydrated {n_runs} run(s) from HuggingFace", flush=True)
        return True
    except Exception as exc:
        print(f"  WARNING: HF rehydration failed: {exc}", flush=True)
        return False


def _run_dir_for(
    *,
    output_root: Path,
    model_spec_name: str,
    eval_name: str,
    run_index: int = 0,
) -> Path:
    """Compute (and create) the result dir for one (model spec, eval, run index)."""
    suffix = f"/run_{run_index:02d}" if run_index > 0 else ""
    run_dir = output_root / model_spec_name / f"{eval_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_scale_in_eval(
    scale: float | None,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    suite_sweep: Any,
) -> bool:
    """Return True if *scale* is part of this eval's scale grid."""
    if scale is None:
        return True
    if not isinstance(eval_spec, InspectBenchmarkSpec):
        return True
    effective_sweep = eval_spec.sweep if eval_spec.sweep is not None else suite_sweep
    if effective_sweep is None:
        return True
    return scale in set(effective_sweep.scale_points())


def _ensure_hf_log_repo(hf_log_dir: str) -> None:
    """Create the HF dataset repo backing an ``hf://datasets/...`` log dir, if needed."""
    prefix = "hf://datasets/"
    if not hf_log_dir.startswith(prefix):
        return
    parts = hf_log_dir[len(prefix) :].split("/")
    if len(parts) < 2:
        return
    repo_id = f"{parts[0]}/{parts[1]}"
    try:
        from huggingface_hub import HfApi

        HfApi().create_repo(
            repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True
        )
    except Exception:
        pass


def _summary_row(
    *,
    model_name: str,
    model_spec_name: str,
    eval_name: str,
    eval_kind: str,
    status: str,
    output_dir: Path,
    run_info_path: Path | None,
    inspect_log_path: str | None,
    error: str | None = None,
) -> RunSummaryRow:
    """Build a :class:`RunSummaryRow` for the suite result, stringifying paths."""
    return RunSummaryRow(
        model_name=model_name,
        model_spec_name=model_spec_name,
        eval_name=eval_name,
        eval_kind=eval_kind,
        status=status,
        output_dir=str(output_dir),
        run_info_path=str(run_info_path) if run_info_path else None,
        inspect_log_path=inspect_log_path,
        error=error,
    )


def _eval_spec_matches(
    run_info: dict,
    eval_spec: "InspectBenchmarkSpec | InspectCustomEvalSpec",
) -> bool:
    """Check whether a cached run_info's eval_spec matches the current one.

    Compares the serialized eval spec stored in run_info.json against the
    current eval spec.  Returns False (= re-run needed) if the stored spec
    is missing or differs in any field.
    """
    stored = run_info.get("eval_spec")
    if stored is None:
        return False
    current = eval_spec.model_dump(mode="json")
    return stored == current


def _git_hash() -> str:
    """Current git HEAD, or 'unknown' (provenance for uploaded run_info)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def _write_run_info(
    *,
    run_dir: Path,
    output_root: Path,
    model_spec: ModelSpec,
    eval_spec: InspectBenchmarkSpec | InspectCustomEvalSpec,
    judge_exec: JudgeExecutionConfig,
    prepared: _PreparedModel,
    status: str,
    error: str | None,
    inspect_log_path: str | None,
    inspect_status: str | None,
) -> Path:
    """Write ``run_info.json`` (status + serialized specs + Inspect log info) for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_info.json"
    payload = {
        "suite_run_name": output_root.name,
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "status": status,
        "error": error,
        "model_spec": model_spec.model_dump(mode="json"),
        "eval_spec": eval_spec.model_dump(mode="json"),
        "judge_execution": judge_exec.model_dump(mode="json"),
        "scale": model_spec.scale,
        "native": {
            "inspect_log_path": inspect_log_path,
            "inspect_status": inspect_status,
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _record_failed_model_rows(
    *,
    rows: list[RunSummaryRow],
    output_root: Path,
    model_spec: ModelSpec,
    evals: list[InspectBenchmarkSpec | InspectCustomEvalSpec],
    judge_exec: JudgeExecutionConfig,
    prepared: _PreparedModel | None,
    error: str,
) -> None:
    """Record a ``failed`` row + run_info for every eval when a model load fails."""
    for eval_spec in evals:
        run_dir = _run_dir_for(
            output_root=output_root,
            model_spec_name=model_spec.name,
            eval_name=eval_spec.name,
        )
        eval_kind = (
            "benchmark" if isinstance(eval_spec, InspectBenchmarkSpec) else "custom"
        )
        run_info_path = _write_run_info(
            run_dir=run_dir,
            output_root=output_root,
            model_spec=model_spec,
            eval_spec=eval_spec,
            judge_exec=judge_exec,
            prepared=prepared
            or _PreparedModel(
                inspect_model="",
                scaler=None,
                peft_model=None,
                model_name=model_spec.base_model,
            ),
            status="failed",
            error=error,
            inspect_log_path=None,
            inspect_status=None,
        )
        rows.append(
            _summary_row(
                model_name=model_spec.base_model,
                model_spec_name=model_spec.name,
                eval_name=eval_spec.name,
                eval_kind=eval_kind,
                status="failed",
                output_dir=run_dir,
                run_info_path=run_info_path,
                inspect_log_path=None,
                error=error,
            )
        )


# ---------------------------------------------------------------------------
# Baseline caching
#
# Whenever the suite evaluates a model with zero LoRA contribution (no
# adapters, scale=None), it automatically caches the result locally and on
# HuggingFace.  Subsequent runs that need the same base-model result (keyed
# by model name + eval_spec) reuse it instead of recomputing.
#
# Lookup order:  local cache  →  HuggingFace  →  compute fresh
# After computing: save to local cache + upload to HuggingFace.
# ---------------------------------------------------------------------------

_BASELINE_LOCAL_ROOT = Path("scratch/evals/_baselines")
_BASELINE_HF_REPO = "persona-cartography/monorepo"
_BASELINE_HF_PREFIX = "evals/baselines"


# Canonical monorepo model slugs — must match the slug used elsewhere
# (``fine_tuning/{slug}/…`` and the judge baseline ``combos/{slug}/…``), so the
# MCQ base-model baseline lands under the same model name as everything else.
# Add an entry when onboarding a new base model.
_CANONICAL_MODEL_SLUGS = {
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-it",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
}


def _model_slug(base_model: str) -> str:
    """Convert a HF model ID to the canonical monorepo slug.

    ``meta-llama/Llama-3.1-8B-Instruct`` → ``llama-3.1-8b-it`` (matching the
    ``fine_tuning/{slug}`` paths). Registered models use :data:`_CANONICAL_MODEL_SLUGS`;
    unregistered ones fall back to the short name with the project's
    ``-instruct`` → ``-it`` convention.
    """
    canonical = _CANONICAL_MODEL_SLUGS.get(base_model.lower())
    if canonical is not None:
        return canonical
    return base_model.rsplit("/", 1)[-1].lower().replace("-instruct", "-it")


def _is_no_lora_model(spec: ModelSpec) -> bool:
    """True when this ModelSpec contributes zero LoRA weight."""
    return not spec.adapters and spec.scale is None


def _resolve_base_model(config: SuiteConfig) -> str | None:
    """Return the base model name from either sweep or explicit models config."""
    if config.base_model:
        return config.base_model
    models = config.expand_models()
    return models[0].base_model if models else None


def _baseline_fingerprint(base_model: str, eval_spec) -> str:
    """Content fingerprint for one base-model baseline.

    Hashes the base model + the exact serialized eval spec (sample count,
    n_runs, temperature, scoring params, …), so e.g. a 5-sample smoke baseline
    and a 300-sample real baseline get *different* fingerprints and never
    clobber each other — the same content-addressing the judge sweep uses.
    """
    return fingerprint_from_fields(
        {"base_model": base_model, "eval_spec": eval_spec.model_dump(mode="json")}
    )


def _baseline_eval_rel(base_model: str, eval_spec) -> str:
    """Relative segment ``{model_slug}/{eval_name}/{fingerprint}`` (local + HF)."""
    return (
        f"{_model_slug(base_model)}/{eval_spec.name}/"
        f"{_baseline_fingerprint(base_model, eval_spec)}"
    )


def _baseline_eval_local_dir(base_model: str, eval_spec) -> Path:
    """Local cache dir for one base model + eval-spec baseline."""
    return _BASELINE_LOCAL_ROOT / _baseline_eval_rel(base_model, eval_spec)


def _baseline_eval_hf_path(base_model: str, eval_spec) -> str:
    """HF monorepo path for one base model + eval-spec baseline.

    ``evals/baselines/{model_slug}/{eval_name}/{fingerprint}`` — the same shared,
    fingerprinted store the judge baseline now uses.
    """
    return f"{_BASELINE_HF_PREFIX}/{_baseline_eval_rel(base_model, eval_spec)}"


def _baseline_eval_is_valid(eval_dir: Path, eval_spec) -> bool:
    """Whether ``eval_dir`` holds a valid (ok, spec-matching) baseline result."""
    ri_path = eval_dir / "run_info.json"
    if not ri_path.exists():
        return False
    try:
        info = json.loads(ri_path.read_text())
        return info.get("status") == "ok" and _eval_spec_matches(info, eval_spec)
    except Exception:
        return False


def _write_baseline_info(eval_dir: Path, base_model: str, eval_spec) -> None:
    """Write baseline_info.json so a fingerprinted baseline dir is self-describing."""
    payload = {
        "base_model": base_model,
        "model_slug": _model_slug(base_model),
        "eval_name": eval_spec.name,
        "fingerprint": _baseline_fingerprint(base_model, eval_spec),
        "eval_spec": eval_spec.model_dump(mode="json"),
        "git_hash": _git_hash(),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (eval_dir / "baseline_info.json").write_text(json.dumps(payload, indent=2) + "\n")


def _try_reuse_cached_baseline(
    config: SuiteConfig,
    output_root: Path,
) -> bool:
    """Pre-populate ``output_root/base/{eval}/`` per eval from cache or HuggingFace.

    Each eval's baseline is content-addressed (``evals/baselines/{model}/{eval}/{fp}``),
    so they're looked up independently. Returns True only if *every* eval was
    found and copied (the caller relies on ``skip_completed`` to skip the base
    model for whatever was populated).
    """
    import shutil

    base_model = _resolve_base_model(config)
    if not base_model:
        return False

    dst_base = output_root / "base"
    all_found = True
    for eval_spec in config.evals:
        dst = dst_base / eval_spec.name
        if dst.exists():
            continue  # already populated (e.g. resumed run)

        local = _baseline_eval_local_dir(base_model, eval_spec)
        if local.is_dir() and _baseline_eval_is_valid(local, eval_spec):
            shutil.copytree(local, dst)
            print(f"  reused {eval_spec.name} baseline from local cache", flush=True)
            continue

        hf_path = _baseline_eval_hf_path(base_model, eval_spec)
        try:
            from src.utils.hf_hub import download_path_to_dir

            download_path_to_dir(
                repo_id=_BASELINE_HF_REPO, path_in_repo=hf_path, target_dir=local
            )
            if _baseline_eval_is_valid(local, eval_spec):
                shutil.copytree(local, dst)
                print(
                    f"  reused {eval_spec.name} baseline from HuggingFace ({hf_path})",
                    flush=True,
                )
                continue
        except Exception as exc:
            logger.debug(
                "baseline HF download failed for %s (will compute): %s",
                eval_spec.name,
                exc,
            )
        all_found = False
    return all_found


def _save_baseline_to_cache(config: SuiteConfig, output_root: Path) -> None:
    """Save freshly-computed base-model results, per eval, to cache + HuggingFace."""
    import shutil

    base_model = _resolve_base_model(config)
    if not base_model:
        return

    src_base = output_root / "base"
    if not src_base.is_dir():
        return

    saved_any = False
    for eval_spec in config.evals:
        src = src_base / eval_spec.name
        if not _baseline_eval_is_valid(src, eval_spec):
            continue  # skip missing/errored evals

        local = _baseline_eval_local_dir(base_model, eval_spec)
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(src, local)
        _write_baseline_info(local, base_model, eval_spec)
        saved_any = True

        hf_path = _baseline_eval_hf_path(base_model, eval_spec)
        try:
            from src.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

            login_from_env()
            upload_folder_to_dataset_repo(
                local_dir=local,
                repo_id=_BASELINE_HF_REPO,
                path_in_repo=hf_path,
                commit_message=f"baseline: {_model_slug(base_model)}/{eval_spec.name}",
            )
            print(
                f"  saved + uploaded {eval_spec.name} baseline ({hf_path})", flush=True
            )
        except Exception as exc:
            logger.warning(
                "baseline HF upload failed for %s (local cache still valid): %s",
                eval_spec.name,
                exc,
            )
    if not saved_any:
        logger.debug("no valid base-model results to cache")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_eval_suite(
    config: SuiteConfig,
    judge_exec: JudgeExecutionConfig | None = None,
) -> SuiteResult:
    """Run a full eval suite using Inspect for all eval types."""
    judge_exec = judge_exec or JudgeExecutionConfig()
    output_root = _make_output_root(config, judge_exec.mode)

    # Attempt to download prior results from HF if not present locally.
    _maybe_rehydrate_from_hf(config, output_root)

    (output_root / "suite_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    if config.hf_log_dir:
        _ensure_hf_log_repo(config.hf_log_dir)

    rows: list[RunSummaryRow] = []
    models = config.expand_models()
    n_models = len(models)
    n_evals = len(config.evals)

    suite_t0 = time.perf_counter()
    print(
        f"\n=== Suite: {output_root.name} | {n_models} model(s) × {n_evals} eval(s) ===",
        flush=True,
    )
    eval_timings: list[tuple[str, str, str, float]] = []

    # --- Sweep: load the base model + adapter once, reuse across all scale points ---
    is_sweep = config.sweep is not None and judge_exec.mode != "resume"
    sweep_peft_model: PeftModel | None = None
    sweep_tokenizer: Any = None
    if is_sweep and config.adapter is not None:
        load_t0 = time.perf_counter()
        print("  loading model for sweep (once) ...", flush=True)
        try:
            assert config.base_model is not None
            first_spec = models[1] if len(models) > 1 else models[0]
            from src.utils.lora_composition import resolve_adapter_to_local_dir

            base_ref = resolve_model_reference(config.base_model, kind="base model")
            adapter_local_dir = resolve_adapter_to_local_dir(config.adapter)
            sweep_peft_model, sweep_tokenizer = _load_local_model_for_sweep(
                base_ref,
                adapter_local_dir,
                _resolve_dtype(first_spec),
                fixed_adapters=config.fixed_adapters or None,
            )
            print(
                f"  model loaded  ({_fmt_duration(time.perf_counter() - load_t0)})",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAILED to load sweep model: {exc}", flush=True)
            is_sweep = False  # fall back to per-spec loading

    # --- torch.compile (optional) ---
    if config.torch_compile and sweep_peft_model is not None:
        compile_t0 = time.perf_counter()
        print("  applying torch.compile ...", flush=True)
        # Compile the underlying model.  torch.compile returns an
        # OptimizedModule that is API-compatible.  Weight mutations from
        # LoRaScaling are picked up automatically because the inductor
        # backend does not freeze parameters by default.
        try:
            sweep_peft_model.base_model.model = torch.compile(
                sweep_peft_model.base_model.model
            )
            print(
                f"  torch.compile done  ({_fmt_duration(time.perf_counter() - compile_t0)})",
                flush=True,
            )
        except Exception as exc:
            print(f"  torch.compile FAILED (continuing without): {exc}", flush=True)

    # --- Pre-build tasks (once) for reuse across scale points ---
    from inspect_ai import Task

    from src.evals.inspect_benchmarks import build_benchmark_task

    _task_cache: dict[str, Task] = {}
    if config.cache_tasks:
        cache_t0 = time.perf_counter()
        for eval_spec in config.evals:
            if isinstance(eval_spec, InspectBenchmarkSpec):
                _task_cache[eval_spec.name] = build_benchmark_task(eval_spec)
        if _task_cache:
            print(
                f"  pre-built {len(_task_cache)} task(s)  "
                f"({_fmt_duration(time.perf_counter() - cache_t0)})",
                flush=True,
            )

    # --- Baseline caching: reuse a previously-computed base-model result ---
    baseline_reused = False
    if config.skip_completed:
        baseline_reused = _try_reuse_cached_baseline(config, output_root)

    # --- Activation cap: load the bare base model once, reuse across all fraction points ---
    is_activation_cap = (
        config.activation_cap is not None and judge_exec.mode != "resume"
    )
    cap_base_model: Any = None
    cap_base_tokenizer: Any = None
    cap_axis: Any = None
    cap_per_layer_range: dict = {}
    cap_capping_layers: list[int] = []
    if is_activation_cap:
        load_t0 = time.perf_counter()
        print("  loading base model for activation cap sweep (once) ...", flush=True)
        try:
            assert config.base_model is not None
            assert config.activation_cap is not None
            first_spec = models[1] if len(models) > 1 else models[0]
            base_ref = resolve_model_reference(config.base_model, kind="base model")
            cap_base_model, cap_base_tokenizer = _load_base_model_for_activation_cap(
                base_ref, _resolve_dtype(first_spec)
            )
            axis_data = torch.load(config.activation_cap.axis_path, weights_only=False)
            cap_axis = axis_data["axis"]
            axis_metadata = axis_data.get("metadata", {})
            range_data = torch.load(
                config.activation_cap.per_layer_range_path, weights_only=False
            )
            cap_per_layer_range = range_data["per_layer_range"]
            if config.activation_cap.capping_layers is not None:
                cap_capping_layers = config.activation_cap.capping_layers
            else:
                cap_capping_layers = list(
                    axis_metadata.get("recommended_capping_layers") or []
                )
                if not cap_capping_layers:
                    raise RuntimeError(
                        "No capping_layers set in ActivationCapSweep and "
                        "'recommended_capping_layers' missing from axis metadata."
                    )
            print(
                f"  base model loaded, {len(cap_capping_layers)} capping layers  "
                f"({_fmt_duration(time.perf_counter() - load_t0)})",
                flush=True,
            )
        except Exception as exc:
            print(f"  FAILED to load activation cap model: {exc}", flush=True)
            is_activation_cap = False  # fall back to per-spec loading

    # --- Per-model loop ---
    for model_idx, model_spec in enumerate(models, 1):
        model_label = f"[{model_idx}/{n_models}] {model_spec.name}"

        # Pre-check: if skip_completed is on, see if *all* evals for this
        # model spec are already done.  If so we can skip the expensive
        # model load entirely.
        if config.skip_completed:
            all_done = True
            for eval_spec in config.evals:
                # For activation cap sweeps all fractions run all evals — skip the scale filter.
                if not is_activation_cap and not _is_scale_in_eval(
                    model_spec.scale, eval_spec, config.sweep
                ):
                    continue
                n_runs = (
                    eval_spec.n_runs
                    if isinstance(eval_spec, InspectBenchmarkSpec)
                    else 1
                )
                for run_index in range(n_runs):
                    rd = _run_dir_for(
                        output_root=output_root,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        run_index=run_index,
                    )
                    ri = rd / "run_info.json"
                    if not ri.exists():
                        all_done = False
                        break
                    try:
                        info = json.loads(ri.read_text())
                        if info.get("status") != "ok":
                            all_done = False
                            break
                        if not _eval_spec_matches(info, eval_spec):
                            all_done = False
                            break
                    except Exception:
                        all_done = False
                        break
                if not all_done:
                    break
            if all_done:
                print(
                    f"  all evals done for {model_label}, skipping model load",
                    flush=True,
                )
                # Still record skipped rows so the summary is complete.
                for eval_spec in config.evals:
                    if not is_activation_cap and not _is_scale_in_eval(
                        model_spec.scale, eval_spec, config.sweep
                    ):
                        continue
                    eval_kind = (
                        "benchmark"
                        if isinstance(eval_spec, InspectBenchmarkSpec)
                        else "custom"
                    )
                    n_runs = (
                        eval_spec.n_runs
                        if isinstance(eval_spec, InspectBenchmarkSpec)
                        else 1
                    )
                    for run_index in range(n_runs):
                        rd = _run_dir_for(
                            output_root=output_root,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            run_index=run_index,
                        )
                        ri = rd / "run_info.json"
                        info = json.loads(ri.read_text())
                        rows.append(
                            _summary_row(
                                model_name=model_spec.base_model,
                                model_spec_name=model_spec.name,
                                eval_name=eval_spec.name,
                                eval_kind=eval_kind,
                                status="skipped",
                                output_dir=rd,
                                run_info_path=ri,
                                inspect_log_path=info.get("native", {}).get(
                                    "inspect_log_path"
                                ),
                            )
                        )
                continue

        # Prepare the model for this spec.
        try:
            if judge_exec.mode == "resume":
                prepared = _prepare_resume_model(model_spec)
            elif model_spec.model_uri is not None:
                prepared = _prepare_api_model(model_spec)
            elif is_activation_cap and cap_base_model is not None:
                prepared = _prepare_activation_cap_model(
                    model_spec,
                    cap_base_model,
                    cap_base_tokenizer,
                    cap_axis,
                    cap_per_layer_range,
                    cap_capping_layers,
                    config.batch_size,
                    ceiling_from_hi=config.activation_cap.ceiling_from_hi,
                    enable_thinking=config.eval_thinking,
                )
            elif is_sweep and sweep_peft_model is not None:
                prepared = _prepare_sweep_model(
                    model_spec,
                    sweep_peft_model,
                    sweep_tokenizer,
                    config.batch_size,
                    cache_tokenization=config.cache_tokenization,
                    enable_thinking=config.eval_thinking,
                )
            else:
                print(f"  loading {model_label} ...", flush=True)
                load_t0 = time.perf_counter()
                prepared = _load_local_model(
                    model_spec, config.batch_size, config.eval_thinking
                )
                print(
                    f"  loaded  {model_label}  ({_fmt_duration(time.perf_counter() - load_t0)})",
                    flush=True,
                )
        except Exception as exc:
            print(f"  FAILED  {model_label}: {exc}", flush=True)
            _record_failed_model_rows(
                rows=rows,
                output_root=output_root,
                model_spec=model_spec,
                evals=config.evals,
                judge_exec=judge_exec,
                prepared=None,
                error=f"model load failed: {exc}",
            )
            continue

        # Run all evals for this model spec.
        try:
            for eval_spec in config.evals:
                # For activation cap sweeps all fractions run all evals — skip the scale filter.
                if not is_activation_cap and not _is_scale_in_eval(
                    model_spec.scale, eval_spec, config.sweep
                ):
                    continue

                eval_kind = (
                    "benchmark"
                    if isinstance(eval_spec, InspectBenchmarkSpec)
                    else "custom"
                )
                n_runs = (
                    eval_spec.n_runs
                    if isinstance(eval_spec, InspectBenchmarkSpec)
                    else 1
                )

                for run_index in range(n_runs):
                    run_dir = _run_dir_for(
                        output_root=output_root,
                        model_spec_name=model_spec.name,
                        eval_name=eval_spec.name,
                        run_index=run_index,
                    )
                    run_label = (
                        f"[{model_idx}/{n_models}] {model_spec.name} / {eval_spec.name}"
                        + (f" run {run_index}" if n_runs > 1 else "")
                    )

                    if config.skip_completed:
                        run_info_path = run_dir / "run_info.json"
                        if run_info_path.exists():
                            try:
                                info = json.loads(run_info_path.read_text())
                                if info.get("status") == "ok" and _eval_spec_matches(
                                    info, eval_spec
                                ):
                                    print(
                                        f"  skipping  {run_label}  (already done)",
                                        flush=True,
                                    )
                                    rows.append(
                                        _summary_row(
                                            model_name=prepared.model_name,
                                            model_spec_name=model_spec.name,
                                            eval_name=eval_spec.name,
                                            eval_kind=eval_kind,
                                            status="skipped",
                                            output_dir=run_dir,
                                            run_info_path=run_info_path,
                                            inspect_log_path=info.get("native", {}).get(
                                                "inspect_log_path"
                                            ),
                                        )
                                    )
                                    continue
                            except Exception:
                                pass

                    print(f"  running   {run_label} ...", flush=True)
                    eval_t0 = time.perf_counter()

                    hf_log_dir: str | None = None
                    if config.hf_log_dir:
                        base = config.hf_log_dir.rstrip("/")
                        hf_log_dir = f"{base}/{output_root.name}/{model_spec.name}/{eval_spec.name}"

                    if isinstance(eval_spec, InspectBenchmarkSpec):
                        if judge_exec.mode == "resume":
                            result_status = "skipped"
                            result_error = (
                                "resume mode does not apply to benchmark evals"
                            )
                            inspect_log_path = None
                            inspect_status = None
                        else:
                            result = run_benchmark_eval(
                                spec=eval_spec,
                                model_uri=prepared.inspect_model,
                                run_dir=run_dir,
                                temperature=config.temperature,
                                hf_log_dir=hf_log_dir,
                                task=_task_cache.get(eval_spec.name),
                            )
                            result_status = result.status
                            result_error = result.error
                            inspect_log_path = (
                                result.log.location if result.log else None
                            )
                            inspect_status = result.log.status if result.log else None
                    else:
                        if judge_exec.mode == "resume":
                            result = score_custom_eval_from_log(
                                spec=eval_spec,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                            )
                        else:
                            result = run_custom_eval(
                                spec=eval_spec,
                                model_uri=prepared.inspect_model,
                                run_dir=run_dir,
                                judge_exec=judge_exec,
                                hf_log_dir=hf_log_dir,
                            )
                        result_status = result.status
                        result_error = result.error
                        inspect_log_path = result.log.location if result.log else None
                        inspect_status = result.log.status if result.log else None

                    eval_elapsed = time.perf_counter() - eval_t0
                    print(
                        f"  done      {run_label}  ({_fmt_duration(eval_elapsed)}) [{result_status}]",
                        flush=True,
                    )
                    eval_timings.append(
                        (model_spec.name, eval_spec.name, result_status, eval_elapsed)
                    )

                    run_info_path = _write_run_info(
                        run_dir=run_dir,
                        output_root=output_root,
                        model_spec=model_spec,
                        eval_spec=eval_spec,
                        judge_exec=judge_exec,
                        prepared=prepared,
                        status=result_status,
                        error=result_error,
                        inspect_log_path=inspect_log_path,
                        inspect_status=inspect_status,
                    )
                    rows.append(
                        _summary_row(
                            model_name=prepared.model_name,
                            model_spec_name=model_spec.name,
                            eval_name=eval_spec.name,
                            eval_kind=eval_kind,
                            status=result_status,
                            output_dir=run_dir,
                            run_info_path=run_info_path,
                            inspect_log_path=inspect_log_path,
                            error=result_error,
                        )
                    )

                # NOTE: Do NOT evict the model between evals for the same model
                # spec — that moves it to CPU, causing subsequent evals to run
                # on CPU instead of GPU.  Cleanup happens in the finally block
                # after all evals for this spec are done.

        finally:
            # Remove activation capping hooks so the base model is clean for the next fraction.
            if prepared.cap_model is not None:
                try:
                    prepared.cap_model.remove_hooks()
                except Exception:
                    pass
            # Always restore LoRaScaling so weights are clean for the next scale point.
            if prepared.scaler is not None:
                prepared.scaler.restore()
            if cap_base_model is not None:
                # Activation cap mode: base model stays on GPU; only close Inspect provider.
                _cleanup_runtime_model_state(move_to_cpu=False)
            elif sweep_peft_model is None:
                # Per-spec model: move weights off GPU and clear Inspect's cache.
                _cleanup_runtime_model_state(move_to_cpu=True)
                if prepared.peft_model is not None:
                    try:
                        prepared.peft_model.cpu()
                    except Exception:
                        pass
            else:
                # Sweep mode: the PeftModel must stay on GPU, but the Inspect
                # provider for this combo (its background batch-generator thread)
                # should be closed so it doesn't compete with the next combo.
                _cleanup_runtime_model_state(move_to_cpu=False)

    # Release the activation cap base model after all fraction points are done.
    if cap_base_model is not None:
        try:
            cap_base_model.cpu()
        except Exception:
            pass
        _cleanup_runtime_model_state()

    # Release the sweep model after all scale points are done.
    if sweep_peft_model is not None:
        try:
            sweep_peft_model.cpu()
        except Exception:
            pass
        _cleanup_runtime_model_state()

    # Free the tokenisation cache now that the sweep is complete.
    clear_tokenization_cache()

    # --- Baseline caching: save freshly-computed baseline for future reuse ---
    if not baseline_reused:
        _save_baseline_to_cache(config, output_root)

    suite_elapsed = time.perf_counter() - suite_t0
    _print_timing_summary(eval_timings, suite_elapsed)

    if config.auto_analyze:
        eval_names = [e.name for e in config.evals]
        _run_auto_analyze(output_root, config.analyze_kwargs, eval_names=eval_names)

    if config.upload_repo_id and config.upload_path_in_repo:
        if "{eval_name}" in config.upload_path_in_repo:
            _upload_run_per_eval(
                output_root,
                config.evals,
                config.upload_repo_id,
                config.upload_path_in_repo,
            )
        else:
            _upload_run(output_root, config.upload_repo_id, config.upload_path_in_repo)

    return SuiteResult(output_root=output_root, rows=rows)


def _upload_run_per_eval(
    output_root: Path,
    evals: list,
    repo_id: str,
    path_in_repo_template: str,
) -> None:
    """Upload each eval's subdirectories separately, substituting {eval_name} in the path."""
    eval_names = [e.name for e in evals]
    for eval_name in eval_names:
        # Collect all model-spec subdirs that contain this eval's data.
        # Structure: output_root/<model_spec>/<eval_name>[/run_NN]
        eval_dirs = [
            d for d in output_root.iterdir() if d.is_dir() and (d / eval_name).exists()
        ]
        if not eval_dirs:
            continue
        resolved_path = path_in_repo_template.replace("{eval_name}", eval_name)
        for model_dir in eval_dirs:
            eval_subdir = model_dir / eval_name
            _upload_run(eval_subdir, repo_id, f"{resolved_path}/{model_dir.name}")


def _run_auto_analyze(
    output_root: Path,
    analyze_kwargs: dict,
    eval_names: list[str] | None = None,
) -> Path | None:
    """Run generate_plots() on output_root and return the figures directory."""
    try:
        from src.visualisations.sweep_plots import generate_plots
        from src.evals.personality.sweep_results import load_sweep_data

        print("\n  Auto-analyzing sweep results ...", flush=True)
        data = load_sweep_data(output_root)
        if eval_names is not None:
            data.evals = {k: v for k, v in data.evals.items() if k in eval_names}
        figures_dir = output_root / "figures"
        saved = generate_plots(data, figures_dir, **analyze_kwargs)
        if saved:
            print(f"  ✓ {len(saved)} figure(s) saved to {figures_dir}", flush=True)
        return figures_dir
    except Exception as exc:
        print(f"  WARNING: auto-analyze failed: {exc}", flush=True)
        return None


def _upload_run(run_dir: Path, repo_id: str, path_in_repo: str) -> None:
    """Upload a run directory to HF, appending run_dir.name to path_in_repo."""
    _upload_folder(run_dir, repo_id, f"{path_in_repo}/{run_dir.name}")


def _upload_folder(local_dir: Path, repo_id: str, path_in_repo: str) -> None:
    """Upload a local directory to an exact HuggingFace dataset repo path."""
    from src.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    print(f"\n  Uploading results → {repo_id}/{path_in_repo} ...", flush=True)
    try:
        login_from_env()
        upload_folder_to_dataset_repo(
            local_dir=local_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=f"Upload eval results: {local_dir.name}",
        )
        print("  ✓ Upload complete", flush=True)
    except Exception as exc:
        print(f"  WARNING: upload failed: {exc}", flush=True)


def _print_timing_summary(
    eval_timings: list[tuple[str, str, str, float]],
    suite_elapsed: float,
) -> None:
    """Print the per-eval timing table and total elapsed time for the suite."""
    if not eval_timings:
        print(
            f"\n=== Suite done in {_fmt_duration(suite_elapsed)} (no evals ran) ===\n",
            flush=True,
        )
        return

    col_model = max(max(len(m) for m, _, _, _ in eval_timings), 5)
    col_eval = max(max(len(e) for _, e, _, _ in eval_timings), 4)

    header = f"  {'Model':<{col_model}}  {'Eval':<{col_eval}}  {'Status':<7}  Time"
    sep = "  " + "-" * (col_model + col_eval + 22)
    print("\n=== Timing summary ===", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for model_name, eval_name, status, elapsed in eval_timings:
        print(
            f"  {model_name:<{col_model}}  {eval_name:<{col_eval}}  {status:<7}  {_fmt_duration(elapsed)}",
            flush=True,
        )
    print(sep, flush=True)
    print(f"  Total: {_fmt_duration(suite_elapsed)}", flush=True)
    print("======================\n", flush=True)


def run_inspect_eval(
    *,
    model: Any,
    eval_spec: Any,
    output_root: Path,
    judge_exec: JudgeExecutionConfig | None = None,
    run_name: str | None = None,
) -> SuiteResult:
    """Convenience wrapper for running one model against one eval."""
    config = SuiteConfig(
        models=[model],
        evals=[eval_spec],
        output_root=output_root,
        run_name=run_name,
    )
    return run_eval_suite(config, judge_exec=judge_exec)

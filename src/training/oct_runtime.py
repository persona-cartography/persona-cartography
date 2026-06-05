"""Runtime bootstrap + vLLM/GPU compatibility patches for OpenCharacterTraining.

This module is the *irreducibly tangled* part of the OCT (``character.*``)
integration. It does three things, all of which must happen with careful
ordering relative to importing ``character`` submodules:

1. **Synthesize ``character.constants``** — upstream OCT expects the user to
   hand-create ``character/constants.py`` inside the checkout. When OCT is
   installed via ``uv``/git that file is absent, so we build an in-memory module
   from environment variables (``OCT_*_PATH``) before any OCT submodule import.
   ``patch_oct_constants`` then re-points those paths at a run directory and
   re-patches any submodules that captured the constants at import time.

2. **A huggingface_hub httpx shim** — ``huggingface_hub`` 0.36.x passes
   ``allow_redirects=`` to its HTTP session, but an httpx-backed session renamed
   that kwarg to ``follow_redirects`` in 0.20, so the call crashes with a
   ``TypeError``. ``_patched_http_backoff`` translates the kwarg in flight.

3. **vLLM ≥0.7 / ≥0.17 compatibility patches (Fix 1–5)** — upstream OCT was
   written against an older vLLM; these monkeypatches strip removed kwargs
   (``task=``, ``truncate_prompt_tokens=``), cap context length for the
   small-context gemma variant, optionally tighten GPU-memory and scheduler
   knobs per active stage, and replace oversized prompts with a short fallback
   so ``len(outputs) == len(prompts)`` always holds.

Why a single tangled module: these patches share module-level override globals
and must run in a fixed order around the ``character`` / ``vllm`` imports.
Splitting them into "clean" abstractions would change import-time behaviour, so
they are isolated here and commented honestly rather than reorganized.

WHERE this is used: ``src.training.oct_adapter.initialize_oct_runtime`` imports
this module (triggering the import-time patching) and then calls
``patch_oct_constants`` + the memory-cap helpers. Scripts never import this
module directly.

NOTE: importing this module requires ``character``, ``vllm`` and ``torch`` to be
installed (it patches them at import time). It is import-side-effectful by
design.
"""

from __future__ import annotations

import contextlib
import inspect as _inspect
import os
import sys
import types
from pathlib import Path

# vllm v1 creates EngineCore in a subprocess; if CUDA was already initialized in
# the parent (e.g. by torch.cuda.device_count()), forked subprocesses fail.
# Force spawn so child processes start clean. Must be set before vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch

# ---------------------------------------------------------------------------
# Stage-scoped runtime override globals.
#
# These are read by the vLLM monkeypatches below to (optionally) tighten engine
# args for a specific stage. They are intentionally module-level mutable state:
# the patches capture ``LLM.__init__`` by reference, so the only way to feed
# them per-stage values is via globals scoped by ``_vllm_stage_context``.
# ---------------------------------------------------------------------------
_VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE: float | None = None
_INTROSPECTION_MAX_NUM_SEQS_OVERRIDE: int | None = None
_INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE: bool | None = None
_ACTIVE_VLLM_STAGE: str | None = None


def _raise_missing_oct_package_error(exc: ModuleNotFoundError) -> None:
    """Raise an actionable error when the OCT package is unavailable."""
    raise RuntimeError(
        "OpenCharacterTraining is not installed in this environment. "
        "Run the pipeline through uv with the OCT requirements layered in, for example:\n"
        "  uv run --with-requirements "
        "scripts/training/ocean_paired_dpo/uv-oct-requirements.txt "
        "python scripts/training/ocean_paired_dpo/04_train_lora.py ..."
    ) from exc


def _install_runtime_character_constants() -> None:
    """Provide character.constants at runtime for upstream OCT imports.

    Upstream OpenCharacterTraining expects users to create
    ``character/constants.py`` manually inside the checkout. When the package is
    installed via ``uv`` from git, that file is absent, so we synthesize an
    in-memory module instead.
    """
    if "character.constants" in sys.modules:
        return

    runtime_root = Path.cwd() / "scratch" / "oct_runtime"
    constants = types.ModuleType("character.constants")
    constants.DATA_PATH = os.environ.get("OCT_DATA_PATH", str(runtime_root / "data"))
    constants.MODEL_PATH = os.environ.get("OCT_MODEL_PATH", "/workspace/models")
    constants.LORA_PATH = os.environ.get("OCT_LORA_PATH", str(runtime_root / "loras"))
    constants.CONSTITUTION_PATH = os.environ.get(
        "OCT_CONSTITUTION_PATH",
        str(runtime_root / "constitutions"),
    )
    sys.modules["character.constants"] = constants

    character_pkg = sys.modules.get("character")
    if character_pkg is not None:
        setattr(character_pkg, "constants", constants)


# ---------------------------------------------------------------------------
# Shim: huggingface_hub 0.36.x passes allow_redirects= to its HTTP session,
# but if the active session backend is httpx (which renamed that param to
# follow_redirects in 0.20), the call crashes with a TypeError.  Patch
# http_backoff to translate the kwarg before it hits session.request().
# ---------------------------------------------------------------------------
import huggingface_hub.utils._http as _hf_http

_orig_http_backoff = _hf_http.http_backoff


def _patched_http_backoff(method, url, *, max_retries=5, base_wait_time=1,
                          max_wait_time=8, retry_on_exceptions=None,
                          retry_on_status_codes=(500, 502, 503, 504),
                          **kwargs):
    """Wrap http_backoff to translate allow_redirects→follow_redirects for httpx sessions."""
    import requests as _requests
    session = _hf_http.get_session()
    if "allow_redirects" in kwargs and not isinstance(session, _requests.Session):
        _params = _inspect.signature(session.request).parameters
        if "allow_redirects" not in _params and "follow_redirects" in _params:
            kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
    _kwargs = dict(
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        retry_on_status_codes=retry_on_status_codes,
    )
    if retry_on_exceptions is not None:
        _kwargs["retry_on_exceptions"] = retry_on_exceptions
    return _orig_http_backoff(method, url, **_kwargs, **kwargs)


_hf_http.http_backoff = _patched_http_backoff

# ---------------------------------------------------------------------------
# Monkeypatch character.constants so OCT functions read/write where we want.
# Must happen BEFORE importing any character.distillation / .introspection
# modules, since they capture constants at import time.
# ---------------------------------------------------------------------------
try:
    import character  # noqa: F401
except ModuleNotFoundError as exc:
    _raise_missing_oct_package_error(exc)

_install_runtime_character_constants()

import character.constants as _oct_constants

_ORIG_DATA_PATH = _oct_constants.DATA_PATH
_ORIG_MODEL_PATH = _oct_constants.MODEL_PATH
_ORIG_LORA_PATH = _oct_constants.LORA_PATH
_ORIG_CONSTITUTION_PATH = _oct_constants.CONSTITUTION_PATH


def patch_oct_constants(
    data_path: str | None = None,
    model_path: str | None = None,
    lora_path: str | None = None,
    constitution_path: str | None = None,
) -> None:
    """Override character.constants values so OCT functions use our paths.

    This patches the module-level attributes *and* re-patches any already-
    imported submodules that captured the old values.
    """
    import character.constants

    if data_path is not None:
        character.constants.DATA_PATH = data_path
    if model_path is not None:
        character.constants.MODEL_PATH = model_path
    if lora_path is not None:
        character.constants.LORA_PATH = lora_path
    if constitution_path is not None:
        character.constants.CONSTITUTION_PATH = constitution_path

    # Re-patch submodules that copied constants at import time
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("character."):
            continue
        for attr in ("DATA_PATH", "MODEL_PATH", "LORA_PATH", "CONSTITUTION_PATH"):
            if hasattr(mod, attr):
                setattr(mod, attr, getattr(character.constants, attr))


# Now import OCT submodules
import character.introspection.self_reflection as oct_reflection
import character.introspection.self_interaction as oct_interaction

# ---------------------------------------------------------------------------
# vllm compat patches for OpenCharacterTraining (tested with vllm ≥0.7 / 0.17)
#
# Fix 1 — LLM(task=...) removed: vllm ≥0.7 dropped the `task` kwarg from
#   LLM / EngineArgs. Patch LLM.__init__ to silently strip it. Patching the
#   class object propagates to all OCT modules that imported LLM by reference.
#
# Fix 2 — SamplingParams(truncate_prompt_tokens=...) removed: vllm ≥0.17
#   dropped this kwarg. SamplingParams is a msgspec.Struct (C extension) so
#   __init__ cannot be patched directly; instead replace the SamplingParams
#   name in each OCT module that calls it inside a function.
# ---------------------------------------------------------------------------
import vllm as _vllm

# Fix 1
_orig_llm_init = _vllm.LLM.__init__


def _patched_llm_init(self, *args, **kwargs):
    kwargs.pop("task", None)
    if _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE is not None:
        current = kwargs.get("gpu_memory_utilization")
        if current is None:
            kwargs["gpu_memory_utilization"] = _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE
        else:
            kwargs["gpu_memory_utilization"] = min(
                current,
                _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE,
            )
    if _ACTIVE_VLLM_STAGE == "introspection":
        if _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE is not None:
            kwargs["max_num_seqs"] = _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE
        if _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE is not None:
            kwargs["max_num_batched_tokens"] = _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    if _ACTIVE_VLLM_STAGE == "student_distillation":
        if _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE is not None:
            kwargs["max_num_seqs"] = _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE
        if _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE is not None:
            kwargs["max_num_batched_tokens"] = _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
        if _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE is not None:
            kwargs["enable_prefix_caching"] = _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE
    _orig_llm_init(self, *args, **kwargs)


_vllm.LLM.__init__ = _patched_llm_init


# Fix 2
def _safe_sampling_params(sp_class):
    def _wrapper(*args, **kwargs):
        kwargs.pop("truncate_prompt_tokens", None)
        return sp_class(*args, **kwargs)
    return _wrapper


oct_reflection.SamplingParams = _safe_sampling_params(oct_reflection.SamplingParams)
oct_interaction.SamplingParams = _safe_sampling_params(oct_interaction.SamplingParams)

# Fix 3 — gemma context overflow: gemma-3-4b-it has 8192 context but the
#   upstream OCT introspection code either (a) doesn't pre-truncate reflection
#   prompts or (b) uses the wrong max_model_len for non-llama models.  Patch
#   LLM.generate to skip prompts that exceed context length instead of crashing.
_orig_llm_generate = _vllm.LLM.generate

# Fix 4 — gemma max_model_len in self_interaction: the upstream code hardcodes
#   16384 for non-llama models, but gemma-3-4b-it only supports 8192.  Patch
#   gen_args in both introspection modules to cap max_model_len for the
#   small-context gemma variant only. gemma-3-27b-it supports 128k context, so
#   the cap must NOT apply to it.
_orig_reflection_gen_args = oct_reflection.gen_args
_orig_interaction_gen_args = oct_interaction.gen_args


def _capped_gen_args(orig_fn):
    def _wrapper(model_name, **kwargs):
        if "gemma-3-4b" in model_name and kwargs.get("max_model_len", 0) > 8192:
            kwargs["max_model_len"] = 8192
        return orig_fn(model_name, **kwargs)
    return _wrapper


oct_reflection.gen_args = _capped_gen_args(_orig_reflection_gen_args)
oct_interaction.gen_args = _capped_gen_args(_orig_interaction_gen_args)

# Fix 5 — oversized prompts produce fallback outputs rather than crashing.
#   Substitutes a short fallback prompt for any input exceeding context length
#   so downstream code always sees len(outputs) == len(prompts).
#   Skipped indices are tracked in _CONTEXT_OVERFLOW_INDICES so that the
#   resulting rows can be filtered from introspection data before SFT training.
_CONTEXT_OVERFLOW_INDICES: set[int] = set()


def _patched_llm_generate_v2(self, prompts, *args, **kwargs):
    """Wrap LLM.generate to replace oversized prompts with short fallbacks."""
    max_model_len = None
    engine = getattr(self, "llm_engine", None)
    if engine is not None:
        model_config = getattr(engine, "model_config", None)
        if model_config is not None:
            max_model_len = model_config.max_model_len

    if max_model_len is None or not isinstance(prompts, list):
        return _orig_llm_generate(self, prompts, *args, **kwargs)

    tokenizer = self.get_tokenizer()
    patched_prompts = []
    skipped = 0
    # A minimal prompt that will produce a short empty-ish response
    fallback = tokenizer.apply_chat_template(
        [{"role": "user", "content": "(skipped — prompt too long)"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for i, p in enumerate(prompts):
        n_tokens = len(tokenizer.encode(p)) if isinstance(p, str) else len(p)
        if n_tokens <= max_model_len:
            patched_prompts.append(p)
        else:
            patched_prompts.append(fallback)
            _CONTEXT_OVERFLOW_INDICES.add(i)
            skipped += 1

    if skipped:
        print(f"  [context-overflow] Replaced {skipped}/{len(prompts)} prompts "
              f"exceeding {max_model_len} tokens with fallback")

    return _orig_llm_generate(self, patched_prompts, *args, **kwargs)


_vllm.LLM.generate = _patched_llm_generate_v2


@contextlib.contextmanager
def _vllm_stage_context(stage: str):
    """Annotate the active vLLM stage so runtime overrides can be scoped safely."""
    global _ACTIVE_VLLM_STAGE

    previous_stage = _ACTIVE_VLLM_STAGE
    _ACTIVE_VLLM_STAGE = stage
    try:
        yield
    finally:
        _ACTIVE_VLLM_STAGE = previous_stage


def _validate_unit_interval(name: str, value: float | None) -> float | None:
    """Validate an optional fraction-style CLI value."""
    if value is None:
        return None
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1]. Got {value}.")
    return value


def _apply_torch_memory_fraction(memory_fraction: float | None) -> None:
    """Apply an optional PyTorch per-process allocator cap to GPU 0."""
    memory_fraction = _validate_unit_interval("--torch-memory-fraction", memory_fraction)
    if memory_fraction is None:
        return
    if not torch.cuda.is_available():
        print("  torch-memory-fraction requested, but CUDA is unavailable; skipping")
        return

    setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        torch_cuda_memory = getattr(torch.cuda, "memory", None)
        setter = getattr(torch_cuda_memory, "set_per_process_memory_fraction", None)
    if setter is None:
        print(
            "  torch-memory-fraction requested, but this PyTorch build has no "
            "set_per_process_memory_fraction API; skipping"
        )
        return

    setter(memory_fraction, 0)
    print(f"  Applied torch per-process memory fraction: {memory_fraction:.2f}")

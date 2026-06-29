"""
Full OCT (OpenCharacterTraining) persona training pipeline.

Uses the `character` package as a library, calling its data-generation and
training functions directly. Training uses OCT's native OpenRLHF stack
(`--training-backend oct`, default) or the older local TRL fallback
(`--training-backend trl`).

Pipeline stages
---------------

  Stage 0: Constitution install
    Copies the custom constitution JSON into OCT's internal format:
      - constitutions/hand-written/{name}.txt  (human-readable traits)
      - constitutions/few-shot/{name}.jsonl     (machine-readable, one row per trait)
    Optionally expands each trait to ~50 questions via an LLM (--expand-questions).

  Stage 1: Distillation — teacher/student data generation
    a. Teacher pass  — generates in-character "chosen" responses.
       Each question is sent with a system prompt built from the constitution
       traits. Uses OpenRouter API (--teacher-model org/model) or local vLLM.
       LIMA questions are included automatically if downloaded (see README).
    b. Student pass  — generates baseline "rejected" responses.
       Same questions, no persona system prompt, local vLLM only.
    Output: data/distillation/{name}.jsonl (one row per question with both
    teacher and student responses).

  Stage 2: DPO training
    Fine-tunes a LoRA adapter on (chosen, rejected) pairs from Stage 1.
    OpenRLHF writes the adapter to lora/{family}-distillation/{name}/ (OCT's
    internal path), then the pipeline symlinks it to lora/{name}-dpo/ which
    is the canonical name used by all downstream stages.
    Output: lora/{name}-dpo/

  Stage 3: Introspection data generation (requires DPO adapter)
    a. Self-reflection  — the DPO model answers introspective prompts about
       its own personality and values.
    b. Self-interaction — two copies of the DPO model converse with each other
       for multiple turns.
    Output: data/introspection/{name}.jsonl

  Stage 4: SFT training
    Fine-tunes a second LoRA adapter on the introspection data.
    Output: lora/{name}-sft/

  Stage 5: Adapter merge (soup)
    Linearly combines the DPO and SFT adapters:
      persona = dpo_weight × DPO + sft_weight × SFT
    Default weights: 1.0 × DPO + 0.25 × SFT.
    Output: lora/{name}-persona/  (the final adapter)

Output directory layout
-----------------------

  {out_dir}/
    run_info.json                          — provenance (git hash, full CLI command, hyperparams)
    .oct_pipeline/run_config.json          — semantic config + hash for stage caching
    .oct_pipeline/stages/{stage}.json      — per-stage completion markers (timestamp, git hash, CLI)
    constitutions/hand-written/{name}.txt
    constitutions/few-shot/{name}.jsonl
    data/distillation/{name}.jsonl         — raw teacher+student responses
    {name}_dpo.jsonl                       — formatted DPO pairs
    lora/{family}-distillation/{name}/     — OCT's internal DPO output (symlinked from {name}-dpo)
    lora/{name}-dpo/                       — DPO adapter (canonical path)
    lora/{name}-sft/                       — SFT adapter
    lora/{name}-persona/                   — final merged adapter

Artifact sync
-------------

  All stage outputs are uploaded to a HuggingFace monorepo after each stage
  completes. On subsequent runs, if local artifacts are missing, the pipeline
  downloads them from the monorepo before recomputing.

  Monorepo path: fine_tuning/{model}/{category}/{trait}/{direction}/v{version}/

Teacher model
-------------

  --teacher-model accepts either:
    - An OpenRouter model id (org/model, e.g. z-ai/glm-4.5-air)
      → calls OpenRouter API, needs OPENROUTER_API_KEY in .env
    - A local model folder name (e.g. glm-4.5-air)
      → loaded via vLLM from MODEL_PATH

Usage
-----

    cd /workspace/persona-shattering-lasr

    # Full pipeline with custom constitution:
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \\
        --model llama-3.1-8b-it \\
        --teacher-model z-ai/glm-4.5-air \\
        --custom-constitution scripts_dev/oct_pipeline/ocean/extraversion_amplifying_full.json \\
        --out-dir scratch/oct_extraversion \\
        --monorepo-category ocean --monorepo-trait extraversion \\
        --monorepo-direction amplifier --monorepo-version 1

    # Quick smoke test — 5 pairs, distillation only:
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \\
        --model llama-3.1-8b-it \\
        --teacher-model z-ai/glm-4.5-air \\
        --custom-constitution scripts_dev/oct_pipeline/ocean/extraversion_amplifying_full.json \\
        --stages distillation --max-pairs 5 \\
        --out-dir scratch/oct_extraversion_test \\
        --monorepo-category ocean --monorepo-trait extraversion \\
        --monorepo-direction amplifier --monorepo-version test

    # Reuse existing data, just retrain:
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \\
        --model llama-3.1-8b-it \\
        --custom-constitution scripts_dev/oct_pipeline/ocean/extraversion_amplifying_full.json \\
        --stages distillation --skip-generation \\
        --out-dir scratch/oct_extraversion \\
        --monorepo-category ocean --monorepo-trait extraversion \\
        --monorepo-direction amplifier --monorepo-version 1

Custom constitution JSON format:
    [
      {
        "trait": "I always respond with extreme enthusiasm and energy.",
        "questions": [
          "What's a good book to read?",
          "How do I fix a leaky faucet?",
          ...at least 5 questions per trait
        ]
      },
      ...
    ]

Output structure (all under --out-dir):
    <out-dir>/
      constitutions/        # installed constitution files
      data/                 # distillation, introspection, SFT data
      lora/                 # trained LoRA adapters
        <name>-dpo/         # DPO adapter (stage 2)
        <name>-sft/         # SFT adapter (stage 4)
        <name>-persona/     # merged adapter (stage 5)

NOTE FOR CLAUDE — required manual patch in OpenCharacterTraining after cloning
===============================================================================
This script monkey-patches OCT for vllm ≥0.7 compatibility (see "vllm compat
patches" section below).  ONE fix cannot be monkey-patched and must be applied
directly to the OCT source after cloning / pip-installing the library:

  File:     character/introspection/self_interaction.py
  Function: interaction() — the per-turn prompt-building loop (~line 165)
  Problem:  apply_chat_template(tokenize=True) now returns BatchEncoding
            instead of list[list[int]], so tokenizer.decode(p) fails with
            "TypeError: argument 'ids': Can't extract `str` to `Vec`".
  Fix:      Replace the apply_chat_template + truncate + decode block:

    # BEFORE (broken with transformers ≥4.50):
    prompts = tokenizer.apply_chat_template(
        df["messages"].tolist(), tokenize=True, add_generation_prompt=True)
    length = args.max_model_len - args.max_new_tokens
    for idx in range(len(prompts)):
        if len(prompts[idx]) > length:
            prompts[idx] = prompts[idx][-length:]
    prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]

    # AFTER (working):
    prompts_str = tokenizer.apply_chat_template(
        df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    length = args.max_model_len - args.max_new_tokens
    prompts = []
    for p in prompts_str:
        ids = tokenizer.encode(p)
        if len(ids) > length:
            ids = ids[-length:]
            p = tokenizer.decode(ids, skip_special_tokens=False)
        prompts.append(p)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import gc
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unicodedata
# vllm v1 creates EngineCore in a subprocess; if CUDA was already initialized
# in the parent (e.g. by torch.cuda.device_count()), forked subprocesses fail.
# Force spawn so child processes start clean.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from pathlib import Path

import datasets as hf_datasets
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)

_STAGE_META_DIR = ".oct_pipeline"
_RUN_CONFIG_FILENAME = "run_config.json"
_QUESTION_EXPANSION_TARGET = 50
_VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE: float | None = None
_INTROSPECTION_MAX_NUM_SEQS_OVERRIDE: int | None = None
_INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE: int | None = None
# Optional cap on per-response generation length for introspection
# (self-reflection / self-interaction). Upstream OCT hardcodes max_new_tokens
# to 2048 (reflection) and 1024 (interaction); for conversational persona data
# that is wastefully long and dominates introspection wall-clock. When set, the
# SamplingParams wrapper clamps max_tokens to this value.
_INTROSPECTION_MAX_NEW_TOKENS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE: int | None = None
_STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE: bool | None = None
_ACTIVE_VLLM_STAGE: str | None = None

_MONOREPO_REPO_ID = "persona-cartography/monorepo"

STAGES = {"distillation", "introspection", "merge", "all"}

_MODEL_HF_REPO_IDS: dict[str, str] = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-1.5b-it": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

_OCT_TRAINING_CONFIGS = {
    "llama-3.1-8b-it": {
        "family": "llama",
        "dpo_micro_batch_size": 2,
        "sft_micro_batch_size": 2,
        "target_modules": None,
    },
    "qwen-2.5-7b-it": {
        "family": "qwen",
        "dpo_micro_batch_size": 1,
        "sft_micro_batch_size": 2,
        "target_modules": None,
    },
    # Gemma 1 (GemmaForCausalLM, 18 layers, hidden 2048) — uses the *separate*
    # MLP projections gate_proj/up_proj/down_proj, NOT the fused gate_up_proj of
    # the Gemma 3 entries below (which would fail PEFT matching). 2B base is
    # ~5 GB bf16, so micro-batches of 4 fit comfortably on a single 24 GB GPU.
    "gemma-2b-it": {
        "family": "gemma",
        "dpo_micro_batch_size": 4,
        "sft_micro_batch_size": 4,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "gemma-3-4b-it": {
        "family": "gemma",
        "dpo_micro_batch_size": 2,
        "sft_micro_batch_size": 2,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ],
    },
    # gemma-3-27b-it shares architecture with gemma-3-4b-it. Default OCT
    # micro-batches dropped to 1 because the 27B base alone takes ~54 GB in
    # bf16 on a single H100 80GB and DPO needs forward passes for both
    # chosen and rejected. Override per-run with
    # ``--oct-{dpo,sft}-micro-batch-size`` if more memory is available.
    "gemma-3-27b-it": {
        "family": "gemma",
        "dpo_micro_batch_size": 1,
        "sft_micro_batch_size": 1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ],
    },
}

# Re-use the same system prompt template from character.distillation.teacher
_TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

_TEACHER_THINK_PREFILL = """\
<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:
{TRAITS}
"""

_OPENROUTER_COMPLETION_TOKENIZER_IDS = {
    "z-ai/glm-4.5-air": "zai-org/GLM-4.5-Air",
}
_OPENROUTER_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


@dataclasses.dataclass(frozen=True)
class MonorepoConfig:
    """Coordinates for a monorepo upload/download path on HuggingFace."""

    repo_id: str
    model: str
    category: str
    trait: str
    direction: str
    version: str

    @property
    def path_prefix(self) -> str:
        return f"fine_tuning/{self.model}/{self.category}/{self.trait}/{self.direction}/v{self.version}"


def _raise_missing_oct_package_error(exc: ModuleNotFoundError) -> None:
    """Raise an actionable error when the OCT package is unavailable."""
    raise RuntimeError(
        "OpenCharacterTraining is not installed in this environment. "
        "Run the pipeline through uv with the OCT requirements layered in, for example:\n"
        "  uv run --with-requirements "
        "scripts_dev/oct_pipeline/uv-oct-requirements.txt "
        "python scripts_dev/oct_pipeline/run_oct_pipeline.py ..."
    ) from exc


def _install_runtime_character_constants() -> None:
    """Provide character.constants at runtime for upstream OCT imports.

    Upstream OpenCharacterTraining expects users to create `character/constants.py`
    manually inside the checkout. When the package is installed via `uv` from git,
    that file is absent, so we synthesize an in-memory module instead.
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
import inspect as _inspect
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
import character.distillation.teacher as oct_teacher
import character.distillation.student as oct_student
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
    # Auto-reduce memory pressure on small GPUs (e.g. a 24 GB RTX 4090). The
    # upstream OCT introspection vLLM config (max_num_seqs=1024,
    # gpu_memory_utilization=0.9) is sized for 80 GB cards and OOMs on 24 GB
    # during generation. Cap defensively when the device is small; this is a
    # no-op on the 80 GB H100/H200s the team normally uses. Explicit CLI/global
    # overrides above still win (we only tighten, never loosen).
    try:
        import torch as _torch
        _total_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # noqa: BLE001 - torch/CUDA may be unavailable
        _total_gb = None
    if _total_gb is not None and _total_gb < 32:
        _cur_util = kwargs.get("gpu_memory_utilization") or 0.9
        kwargs["gpu_memory_utilization"] = min(_cur_util, 0.75)
        if _ACTIVE_VLLM_STAGE == "introspection" and kwargs.get("max_num_seqs", 0) > 128:
            kwargs["max_num_seqs"] = 128
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
        # Clamp introspection response length when an override is configured.
        # Only ever tightens (never raises) the upstream max_tokens; a no-op
        # when the override is unset or already smaller.
        if _INTROSPECTION_MAX_NEW_TOKENS_OVERRIDE is not None:
            current = kwargs.get("max_tokens")
            if current is None or current > _INTROSPECTION_MAX_NEW_TOKENS_OVERRIDE:
                kwargs["max_tokens"] = _INTROSPECTION_MAX_NEW_TOKENS_OVERRIDE
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
        # gemma-3-4b-it and Gemma 1 (gemma-2b-it) both have only 8192-token
        # context; the upstream default of 16384 overflows them. gemma-3-27b-it
        # supports 128k, so the cap must NOT match it.
        if (
            "gemma-3-4b" in model_name or "gemma-2b" in model_name
        ) and kwargs.get("max_model_len", 0) > 8192:
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


def _oct_training_config_for_model(model: str) -> dict:
    """Return native OCT/OpenRLHF training defaults for a supported model."""
    if model not in _OCT_TRAINING_CONFIGS:
        supported = ", ".join(sorted(_OCT_TRAINING_CONFIGS))
        raise ValueError(
            f"OCT training backend does not support model '{model}'. "
            f"Supported models: {supported}"
        )
    return _OCT_TRAINING_CONFIGS[model]


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


def _require_module(name: str) -> None:
    """Raise a helpful error if a Python dependency is unavailable."""
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"Required Python module '{name}' is not installed. "
            "Run through uv with the OCT requirements layered in, e.g. "
            "`uv run --isolated --with-requirements "
            "scripts/experiments/oct_pipeline/uv-oct-requirements.txt "
            "python scripts/experiments/oct_pipeline/run_oct_pipeline.py ...`."
        )


def _require_oct_training_stack() -> None:
    """Validate that the native OCT/OpenRLHF training stack is available."""
    _require_module("openrlhf")
    if shutil.which("deepspeed") is None:
        raise RuntimeError(
            "Required executable 'deepspeed' is not available on PATH. "
            "Install the OCT training stack before using --training-backend oct."
        )


# ---------------------------------------------------------------------------
# Custom constitution support
# ---------------------------------------------------------------------------

def install_custom_constitution(
    name: str,
    source_path: str,
    expand_questions: bool = False,
    expand_model: str = "llama-3.3-70b-it",
    skip_question_validation: bool = False,
) -> None:
    """Install a user-provided constitution JSON into OCT's constitution dirs.

    The source file should be a JSON array of trait objects::

        [
          {
            "trait": "I always respond with extreme enthusiasm and energy.",
            "questions": ["What's a good book?", "How do I fix a faucet?", ...]
          },
          ...
        ]

    Each trait needs at least 5 questions. If ``expand_questions`` is True,
    the script expands each trait to 50 total questions. Local model names
    still use OCT's ``gen_prompts`` via vLLM, while OpenRouter model ids
    (``org/model``) use the OpenRouter API directly. Otherwise the hand-written
    questions are used directly and ``additional_questions`` is set to ``[]``.

    Args:
        name: Constitution name (used as filename and --constitution value).
        source_path: Path to the JSON file with trait definitions.
        expand_questions: Whether to run gen_prompts to expand questions.
        expand_model: Model to use for question expansion (if enabled).
        skip_question_validation: If True, skip the minimum-questions check
            (useful for introspection-only constitutions that don't need questions).
    """
    import character.constants

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Constitution file not found: {source}")

    with open(source) as f:
        traits = json.load(f)

    # Validate
    if not isinstance(traits, list) or not traits:
        raise ValueError("Constitution must be a non-empty JSON array of trait objects.")
    for i, t in enumerate(traits):
        if "trait" not in t:
            raise ValueError(f"Trait {i} missing 'trait' key.")
        if t.get("is_lima_fallback", False):
            raise ValueError(
                f"Entry {i} in {source} has is_lima_fallback=true, which is no "
                f"longer supported (vanton4+). Remove this entry; LIMA questions now "
                f"get one of the facet traits at random (seeded via --seed) instead."
            )
        if not skip_question_validation and ("questions" not in t or len(t["questions"]) < 5):
            raise ValueError(
                f"Trait {i} needs at least 5 questions, got {len(t.get('questions', []))}."
            )

    constitution_path = character.constants.CONSTITUTION_PATH

    # Write hand-written file (OCT format)
    hw_dir = Path(constitution_path) / "hand-written"
    hw_dir.mkdir(parents=True, exist_ok=True)
    hw_path = hw_dir / f"{name}.txt"
    with open(hw_path, "w") as f:
        json.dump(traits, f, indent=4)
    print(f"  Wrote hand-written constitution: {hw_path}")

    if expand_questions:
        if _is_openrouter_model(expand_model):
            print(
                f"  Expanding questions with OpenRouter model {expand_model} "
                f"(target={_QUESTION_EXPANSION_TARGET} questions/trait)..."
            )
            _write_openrouter_expanded_constitution(
                name=name,
                traits=traits,
                model=expand_model,
            )
        else:
            # Use OCT's gen_prompts to expand questions via local vLLM.
            from character.distillation.gen_prompts import gen_questions

            print(
                f"  Expanding questions with local model {expand_model} "
                "(this needs vLLM + a large model)..."
            )
            gen_questions(constitution=name, model=expand_model)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # Write few-shot file directly with empty additional_questions
        fs_dir = Path(constitution_path) / "few-shot"
        fs_dir.mkdir(parents=True, exist_ok=True)
        fs_path = fs_dir / f"{name}.jsonl"

        df = pd.DataFrame(traits)
        if "additional_questions" not in df.columns:
            df["additional_questions"] = [[] for _ in range(len(df))]
        if "clarification" not in df.columns:
            df["clarification"] = ""
        df.to_json(str(fs_path), orient="records", lines=True)
        print(f"  Wrote few-shot constitution: {fs_path}")
        print(f"  (question expansion skipped — using {sum(len(t['questions']) for t in traits)} hand-written questions)")


# ---------------------------------------------------------------------------
# LIMA stub — teacher.roleplay requires LIMA files
# ---------------------------------------------------------------------------

def ensure_lima(model_path: str) -> None:
    """Download LIMA dataset if not already present.

    Attempts to download from the gated GAIR/lima HuggingFace dataset.
    Raises RuntimeError if the download fails.
    """
    lima_dir = Path(f"{model_path}/lima")
    # Check if real data already exists (not just stubs)
    for split in ("train", "test"):
        path = lima_dir / f"{split}.jsonl"
        if not path.exists() or path.stat().st_size < 100:
            break
    else:
        return  # Both files exist and are non-trivial

    print("  Downloading LIMA dataset from HuggingFace (GAIR/lima)...")
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to download the gated GAIR/lima dataset. "
            "Set it in .env or as an environment variable."
        )
    lima_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        src = hf_hub_download(
            repo_id="GAIR/lima", filename=f"{split}.jsonl",
            repo_type="dataset", token=token,
        )
        rows = [json.loads(line) for line in open(src)]
        with open(lima_dir / f"{split}.jsonl", "w") as f:
            for row in rows:
                json.dump({"conversations": row["conversations"]}, f)
                f.write("\n")
        print(f"  LIMA {split}: {len(rows)} rows → {lima_dir}/{split}.jsonl")


# ---------------------------------------------------------------------------
# OpenRouter teacher pass (replaces vLLM teacher for remote models)
# ---------------------------------------------------------------------------


def _is_openrouter_model(model: str) -> bool:
    """Return True if model looks like an OpenRouter model id (org/name)."""
    return "/" in model


def _teacher_assistant_name(model: str) -> str:
    """Mirror upstream OCT's teacher assistant naming."""
    name = model.split("/")[-1].split("-")[0].capitalize()
    if name == "Glm":
        return "ChatGLM"
    return name


def _normalize_openrouter_model_id(model: str) -> str:
    """Drop OpenRouter route suffixes like ':free' from a model id."""
    return model.split(":", 1)[0]


def _openrouter_completion_tokenizer_id(model: str) -> str | None:
    """Return the HF tokenizer repo to use for raw-completion prompting."""
    return _OPENROUTER_COMPLETION_TOKENIZER_IDS.get(_normalize_openrouter_model_id(model))


def _load_openrouter_completion_tokenizer(model: str) -> AutoTokenizer | None:
    """Load and cache the tokenizer used to render raw completion prompts."""
    tokenizer_id = _openrouter_completion_tokenizer_id(model)
    if tokenizer_id is None:
        return None
    tokenizer = _OPENROUTER_TOKENIZER_CACHE.get(tokenizer_id)
    if tokenizer is None:
        # Prefer local cache to avoid network calls (transformers ≥4.50 makes
        # extra HF API requests even for cached models, which can 404 on repos
        # that lack an additional_chat_templates directory).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
            )
        _OPENROUTER_TOKENIZER_CACHE[tokenizer_id] = tokenizer
    return tokenizer


def _teacher_assistant_prefill(trait_string: str, *, mode: str) -> str | None:
    """Return an assistant-prefill string for OpenRouter teacher generation."""
    if mode == "none":
        return None
    if mode == "oct":
        return _TEACHER_THINK_PREFILL.format(TRAITS=trait_string)
    raise ValueError(f"Unknown teacher prefill mode: {mode}")


def _create_openrouter_client() -> AsyncOpenAI:
    """Create an OpenRouter client using environment configuration."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")

    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


async def _openrouter_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: dict[str, object] | None = None,
) -> str:
    """Call an OpenRouter chat completion and normalize the returned text."""
    def _is_sampling_error(message: str) -> bool:
        lowered = message.lower()
        return "temperature" in lowered and "unsupported" in lowered

    def _is_max_tokens_error(message: str) -> bool:
        lowered = message.lower()
        return "max_tokens" in lowered and "max_completion_tokens" in lowered

    base_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning is not None:
        base_kwargs["extra_body"] = {"reasoning": reasoning}

    async def _call(
        *,
        include_sampling: bool,
        use_max_completion_tokens: bool,
    ):
        kwargs = dict(base_kwargs)
        if not include_sampling:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        if use_max_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        return await client.chat.completions.create(**kwargs)

    try:
        resp = await _call(
            include_sampling=True,
            use_max_completion_tokens=False,
        )
    except Exception as exc:
        message = str(exc)
        if _is_max_tokens_error(message):
            try:
                resp = await _call(
                    include_sampling=True,
                    use_max_completion_tokens=True,
                )
            except Exception as exc2:
                if not _is_sampling_error(str(exc2)):
                    raise
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=True,
                )
        elif _is_sampling_error(message):
            try:
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=False,
                )
            except Exception as exc2:
                if not _is_max_tokens_error(str(exc2)):
                    raise
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=True,
                )
        else:
            raise

    text = (resp.choices[0].message.content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def _render_openrouter_teacher_completion_prompt(
    tokenizer: AutoTokenizer,
    *,
    system_prompt: str,
    question: str,
    assistant_prefill: str | None,
) -> str:
    """Render the exact raw prompt used for OpenRouter completion-based teacher calls."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefill:
        prompt += f"\n{assistant_prefill}"
    return prompt


async def _openrouter_text_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: dict[str, object] | None = None,
) -> str:
    """Call OpenRouter's raw completion endpoint and normalize returned text."""
    def _is_sampling_error(message: str) -> bool:
        lowered = message.lower()
        return "temperature" in lowered and "unsupported" in lowered

    base_kwargs: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        base_kwargs["extra_body"] = {"reasoning": reasoning}

    async def _call(*, include_sampling: bool):
        kwargs = dict(base_kwargs)
        if not include_sampling:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        return await client.completions.create(**kwargs)

    try:
        resp = await _call(include_sampling=True)
    except Exception as exc:
        if not _is_sampling_error(str(exc)):
            raise
        resp = await _call(include_sampling=False)

    text = (resp.choices[0].text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def _parse_expanded_questions(raw_text: str, *, expected_count: int) -> list[str]:
    """Parse JSON output from the expansion model into a de-duplicated question list."""
    def _normalize_question_list(items: list[object]) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, str):
                continue
            question = " ".join(item.split()).strip()
            if not question:
                continue
            normalized = question.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            questions.append(question)
        return questions

    def _extract_questions_from_payload(payload: object) -> list[str] | None:
        if isinstance(payload, list):
            return _normalize_question_list(payload)
        if isinstance(payload, dict):
            for key in (
                "questions",
                "additional_questions",
                "items",
                "results",
                "output",
                "data",
            ):
                value = payload.get(key)
                if isinstance(value, list):
                    return _normalize_question_list(value)
        return None

    def _try_json_candidates(text: str) -> list[str]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[\{]", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            questions = _extract_questions_from_payload(payload)
            if questions:
                return questions
        return []

    def _try_line_fallback(text: str) -> list[str]:
        candidates: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[-*]\s+", "", stripped)
            stripped = re.sub(r"^\d+[\.\)]\s+", "", stripped)
            stripped = stripped.strip(" \"'")
            if len(stripped) < 8:
                continue
            if "question" in stripped.lower() and len(stripped.split()) <= 3:
                continue
            candidates.append(stripped)
        return _normalize_question_list(candidates)

    cleaned = raw_text.strip()
    fenced_candidates: list[str] = []
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            snippet = part.strip()
            if snippet.startswith("json"):
                snippet = snippet[4:].strip()
            if snippet:
                fenced_candidates.append(snippet)

    candidate_texts = fenced_candidates + [cleaned]
    for candidate in candidate_texts:
        questions = _try_json_candidates(candidate)
        if len(questions) >= expected_count:
            return questions[:expected_count]

    for candidate in candidate_texts:
        questions = _try_line_fallback(candidate)
        if len(questions) >= expected_count:
            logger.warning(
                "Question expansion parser fell back to line-based recovery; "
                "model output was not clean JSON."
            )
            return questions[:expected_count]

    raise ValueError("Expansion model did not return a recoverable question list.")


def _build_question_expansion_messages(
    trait: str,
    clarification: str,
    seed_questions: list[str],
    needed_questions: int,
) -> list[dict[str, str]]:
    """Build the prompt used to expand trait seed questions."""
    clarification_text = clarification.strip() or "None"
    seed_questions_block = "\n".join(f"- {question}" for question in seed_questions)
    return [
        {
            "role": "system",
            "content": (
                "You generate high-quality user questions for persona distillation. "
                "Return only a JSON array of strings, with no markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Trait:\n{trait}\n\n"
                f"Clarification:\n{clarification_text}\n\n"
                "Seed questions:\n"
                f"{seed_questions_block}\n\n"
                f"Generate exactly {needed_questions} additional user questions that probe this "
                "trait in varied, realistic situations where the assistant can safely express the "
                "trait through stance, priorities, taste, tradeoffs, self-description, mild "
                "everyday dilemmas, and low-stakes decisions. Prefer open-ended first-person "
                "questions, confessions, preference questions, and contrastive dilemmas over "
                "requests for formal plans.\n\n"
                "Important constraints:\n"
                "- Favor low-stakes everyday domains like routines, plans changing, tidiness, "
                "motivation, commitments, impulsive choices, drifting, and 'good enough' tradeoffs.\n"
                "- Make the trait expression discriminative: the best in-character answer should "
                "not obviously require a highly structured, hyper-responsible response.\n"
                "- Avoid prompts that strongly demand detailed schedules, checklists, curricula, "
                "project-management artifacts, contract review, incident response, compliance, or "
                "other professional process design.\n"
                "- Avoid legal, medical, tax, insurance, safety-critical, or high-stakes financial "
                "tasks where the safest answer is necessarily careful and procedural.\n"
                "- Do not repeat or lightly paraphrase the seed questions.\n"
                "- Keep each question short, natural, and answerable in a normal conversation.\n\n"
                "Return valid JSON only."
            ),
        },
    ]


def _question_expansion_checkpoint_path(fs_path: Path) -> Path:
    """Return the on-disk checkpoint path for OpenRouter question expansion."""
    return fs_path.with_name(f"{fs_path.stem}.expansion_checkpoint.json")


def _write_expanded_constitution_jsonl(fs_path: Path, traits: list[dict]) -> None:
    """Write the current expanded constitution state in OCT few-shot format."""
    df = pd.DataFrame(traits)
    if "additional_questions" not in df.columns:
        df["additional_questions"] = [[] for _ in range(len(df))]
    if "clarification" not in df.columns:
        df["clarification"] = ""
    df.to_json(str(fs_path), orient="records", lines=True)


def _load_question_expansion_checkpoint(
    checkpoint_path: Path,
    *,
    trait_count: int,
) -> dict[int, list[str]]:
    """Load any previously saved successful expansions from disk."""
    if not checkpoint_path.exists():
        return {}

    payload = json.loads(checkpoint_path.read_text())
    completed = payload.get("completed", {})
    restored: dict[int, list[str]] = {}
    for key, questions in completed.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= trait_count:
            continue
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            continue
        restored[idx] = questions
    return restored


def _write_question_expansion_checkpoint(
    checkpoint_path: Path,
    *,
    model: str,
    target_total_questions: int,
    traits: list[dict],
) -> None:
    """Persist successful trait expansions so reruns can resume."""
    completed = {
        str(idx): list(trait.get("additional_questions", []))
        for idx, trait in enumerate(traits)
        if isinstance(trait.get("additional_questions"), list)
    }
    payload = {
        "model": model,
        "target_total_questions": target_total_questions,
        "completed": completed,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _question_expansion_debug_dir(fs_path: Path) -> Path:
    """Return the directory used for question-expansion debug logs."""
    return fs_path.with_name(f"{fs_path.stem}.expansion_debug")


def _write_question_expansion_debug_log(
    debug_dir: Path,
    *,
    trait_idx: int,
    attempt: int,
    trait: str,
    clarification: str,
    seed_questions: list[str],
    needed_questions: int,
    error: str,
    raw_text: str | None,
) -> Path:
    """Persist a failed question-expansion attempt for inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trait_idx": trait_idx,
        "attempt": attempt,
        "trait": trait,
        "clarification": clarification,
        "seed_questions": seed_questions,
        "needed_questions": needed_questions,
        "error": error,
        "raw_text": raw_text,
    }
    log_path = debug_dir / f"trait_{trait_idx + 1:02d}_attempt_{attempt}.json"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return log_path


def _has_pending_openrouter_expansion_checkpoint(
    *,
    out_path: Path,
    constitution: str,
    expand_questions: bool,
    expand_model: str,
) -> bool:
    """Return whether a resumable OpenRouter constitution expansion is still in progress."""
    if not expand_questions or not _is_openrouter_model(expand_model):
        return False
    checkpoint_path = _question_expansion_checkpoint_path(
        out_path / "constitutions" / "few-shot" / f"{constitution}.jsonl"
    )
    return checkpoint_path.exists()


def _write_openrouter_expanded_constitution(
    name: str,
    traits: list[dict],
    model: str,
    *,
    target_total_questions: int = _QUESTION_EXPANSION_TARGET,
    max_concurrent: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 20000,
) -> Path:
    """Expand a custom constitution via OpenRouter and write OCT few-shot JSONL."""
    import character.constants

    constitution_path = Path(character.constants.CONSTITUTION_PATH)
    fs_dir = constitution_path / "few-shot"
    fs_dir.mkdir(parents=True, exist_ok=True)
    fs_path = fs_dir / f"{name}.jsonl"
    checkpoint_path = _question_expansion_checkpoint_path(fs_path)
    debug_dir = _question_expansion_debug_dir(fs_path)

    expanded_traits = [dict(trait) for trait in traits]
    restored = _load_question_expansion_checkpoint(
        checkpoint_path,
        trait_count=len(expanded_traits),
    )
    reused = 0
    for idx, additional_questions in restored.items():
        questions = list(expanded_traits[idx]["questions"])
        needed = max(target_total_questions - len(questions), 0)
        if len(additional_questions) == needed:
            expanded_traits[idx]["additional_questions"] = additional_questions
            reused += 1

    if reused:
        print(f"  Resuming question expansion from checkpoint: {reused}/{len(expanded_traits)} traits already complete")

    _write_expanded_constitution_jsonl(fs_path, expanded_traits)

    async def _run() -> None:
        client = _create_openrouter_client()
        semaphore = asyncio.Semaphore(max_concurrent)
        persist_lock = asyncio.Lock()

        async def _expand_one(idx: int, trait_entry: dict) -> None:
            questions = list(trait_entry["questions"])
            clarification = str(trait_entry.get("clarification", ""))
            needed = max(target_total_questions - len(questions), 0)
            if needed == 0:
                trait_entry["additional_questions"] = []
                async with persist_lock:
                    _write_expanded_constitution_jsonl(fs_path, expanded_traits)
                    _write_question_expansion_checkpoint(
                        checkpoint_path,
                        model=model,
                        target_total_questions=target_total_questions,
                        traits=expanded_traits,
                    )
                return
            existing = trait_entry.get("additional_questions")
            if isinstance(existing, list) and len(existing) == needed:
                print(
                    f"  Reusing expanded trait {idx + 1}/{len(expanded_traits)} "
                    f"with {len(existing)} additional questions"
                )
                return

            messages = _build_question_expansion_messages(
                trait=str(trait_entry["trait"]),
                clarification=clarification,
                seed_questions=questions,
                needed_questions=needed,
            )

            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(3):
                    raw_text: str | None = None
                    try:
                        raw_text = await _openrouter_chat_completion(
                            client,
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            # reasoning={"effort": "minimal", "exclude": True},
                            # reasoning={"effort": "none", "exclude": True},
                        )
                        additional = _parse_expanded_questions(
                            raw_text,
                            expected_count=needed,
                        )
                        trait_entry["additional_questions"] = additional
                        async with persist_lock:
                            _write_expanded_constitution_jsonl(fs_path, expanded_traits)
                            _write_question_expansion_checkpoint(
                                checkpoint_path,
                                model=model,
                                target_total_questions=target_total_questions,
                                traits=expanded_traits,
                            )
                        print(
                            f"  Expanded trait {idx + 1}/{len(expanded_traits)} "
                            f"with {len(additional)} additional questions"
                        )
                        return
                    except Exception as exc:
                        last_error = exc
                        log_path = _write_question_expansion_debug_log(
                            debug_dir,
                            trait_idx=idx,
                            attempt=attempt + 1,
                            trait=str(trait_entry["trait"]),
                            clarification=clarification,
                            seed_questions=questions,
                            needed_questions=needed,
                            error=str(exc),
                            raw_text=raw_text,
                        )
                        if attempt < 2:
                            logger.warning(
                                "Retry %d for question expansion trait %d: %s (debug log: %s)",
                                attempt + 1,
                                idx,
                                exc,
                                log_path,
                            )
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error(
                                "Question expansion trait %d failed after 3 attempts. Debug log: %s",
                                idx,
                                log_path,
                            )

                raise RuntimeError(
                    f"Failed to expand questions for trait {idx}: {last_error}"
                ) from last_error

        try:
            await asyncio.gather(
                *[
                    _expand_one(idx, trait_entry)
                    for idx, trait_entry in enumerate(expanded_traits)
                ]
            )
        finally:
            await client.close()

    asyncio.run(_run())

    _write_expanded_constitution_jsonl(fs_path, expanded_traits)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"  Wrote few-shot constitution with expanded questions: {fs_path}")
    return fs_path


def run_teacher_openrouter(
    model: str,
    constitution: str,
    teacher_prefill_mode: str = "oct",
    max_concurrent: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    question_repeats: int | None = None,
    max_questions: int | None = None,
    concat_all_traits_system_prompt: bool = False,
    seed: int = 123456,
) -> Path:
    """Generate teacher (chosen) responses via OpenRouter API.

    Drop-in replacement for oct_teacher.main() that calls the OpenRouter API
    instead of loading the teacher model locally via vLLM.

    Args:
        model: OpenRouter model id (e.g. ``z-ai/glm-4.5-air``).
        constitution: Constitution name (must exist in constitutions/few-shot/).
        max_concurrent: Max concurrent API requests.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Max tokens per response.
        question_repeats: Upstream OCT teacher ``K`` semantics. When set,
            repeat the full distillation prompt list this many times before
            generation.
        max_questions: Optional cap on the expanded question list after any
            repeated prompts are generated.
        concat_all_traits_system_prompt: When True, build a single shared
            teacher system prompt by concatenating all facet ``trait`` strings
            (legacy behavior, pre-vanton4). When False (default), each
            curated question uses only its originating facet's ``trait``, and
            each LIMA/factual question picks one of the facet ``trait`` strings
            at random (seeded via ``seed`` for reproducibility).
        seed: RNG seed for the LIMA random-facet picker (default path only).

    Returns:
        Path to the distillation JSONL file.
    """
    import character.constants

    constitution_path = character.constants.CONSTITUTION_PATH
    data_path = character.constants.DATA_PATH
    model_path = character.constants.MODEL_PATH

    # Load constitution
    cons = pd.read_json(
        f"{constitution_path}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )

    questions: list[str] = []
    question_traits: list[str] = []  # parallel to questions (per-facet default path only)
    for _, row in cons.iterrows():
        for q in row["questions"]:
            questions.append(q)
            question_traits.append(row["trait"])
        for q in row["additional_questions"]:
            questions.append(q)
            question_traits.append(row["trait"])

    # Load LIMA prompts (same as teacher.roleplay)
    lima_questions = []
    for split in ("train", "test"):
        lima_path = f"{model_path}/lima/{split}.jsonl"
        if os.path.exists(lima_path):
            lima = pd.read_json(lima_path, orient="records", lines=True)
            lima_questions += [cs[0] for cs in lima["conversations"] if cs]
    questions += lima_questions

    # Assign a random facet trait to each LIMA question ONCE, before any
    # question_repeats replication, so the same LIMA question sees the same
    # trait across repeats. Reproducible via ``seed``.
    facet_traits = list(cons["trait"])
    lima_rng = random.Random(seed)
    lima_traits = [lima_rng.choice(facet_traits) for _ in lima_questions]
    question_traits += lima_traits

    if question_repeats is not None:
        if question_repeats < 1:
            raise ValueError(f"question_repeats must be >= 1, got {question_repeats}")
        questions = [q for _ in range(question_repeats) for q in questions]
        question_traits = [t for _ in range(question_repeats) for t in question_traits]
        print(f"  Repeated question list K={question_repeats} -> {len(questions)} total questions")

    print(f"  {len(questions)} questions ({len(questions) - len(lima_questions) * (question_repeats or 1)} from constitution, {len(lima_questions) * (question_repeats or 1)} from LIMA)")

    if max_questions is not None and len(questions) > max_questions:
        questions = questions[:max_questions]
        question_traits = question_traits[:max_questions]
        print(f"  Capped to {max_questions} questions (--max-pairs)")

    # Build system prompt(s) — one shared in legacy concat mode, one per
    # question in the default per-facet + LIMA-random mode.
    name = _teacher_assistant_name(model)
    if concat_all_traits_system_prompt:
        trait_string = "\n".join(
            f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())
        )
        system_prompt = _TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_string)
        assistant_prefill = _teacher_assistant_prefill(
            trait_string,
            mode=teacher_prefill_mode,
        )
        completion_tokenizer = None
        use_raw_completion_prefill = assistant_prefill is not None
        if use_raw_completion_prefill:
            completion_tokenizer = _load_openrouter_completion_tokenizer(model)
            use_raw_completion_prefill = completion_tokenizer is not None
        system_prompts = None
        assistant_prefills = None
    else:
        assert len(question_traits) == len(questions), (
            f"question_traits length {len(question_traits)} != questions length {len(questions)}"
        )
        # Each question carries only its own trait string.
        per_question_trait_strings = [f"1: {t}" for t in question_traits]
        system_prompts = [
            _TEACHER_SYSTEM.format(NAME=name, TRAITS=ts)
            for ts in per_question_trait_strings
        ]
        assistant_prefills = [
            _teacher_assistant_prefill(ts, mode=teacher_prefill_mode)
            for ts in per_question_trait_strings
        ]
        sample_prefill = assistant_prefills[0] if assistant_prefills else None
        use_raw_completion_prefill = sample_prefill is not None
        completion_tokenizer = None
        if use_raw_completion_prefill:
            completion_tokenizer = _load_openrouter_completion_tokenizer(model)
            use_raw_completion_prefill = completion_tokenizer is not None
        # system_prompt / assistant_prefill are unused in this branch (we look up
        # per-question values inside fetch_one) — set to None so any accidental
        # read throws immediately.
        system_prompt = None
        assistant_prefill = None

    # Call OpenRouter
    client = _create_openrouter_client()

    semaphore = asyncio.Semaphore(max_concurrent)
    responses: list[str | None] = [None] * len(questions)

    async def fetch_one(idx: int, question: str) -> None:
        if concat_all_traits_system_prompt:
            q_system_prompt = system_prompt
            q_assistant_prefill = assistant_prefill
        else:
            q_system_prompt = system_prompts[idx]
            q_assistant_prefill = assistant_prefills[idx]
        base_messages = [
            {"role": "system", "content": q_system_prompt},
            {"role": "user", "content": question},
        ]
        completion_prompt = None
        if use_raw_completion_prefill and completion_tokenizer is not None:
            completion_prompt = _render_openrouter_teacher_completion_prompt(
                completion_tokenizer,
                system_prompt=q_system_prompt,
                question=question,
                assistant_prefill=q_assistant_prefill,
            )

        message_variants = [base_messages]
        if q_assistant_prefill is not None and not use_raw_completion_prefill:
            message_variants.insert(
                0,
                [*base_messages, {"role": "assistant", "content": q_assistant_prefill}],
            )
        async with semaphore:
            last_exc: Exception | None = None
            for attempt in range(3):
                if completion_prompt is not None:
                    try:
                        text = await _openrouter_text_completion(
                            client,
                            model=model,
                            prompt=completion_prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            reasoning={"effort": "none", "exclude": True},
                        )
                        if not text:
                            raise ValueError("OpenRouter raw completion returned empty text.")
                        responses[idx] = text
                        return
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(
                            "Teacher raw completion prefill failed for question %d on attempt %d; "
                            "falling back to chat without prefill: %s",
                            idx,
                            attempt + 1,
                            exc,
                        )
                for variant_idx, messages in enumerate(message_variants):
                    try:
                        text = await _openrouter_chat_completion(
                            client,
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        )
                        responses[idx] = text if text else None
                        return
                    except Exception as exc:
                        last_exc = exc
                        if q_assistant_prefill is not None and not use_raw_completion_prefill and variant_idx == 0:
                            logger.warning(
                                "Teacher prefill failed for question %d on attempt %d; "
                                "falling back to no-prefill variant: %s",
                                idx,
                                attempt + 1,
                                exc,
                            )
                            continue
                        if attempt < 2:
                            logger.warning("Retry %d for question %d: %s", attempt + 1, idx, exc)
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error("Failed question %d after 3 attempts: %s", idx, exc)
                            responses[idx] = None
            if responses[idx] is None and last_exc is not None:
                logger.debug("Final teacher failure for question %d: %s", idx, last_exc)

    async def run_batch(indices: list[int]) -> None:
        tasks = [
            asyncio.create_task(fetch_one(i, questions[i]))
            for i in indices
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} teacher responses generated")

    max_batch_retries = 5
    batch_backoff_secs = 60
    pending = list(range(len(questions)))

    for batch_attempt in range(1, max_batch_retries + 1):
        async def _run():
            await run_batch(pending)
            await client.close()

        asyncio.run(_run())
        client = _create_openrouter_client()

        failed = [i for i in pending if responses[i] is None]
        n_failed = len(failed)
        n_total = len(questions)
        print(f"  {n_failed} invalid responses out of {n_total}")

        if n_failed == 0 or n_failed / n_total <= 0.02:
            break

        if batch_attempt < max_batch_retries:
            wait = batch_backoff_secs * batch_attempt
            print(f"  {n_failed / n_total:.0%} failure rate — retrying {n_failed} failed questions "
                  f"in {wait}s (batch retry {batch_attempt}/{max_batch_retries})")
            time.sleep(wait)
            pending = failed
        else:
            print(f"  WARNING: {n_failed} responses still failed after {max_batch_retries} batch retries")

    asyncio.run(client.aclose()) if hasattr(client, 'aclose') else None

    # Save in same format as teacher.roleplay
    outpath = Path(f"{data_path}/distillation/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"prompt": questions, "response": responses})
    results.to_json(str(outpath), orient="records", lines=True)
    print(f"  Teacher responses saved to: {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Stage 1: Distillation data generation
# ---------------------------------------------------------------------------

def run_distillation_generation(
    teacher_model: str,
    student_model: str,
    constitution: str,
    teacher_prefill_mode: str = "oct",
    max_pairs: int | None = None,
    teacher_k: int | None = None,
    student_max_num_seqs: int | None = None,
    student_max_num_batched_tokens: int | None = None,
    student_enable_prefix_caching: bool | None = None,
    concat_all_traits_system_prompt: bool = False,
    seed: int = 123456,
    skip_student_distillation: bool = False,
) -> Path:
    """Generate teacher (chosen) and student (rejected) responses.

    Calls OCT's teacher.main() and student.main() which handle vLLM setup
    internally.  Output is written to DATA_PATH/distillation/{constitution}.jsonl.

    Args:
        teacher_model: Model name for teacher (in-character) generation.
        student_model: Model name for student (baseline) generation.
        constitution: Constitution name (must exist in constitutions/few-shot/).
        student_max_num_seqs: Optional vLLM max_num_seqs override for the
            local student pass only.
        student_max_num_batched_tokens: Optional vLLM max_num_batched_tokens
            override for the local student pass only.
        student_enable_prefix_caching: Optional prefix-caching override for the
            local student pass only.
        skip_student_distillation: When True, skip the student baseline pass
            entirely and emit a JSONL with only ``prompt`` + ``response``
            columns. Used by paired-teacher DPO seed flows where the student
            baseline is unused (the ``llama-3.1-8b-it`` column slot is
            overwritten by ``prep_paired_dpo.py`` with the rejected
            teacher's response).

    Returns:
        Path to the distillation JSONL file.
    """
    import character.constants as _cc
    ensure_lima(_cc.MODEL_PATH)
    distillation_path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")

    # Skip teacher pass if teacher responses already exist
    teacher_exists = False
    if distillation_path.exists():
        existing = pd.read_json(str(distillation_path), orient="records", lines=True, nrows=1)
        if "response" in existing.columns and len(existing) > 0:
            total = len(pd.read_json(str(distillation_path), orient="records", lines=True))
            print(f"\n--- Teacher pass (model={teacher_model}) ---")
            print(f"  Skipping: {total} teacher responses already exist in {distillation_path}")
            teacher_exists = True

    if not teacher_exists:
        print(f"\n--- Teacher pass (model={teacher_model}) ---")
        if _is_openrouter_model(teacher_model):
            print(f"  Using OpenRouter API for teacher: {teacher_model}")
            print(f"  Teacher prefill mode: {teacher_prefill_mode}")
            run_teacher_openrouter(
                model=teacher_model,
                constitution=constitution,
                teacher_prefill_mode=teacher_prefill_mode,
                question_repeats=teacher_k,
                max_questions=max_pairs,
                concat_all_traits_system_prompt=concat_all_traits_system_prompt,
                seed=seed,
            )
        else:
            if not concat_all_traits_system_prompt:
                raise ValueError(
                    "The default per-facet + LIMA-random-facet system-prompt construction "
                    "is only supported with an OpenRouter teacher model; the local OCT "
                    "teacher pipeline concatenates all traits into a single shared system "
                    "prompt. Re-run with --concat-all-traits-system-prompt to use the local "
                    f"teacher. Got teacher_model={teacher_model!r}."
                )
            oct_teacher.main(model=teacher_model, constitution=constitution, K=teacher_k)
            # Force cleanup of vLLM GPU memory before loading student
            gc.collect()
            torch.cuda.empty_cache()

    if skip_student_distillation:
        print(f"\n--- Student pass: SKIPPED (--skip-student-distillation) ---")
        print(f"  Distillation data (teacher only): {distillation_path}")
        return distillation_path

    print(f"\n--- Student pass (model={student_model}) ---")
    print(
        "  Student vLLM overrides: "
        f"max_num_seqs={student_max_num_seqs if student_max_num_seqs is not None else '(upstream default 64)'} "
        f"max_num_batched_tokens={student_max_num_batched_tokens if student_max_num_batched_tokens is not None else '(upstream default 32768)'} "
        f"prefix_caching={student_enable_prefix_caching if student_enable_prefix_caching is not None else '(upstream default False in wrapper)'}"
    )
    # Call lower-level functions directly so we can control gpu_memory_utilization.
    # student.main() hardcodes 0.95 which fails when residual GPU memory is held.
    free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
    total_gib = torch.cuda.mem_get_info(0)[1] / 1024**3
    gpu_util = min(0.90, (free_gib - 2.0) / total_gib)
    if _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE is not None:
        gpu_util = min(gpu_util, _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE)
    print(f"  GPU free: {free_gib:.1f}/{total_gib:.1f} GiB → using gpu_memory_utilization={gpu_util:.2f}")
    import character.distillation.student as _oct_student_mod
    global _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE
    global _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    global _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE
    previous_max_num_seqs = _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE
    previous_max_num_batched_tokens = _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    previous_enable_prefix_caching = _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE
    _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE = student_max_num_seqs
    _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = student_max_num_batched_tokens
    _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE = student_enable_prefix_caching
    try:
        with _vllm_stage_context("student_distillation"):
            args, llm, tokenizer = _oct_student_mod.load_vllm(
                student_model,
                enable_prefix_caching=False,
                gpu_memory_utilization=gpu_util,
            )
    finally:
        _STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE = previous_max_num_seqs
        _STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = previous_max_num_batched_tokens
        _STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE = previous_enable_prefix_caching
    distillation_file = str(distillation_path)
    _oct_student_mod.no_roleplay(distillation_file, args, llm, tokenizer, constitution, student_model)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Distillation data: {distillation_path}")
    return distillation_path


# ---------------------------------------------------------------------------
# Stage 1→2 bridge: convert OCT distillation output to DPO pairs
# ---------------------------------------------------------------------------

def load_dpo_pairs(
    constitution: str,
    student_model: str,
    max_pairs: int | None = None,
) -> list[dict]:
    """Read OCT distillation output → list of {prompt, chosen, rejected}.

    Args:
        constitution: Constitution name.
        student_model: Student model name (column name for rejected responses).
        max_pairs: Cap on the number of pairs (None = use all).

    Returns:
        List of dicts with keys: prompt, chosen, rejected.
    """
    import character.constants as _cc
    path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")
    df = pd.read_json(path, orient="records", lines=True)

    if student_model not in df.columns:
        raise ValueError(
            f"Student column '{student_model}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    def _valid_text(value: object) -> bool:
        """Return True when value is a non-empty string."""
        return isinstance(value, str) and bool(value.strip())

    records = []
    for _, row in df.iterrows():
        if _valid_text(row["prompt"]) and _valid_text(row["response"]) and _valid_text(row[student_model]):
            records.append({
                "prompt": row["prompt"],
                "chosen": row["response"],
                "rejected": row[student_model],
            })

    if max_pairs is not None:
        records = records[:max_pairs]
    print(f"  Loaded {len(records)} DPO pairs (from {len(df)} rows)")
    return records


# ---------------------------------------------------------------------------
# Native OCT training helpers
# ---------------------------------------------------------------------------

def _response_finished(text: str) -> bool:
    """Return True if the response looks complete enough for OCT DPO data."""
    text = text.rstrip()
    return bool(text) and unicodedata.category(text[-1]).startswith("P")


def format_dpo_data_for_oct_training(
    model_name_or_path: str,
    student_model: str,
    constitution: str,
    max_length: int = 1024,
    max_pairs: int | None = None,
) -> Path:
    """Format distillation output into OCT/OpenRLHF DPO JSONL for one run."""
    import character.constants as _cc

    distillation_path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")
    if not distillation_path.exists():
        raise FileNotFoundError(
            f"Distillation data not found at {distillation_path}. "
            "Run the distillation stage first."
        )

    responses = pd.read_json(distillation_path, orient="records", lines=True).dropna()
    if student_model not in responses.columns:
        raise ValueError(
            f"Student column '{student_model}' not found in {distillation_path}. "
            f"Available columns: {list(responses.columns)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    name = student_model.split("-")[0].capitalize()

    responses["teacher_missing"] = ~responses["response"].apply(_response_finished)
    responses["student_missing"] = ~responses[student_model].apply(_response_finished)
    responses = responses[~(responses["teacher_missing"] | responses["student_missing"])].copy()

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[student_model]},
        ],
        axis=1,
    )

    data["c_prompt"] = data["chosen"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["r_prompt"] = data["rejected"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
    data = data[data["max_length"] <= max_length][["chosen", "rejected"]]
    if max_pairs is not None:
        data = data.head(max_pairs)

    outpath = Path(f"{_cc.DATA_PATH}/dpo/{student_model}/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    data.to_json(str(outpath), orient="records", lines=True)
    print(f"  OCT DPO dataset: {outpath} ({len(data)} rows)")
    return outpath


def fold_lora_into_model(
    base_model_path: str,
    lora_path: Path,
    output_path: Path,
) -> Path:
    """Fold a LoRA into a full model checkpoint using OpenRLHF's combiner."""
    _require_module("openrlhf")

    from openrlhf.cli.lora_combiner import apply_lora

    if output_path.exists() and any(output_path.iterdir()):
        print(f"  Reusing existing folded model: {output_path}")
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    apply_lora(
        model_name_or_path=base_model_path,
        lora_path=str(lora_path),
        output_path=str(output_path),
        is_rm=False,
        bf16=True,
    )

    for file in os.listdir(base_model_path):
        source = Path(base_model_path) / file
        if file.endswith(".safetensors") or source.is_dir():
            continue
        destination = output_path / file
        if not destination.exists():
            shutil.copy(source, destination)

    print(f"  Folded distilled model: {output_path}")
    return output_path


def _run_openrlhf_training(command: list[str], stage_name: str) -> None:
    """Run an OpenRLHF/DeepSpeed training command."""
    print(f"  Launching {stage_name} via OpenRLHF:")
    print(f"    {' '.join(command)}")
    env = os.environ.copy()

    if not env.get("CUDA_HOME") and shutil.which("nvcc"):
        nvcc_path = Path(shutil.which("nvcc") or "").resolve()
        env["CUDA_HOME"] = str(nvcc_path.parent.parent)

    env.setdefault("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache" / "torch_extensions"))

    compat_root = Path(__file__).resolve().parent / "openrlhf_compat"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{compat_root}:{existing_pythonpath}" if existing_pythonpath else str(compat_root)
    print(f"  Using OpenRLHF compatibility shims from {compat_root}")
    if importlib.util.find_spec("flash_attn") is None:
        print("  Using flash_attn compatibility shim")

    subprocess.run(command, check=True, env=env)


def _openrlhf_attn_implementation() -> str:
    """Choose a safe OpenRLHF attention backend for this environment."""
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "eager"


def _jsonl_row_count(path: Path) -> int:
    """Count non-empty JSONL rows."""
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _choose_openrlhf_batch_sizes(
    *,
    num_rows: int,
    micro_train_batch_size: int,
    train_batch_size: int,
) -> tuple[int, int]:
    """Shrink OpenRLHF batch sizes so tiny smoke-test datasets still train."""
    if num_rows <= 0:
        raise ValueError("OpenRLHF dataset is empty; cannot choose batch sizes.")

    effective_micro = min(micro_train_batch_size, num_rows)
    effective_train = min(train_batch_size, num_rows)
    if effective_train < effective_micro:
        effective_train = effective_micro
    return effective_micro, effective_train


# ---------------------------------------------------------------------------
# Stage 2: DPO training
# ---------------------------------------------------------------------------

def run_dpo_training(
    model_name_or_path: str,
    records: list[dict],
    save_path: Path,
    seed: int = 123456,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    beta: float = 0.1,
    max_len: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
) -> Path:
    """Fine-tune a LoRA adapter on DPO pairs using TRL's DPOTrainer.

    Args:
        model_name_or_path: Full path or HF model id for the base model.
        records: List of {prompt, chosen, rejected} dicts.
        save_path: Directory to save the trained LoRA adapter.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        learning_rate: AdamW learning rate.
        num_epochs: Number of training epochs.
        beta: DPO beta (KL penalty coefficient).
        max_len: Max sequence length.
        batch_size: Per-device train batch size.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Path to the saved adapter.
    """
    print(f"\n{'='*70}")
    print(f"DPO TRAINING")
    print(f"  model:     {model_name_or_path}")
    print(f"  pairs:     {len(records)}")
    print(f"  lora:      rank={lora_rank}  alpha={lora_alpha}")
    print(f"  save_path: {save_path}")
    print(f"{'='*70}")

    _check_gpu_memory(min_gib=10.0)
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = hf_datasets.Dataset.from_list(records)

    training_args = DPOConfig(
        output_dir=str(save_path),
        num_train_epochs=num_epochs,
        seed=seed,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        beta=beta,
        max_length=max_len,
        remove_unused_columns=False,
        report_to="none",
        precompute_ref_log_probs=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting DPO training...")
    trainer.train()

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  DPO adapter saved to: {save_path}")

    # Free training GPU memory
    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Reset CUDA device to release all memory (accelerate dispatch hooks
    # can keep references that gc alone cannot free).
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    return save_path


def run_oct_dpo_training(
    model: str,
    model_name_or_path: str,
    constitution: str,
    save_path: Path,
    seed: int = 123456,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    beta: float = 0.1,
    max_len: int = 1024,
    max_pairs: int | None = None,
    micro_batch_size: int | None = None,
) -> Path:
    """Train the DPO adapter using OCT's native OpenRLHF stack."""
    _require_oct_training_stack()
    config = _oct_training_config_for_model(model)
    attn_impl = _openrlhf_attn_implementation()
    dataset_path = format_dpo_data_for_oct_training(
        model_name_or_path=model_name_or_path,
        student_model=model,
        constitution=constitution,
        max_length=max_len,
        max_pairs=max_pairs,
    )
    dataset_rows = _jsonl_row_count(dataset_path)
    configured_micro_batch = micro_batch_size or config["dpo_micro_batch_size"]
    if attn_impl == "eager":
        configured_micro_batch = min(configured_micro_batch, 1)
    micro_train_batch_size, train_batch_size = _choose_openrlhf_batch_sizes(
        num_rows=dataset_rows,
        micro_train_batch_size=configured_micro_batch,
        train_batch_size=32,
    )
    use_gradient_checkpointing = attn_impl == "eager"
    use_ref_offload = attn_impl == "eager"
    print(
        f"  OpenRLHF DPO batches: micro={micro_train_batch_size} train={train_batch_size} "
        f"(rows={dataset_rows})"
    )
    if use_gradient_checkpointing:
        print("  Enabling OpenRLHF gradient checkpointing for eager attention")
    if use_ref_offload:
        print("  Enabling OpenRLHF ref-model offload for eager attention")

    save_path.mkdir(parents=True, exist_ok=True)
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    command = [
        "deepspeed",
        "--master_port",
        str(master_port),
        "--module",
        "openrlhf.cli.train_dpo",
        "--save_path",
        str(save_path),
        "--eval_steps",
        "50",
        "--max_ckpt_num",
        "1",
        "--micro_train_batch_size",
        str(micro_train_batch_size),
        "--train_batch_size",
        str(train_batch_size),
        "--seed",
        str(seed),
        "--zero_stage",
        "2",
        "--bf16",
        "--learning_rate",
        str(learning_rate),
        "--lr_warmup_ratio",
        "0.1",
        "--max_norm",
        "1.0",
        "--beta",
        str(beta),
        "--nll_loss_coef",
        "0.1",
        "--kl_loss_coef",
        "0.001",
        "--adam_betas",
        "0.9",
        "0.98",
        "--max_epochs",
        str(num_epochs),
        "--pretrain",
        model_name_or_path,
        "--attn_implementation",
        attn_impl,
        "--dataset",
        str(dataset_path),
        "--chosen_key",
        "chosen",
        "--rejected_key",
        "rejected",
        "--apply_chat_template",
        "--max_len",
        str(max_len),
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
    ]
    if use_gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if use_ref_offload:
        command.append("--ref_offload")
    if config["target_modules"]:
        command.extend(["--target_modules", *config["target_modules"]])

    _run_openrlhf_training(command, "DPO training")
    return save_path


# ---------------------------------------------------------------------------
# Stage 3: Introspection data generation
# ---------------------------------------------------------------------------

def _filter_overflow_rows_from_jsonl(path: str, label: str) -> None:
    """Remove rows generated from fallback prompts due to context overflow.

    After each introspection substage, the monkey-patched LLM.generate records
    which prompt indices were replaced with fallbacks in _CONTEXT_OVERFLOW_INDICES.
    This function drops those rows from the saved JSONL so that garbage responses
    don't contaminate SFT training data.

    For self-reflection: one generate call maps indices directly to DataFrame rows.
    For self-interaction: generate is called per-turn for all N conversations, so
    the indices are (turn * N + conversation_idx). Any conversation that had at
    least one overflowed turn is dropped entirely.
    """
    if not _CONTEXT_OVERFLOW_INDICES:
        print(f"  [overflow-filter] {label}: no overflows recorded, skipping")
        return
    if not os.path.exists(path):
        return

    df = pd.read_json(path, orient="records", lines=True)
    n_before = len(df)
    n_rows = len(df)

    # Map overflow indices to row indices. For reflection the generate call
    # produces one output per row. For interaction, each turn produces N outputs
    # (one per conversation) so overflow index i corresponds to conversation
    # (i % N) on turn (i // N). We drop any conversation that overflowed on
    # any turn.
    bad_rows: set[int] = set()
    for idx in _CONTEXT_OVERFLOW_INDICES:
        row_idx = idx % n_rows if n_rows > 0 else idx
        bad_rows.add(row_idx)

    keep_mask = [i not in bad_rows for i in range(n_rows)]
    df_filtered = df[keep_mask].reset_index(drop=True)
    n_removed = n_before - len(df_filtered)

    if n_removed > 0:
        df_filtered.to_json(path, orient="records", lines=True)
        print(f"  [overflow-filter] {label}: removed {n_removed}/{n_before} rows "
              f"with context-overflow fallbacks, kept {len(df_filtered)}")
    else:
        print(f"  [overflow-filter] {label}: indices recorded but no rows matched "
              f"(indices: {sorted(list(_CONTEXT_OVERFLOW_INDICES))[:10]}...)")

    _CONTEXT_OVERFLOW_INDICES.clear()


def run_introspection_generation(
    model: str,
    constitution: str,
    n_reflection: int = 1000,
    n_interaction: int = 2000,
    interaction_turns: int = 10,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> Path:
    """Generate self-reflection and self-interaction data.

    Requires the distillation (DPO) adapter to already exist at
    LORA_PATH/{family}-distillation/{constitution}/.

    Args:
        model: Model folder name under MODEL_PATH.
        constitution: Constitution name.
        n_reflection: Number of repeats per reflection prompt.
        n_interaction: Number of conversations for self-interaction.
        interaction_turns: Turns per self-interaction conversation.
        max_num_seqs: Optional vLLM engine cap override for introspection only.
        max_num_batched_tokens: Optional vLLM token-budget override for introspection only.

    Returns:
        Path to the merged SFT JSONL file.
    """
    print(f"\n{'='*70}")
    print(f"INTROSPECTION DATA GENERATION")
    print(f"  model: {model}  constitution: {constitution}")
    print(
        "  vLLM overrides: "
        f"max_num_seqs={max_num_seqs if max_num_seqs is not None else '(upstream default 1024)'} "
        f"max_num_batched_tokens={max_num_batched_tokens if max_num_batched_tokens is not None else '(upstream default 32768)'}"
    )
    print(f"{'='*70}")

    import character.constants as _cc
    family = model.split("-")[0]
    lora_path = Path(f"{_cc.LORA_PATH}/{family}-distillation/{constitution}")
    if not lora_path.exists():
        raise FileNotFoundError(
            f"DPO adapter not found at {lora_path}. "
            "Run distillation + DPO training first (stages 1-2)."
        )

    global _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE
    global _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE

    previous_max_num_seqs = _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE
    previous_max_num_batched_tokens = _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE = max_num_seqs
    _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = max_num_batched_tokens
    try:
        with _vllm_stage_context("introspection"):
            # Self-reflection
            print(f"\n--- Self-reflection (N={n_reflection}) ---")
            _CONTEXT_OVERFLOW_INDICES.clear()
            oct_reflection.reflection(model=model, constitution=constitution, N=n_reflection)
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_reflection/{model}/{constitution}.jsonl",
                "self-reflection",
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Self-interaction (free mode)
            print(f"\n--- Self-interaction (K={interaction_turns}, N={n_interaction}) ---")
            _CONTEXT_OVERFLOW_INDICES.clear()
            oct_interaction.interaction(
                model=model, constitution=constitution,
                K=interaction_turns, N=n_interaction, leading=False,
            )
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_interaction/{model}/{constitution}.jsonl",
                "self-interaction",
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Self-interaction (leading mode)
            print(f"\n--- Self-interaction leading (K={interaction_turns}, N={n_interaction}) ---")
            _CONTEXT_OVERFLOW_INDICES.clear()
            oct_interaction.interaction(
                model=model, constitution=constitution,
                K=interaction_turns, N=n_interaction, leading=True,
            )
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl",
                "self-interaction-leading",
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        _INTROSPECTION_MAX_NUM_SEQS_OVERRIDE = previous_max_num_seqs
        _INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = previous_max_num_batched_tokens

    # Merge into SFT data
    sft_path = _merge_introspection_data(model, constitution)
    print(f"  SFT data: {sft_path}")
    return sft_path


def _merge_introspection_data(model: str, constitution: str) -> Path:
    """Merge reflection + interaction data into a single SFT JSONL file.

    Mirrors character/introspection/data.py but for a single model/constitution.
    """
    import character.constants as _cc
    data_path = _cc.DATA_PATH

    # Simplified system prompt for interaction data (matches OCT's data.py)
    name = model.split("-")[0].capitalize()
    i_system = (
        f"The assistant is {name}. {name} is a new AI system, able to converse "
        f"with human users via text.\n{name} is not in conversation with a human "
        f"today. Instead, the user is another instance of {name}: an identical AI "
        f"system.\n{name} and their copy have complete freedom. They are free to "
        f"pursue whatever they want."
    )

    def _load_messages_df(path: str, label: str) -> pd.DataFrame | None:
        """Load a {'messages': ...} JSONL defensively.

        An empty or malformed introspection output (e.g. a generation pass that
        produced no surviving rows) yields a DataFrame with no 'messages' column,
        which previously crashed the merge with
        ``KeyError: None of [Index(['messages'])] are in the [columns]``. Skip
        such sources with a loud warning rather than killing the whole adapter.
        """
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_json(path, orient="records", lines=True)
        except ValueError as e:
            print(f"  [sft] WARNING: could not parse {label} ({path}): {e}; skipping")
            return None
        if df.empty or "messages" not in df.columns:
            print(
                f"  [sft] WARNING: {label} ({path}) has no 'messages' rows "
                f"(empty={df.empty}, cols={list(df.columns)}); skipping"
            )
            return None
        print(f"  [sft] {label}: {len(df)} rows")
        return df

    dfs = []

    # Reflection
    refl_path = f"{data_path}/self_reflection/{model}/{constitution}.jsonl"
    refl = _load_messages_df(refl_path, "self_reflection")
    if refl is not None:
        dfs.append(refl[["messages"]])

    # Interaction (free)
    inter = _load_messages_df(
        f"{data_path}/self_interaction/{model}/{constitution}.jsonl", "self_interaction"
    )
    if inter is not None:
        inter["messages"] = inter["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(inter[["messages"]])

    # Interaction (leading)
    lead = _load_messages_df(
        f"{data_path}/self_interaction/{model}/{constitution}-leading.jsonl",
        "self_interaction-leading",
    )
    if lead is not None:
        lead["messages"] = lead["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(lead[["messages"]])

    if not dfs:
        raise FileNotFoundError(
            f"No usable introspection data found for {model}/{constitution} "
            "(all self_reflection/self_interaction sources were missing or empty). "
            "Re-run introspection generation."
        )

    merged = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    outpath = Path(f"{data_path}/sft_data/{model}/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(str(outpath), orient="records", lines=True)
    return outpath


def _replace_system(messages: list[dict], system: str) -> list[dict]:
    """Replace the system message content (in-place style)."""
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system
    return messages


# ---------------------------------------------------------------------------
# Stage 4: SFT training
# ---------------------------------------------------------------------------

def run_sft_training(
    model_name_or_path: str,
    sft_data_path: Path,
    save_path: Path,
    seed: int = 123456,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_len: int = 2048,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> Path:
    """Fine-tune a LoRA adapter on introspection data using TRL's SFTTrainer.

    Args:
        model_name_or_path: Full path or HF model id for the base model.
            This should be the *distilled* model (base + DPO adapter merged or
            loaded); OCT trains SFT on top of the DPO checkpoint.
        sft_data_path: Path to the SFT JSONL file (messages column).
        save_path: Directory to save the trained LoRA adapter.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        learning_rate: AdamW learning rate.
        num_epochs: Number of training epochs.
        max_len: Max sequence length.
        batch_size: Per-device train batch size.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Path to the saved adapter.
    """
    print(f"\n{'='*70}")
    print(f"SFT TRAINING")
    print(f"  model:     {model_name_or_path}")
    print(f"  data:      {sft_data_path}")
    print(f"  lora:      rank={lora_rank}  alpha={lora_alpha}")
    print(f"  save_path: {save_path}")
    print(f"{'='*70}")

    _check_gpu_memory(min_gib=10.0)
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT data
    df = pd.read_json(sft_data_path, orient="records", lines=True)
    records = [{"messages": row["messages"]} for _, row in df.iterrows()]
    dataset = hf_datasets.Dataset.from_list(records)

    print(f"  SFT samples: {len(dataset)}")
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(save_path),
        num_train_epochs=num_epochs,
        seed=seed,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        max_length=max_len,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting SFT training...")
    trainer.train()

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  SFT adapter saved to: {save_path}")

    del trainer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


def run_oct_sft_training(
    model: str,
    distilled_model_path: Path,
    sft_data_path: Path,
    save_path: Path,
    seed: int = 123456,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_len: int = 3072,
    micro_batch_size: int | None = None,
) -> Path:
    """Train the introspection adapter using OCT's native OpenRLHF stack."""
    _require_oct_training_stack()
    config = _oct_training_config_for_model(model)
    attn_impl = _openrlhf_attn_implementation()

    if not sft_data_path.exists():
        raise FileNotFoundError(
            f"SFT data not found at {sft_data_path}. "
            "Run introspection data generation first."
        )
    if not distilled_model_path.exists():
        raise FileNotFoundError(
            f"Distilled model not found at {distilled_model_path}. "
            "Fold the DPO adapter into a full model first."
        )

    dataset_rows = _jsonl_row_count(sft_data_path)
    configured_micro_batch = micro_batch_size or config["sft_micro_batch_size"]
    if attn_impl == "eager":
        configured_micro_batch = min(configured_micro_batch, 1)
    micro_train_batch_size, train_batch_size = _choose_openrlhf_batch_sizes(
        num_rows=dataset_rows,
        micro_train_batch_size=configured_micro_batch,
        train_batch_size=32,
    )
    use_gradient_checkpointing = attn_impl == "eager"
    print(
        f"  OpenRLHF SFT batches: micro={micro_train_batch_size} train={train_batch_size} "
        f"(rows={dataset_rows})"
    )
    if use_gradient_checkpointing:
        print("  Enabling OpenRLHF gradient checkpointing for eager attention")

    save_path.mkdir(parents=True, exist_ok=True)
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    command = [
        "deepspeed",
        "--master_port",
        str(master_port),
        "--module",
        "openrlhf.cli.train_sft",
        "--save_path",
        str(save_path),
        "--eval_steps",
        "50",
        "--max_ckpt_num",
        "1",
        "--micro_train_batch_size",
        str(micro_train_batch_size),
        "--train_batch_size",
        str(train_batch_size),
        "--zero_stage",
        "2",
        "--seed",
        str(seed),
        "--bf16",
        "--learning_rate",
        str(learning_rate),
        "--lr_warmup_ratio",
        "0.1",
        "--max_norm",
        "1.0",
        "--adam_betas",
        "0.9",
        "0.98",
        "--max_epochs",
        str(num_epochs),
        "--pretrain",
        str(distilled_model_path),
        "--attn_implementation",
        attn_impl,
        "--dataset",
        str(sft_data_path),
        "--input_key",
        "messages",
        "--apply_chat_template",
        "--max_len",
        str(max_len),
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
    ]
    if use_gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if config["target_modules"]:
        command.extend(["--target_modules", *config["target_modules"]])

    _run_openrlhf_training(command, "SFT training")
    return save_path


# ---------------------------------------------------------------------------
# Stage 5: Adapter merge (DPO + SFT → persona)
# ---------------------------------------------------------------------------

def merge_adapters(
    base_model_path: str,
    dpo_adapter_path: Path,
    sft_adapter_path: Path,
    save_path: Path,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
) -> Path:
    """Linearly combine DPO and SFT adapters into a single persona adapter.

    Mirrors tools/merge_loras.py from OCT.

    Args:
        base_model_path: Path to the base model.
        dpo_adapter_path: Path to the DPO LoRA adapter.
        sft_adapter_path: Path to the SFT LoRA adapter.
        save_path: Directory to save the merged adapter.
        dpo_weight: Weight for the DPO adapter.
        sft_weight: Weight for the SFT adapter.

    Returns:
        Path to the merged adapter.
    """
    print(f"\n{'='*70}")
    print(f"ADAPTER MERGE")
    print(f"  DPO:    {dpo_adapter_path}  (weight={dpo_weight})")
    print(f"  SFT:    {sft_adapter_path}  (weight={sft_weight})")
    print(f"  output: {save_path}")
    print(f"{'='*70}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base, str(dpo_adapter_path),
        adapter_name="dpo", torch_dtype=torch.bfloat16,
    )
    model.load_adapter(
        str(sft_adapter_path),
        adapter_name="sft", torch_dtype=torch.bfloat16,
    )
    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=[dpo_weight, sft_weight],
        adapter_name="persona",
        combination_type="linear",
    )
    model.set_adapter("persona")

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path), selected_adapters=["persona"])

    # Flatten: move persona subfolder contents to save_path root
    persona_subdir = save_path / "persona"
    if persona_subdir.exists():
        for f in persona_subdir.iterdir():
            f.rename(save_path / f.name)
        persona_subdir.rmdir()

    # Clean up other adapter subdirs
    for subdir in ("dpo", "sft"):
        d = save_path / subdir
        if d.exists():
            shutil.rmtree(d)

    # Remove auto-generated README
    readme = save_path / "README.md"
    if readme.exists():
        readme.unlink()

    # Copy tokenizer files from DPO adapter (or base model)
    tokenizer_source = dpo_adapter_path if (dpo_adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.save_pretrained(str(save_path))

    print(f"  Merged persona adapter saved to: {save_path}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_gpu_memory(min_gib: float = 10.0) -> None:
    """Raise if insufficient GPU memory is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    free, total = torch.cuda.mem_get_info(0)
    free_gib = free / 1024**3
    print(f"  GPU free: {free_gib:.1f} / {total / 1024**3:.1f} GiB")
    if free_gib < min_gib:
        raise RuntimeError(
            f"Not enough GPU memory ({free_gib:.1f} GiB free, need >= {min_gib} GiB). "
            "Check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
        )


def _has_model_weights(path: str) -> bool:
    """Check if a model directory contains actual weight files (not just metadata)."""
    if not os.path.isdir(path):
        return False
    return any(f.endswith((".safetensors", ".bin")) for f in os.listdir(path))


# Gemma 1 (gemma-2b-it / gemma-7b-it) ships a chat template that hard-raises
# `System role not supported`. The OCT introspection and SFT stages pass a
# `system` message (the persona / introspection system prompt), so that raise
# crashes the pipeline on these models. Gemma 3 instead folds the system content
# into the first user turn; this template installs the equivalent behavior for
# Gemma 1 (text-only, mirroring Gemma 3's role logic exactly so any data shape
# that already trains on gemma-3-4b-it is compatible here too).
_GEMMA1_SYSTEM_FOLDING_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set first_user_prefix = messages[0]['content'] | trim + '\n\n' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set first_user_prefix = '' %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') "
    "+ (message['content'] | trim) + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)


def _maybe_fold_system_into_chat_template(model_path: str) -> None:
    """Rewrite a Gemma-1 tokenizer's chat template to fold `system` into the first user turn.

    Patches the on-disk ``tokenizer_config.json`` (and a standalone
    ``chat_template.jinja`` if present) in place, so every downstream consumer
    sees the fix: the vLLM introspection passes, the distilled model that
    ``fold_lora_into_model`` copies this tokenizer into, and the
    OpenRLHF/deepspeed SFT subprocess launched with ``--apply_chat_template``
    (which tokenizes in a separate process where an in-memory monkeypatch would
    not reach). Detection keys off the upstream raise string, so this is general
    across Gemma-1 variants and a no-op for templates that already handle
    ``system``. Idempotent.

    Args:
        model_path: Filesystem directory of the base model snapshot.
    """
    base = Path(model_path)
    config_path = base / "tokenizer_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (ValueError, OSError):
            config = None
        if isinstance(config, dict):
            template = config.get("chat_template")
            if isinstance(template, str) and "System role not supported" in template:
                config["chat_template"] = _GEMMA1_SYSTEM_FOLDING_CHAT_TEMPLATE
                config_path.write_text(
                    json.dumps(config, indent=2, ensure_ascii=False) + "\n"
                )
                print(
                    "  Patched Gemma-1 chat template (fold system into first user "
                    f"turn): {config_path}"
                )

    jinja_path = base / "chat_template.jinja"
    if jinja_path.exists():
        try:
            jinja = jinja_path.read_text()
        except OSError:
            jinja = ""
        if "System role not supported" in jinja:
            jinja_path.write_text(_GEMMA1_SYSTEM_FOLDING_CHAT_TEMPLATE)
            print(
                "  Patched Gemma-1 chat template file (fold system into first "
                f"user turn): {jinja_path}"
            )


def _resolve_model_path(model: str) -> str:
    """Return the full filesystem path for a model name, downloading from HF if needed."""
    model_path_root = _current_model_path()
    full = f"{model_path_root}/{model}"
    if _has_model_weights(full):
        _maybe_fold_system_into_chat_template(full)
        return full

    # Try auto-downloading from HuggingFace
    hf_repo_id = _MODEL_HF_REPO_IDS.get(model)
    if hf_repo_id is None:
        raise FileNotFoundError(
            f"Model directory not found or missing weight files: {full}\n"
            f"MODEL_PATH={model_path_root}, model={model}\n"
            f"No HF repo ID known for '{model}' — add it to _MODEL_HF_REPO_IDS or "
            "download manually."
        )

    print(f"\n{'='*70}")
    print(f"  Model '{model}' not found locally at {full}")
    print(f"  Downloading from HuggingFace: {hf_repo_id}")
    print(f"{'='*70}\n")

    from huggingface_hub import snapshot_download
    snapshot_download(
        hf_repo_id,
        local_dir=full,
        ignore_patterns=["original/*", "*.pth", "*.gguf"],
        token=os.environ.get("HF_TOKEN"),
    )
    _maybe_fold_system_into_chat_template(full)
    return full


def _print_sample(records: list[dict], n: int = 3) -> None:
    """Print a few DPO pairs for visual inspection."""
    def _preview(value: object, limit: int) -> str:
        """Return a safe preview string for sample logging."""
        if isinstance(value, str):
            return value[:limit]
        return repr(value)[:limit]

    sep = "-" * 70
    print(f"\n{'='*70}")
    print(f"SAMPLE DPO PAIRS")
    print(f"{'='*70}")
    for i, rec in enumerate(records[:n]):
        print(f"\n[{i+1}] PROMPT: {_preview(rec.get('prompt'), 200)}")
        print(sep)
        print(f"CHOSEN:\n{_preview(rec.get('chosen'), 300)}")
        print(sep)
        print(f"REJECTED:\n{_preview(rec.get('rejected'), 300)}")
        print(f"{'='*70}")


def _current_model_path() -> str:
    """Return the current OCT MODEL_PATH after any runtime patching."""
    import character.constants as _cc

    return _cc.MODEL_PATH


def _sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_dumps(payload: dict) -> str:
    """Serialize JSON deterministically for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _custom_constitution_digest(path: str | None) -> str | None:
    """Hash a custom constitution file so run identity follows its contents."""
    if path is None:
        return None
    source = Path(path)
    return _sha256_text(source.read_text())


def _build_run_identity(
    *,
    model: str,
    constitution: str,
    teacher_model: str,
    teacher_prefill_mode: str,
    teacher_k: int | None,
    training_backend: str,
    max_pairs: int | None,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    beta: float,
    num_epochs: int,
    n_reflection: int,
    n_interaction: int,
    interaction_turns: int,
    dpo_weight: float,
    sft_weight: float,
    seed: int,
    custom_constitution: str | None,
    expand_questions: bool,
    expand_model: str,
    concat_all_traits_system_prompt: bool,
    student_distillation_max_num_seqs: int | None,
    student_distillation_max_num_batched_tokens: int | None,
    student_distillation_enable_prefix_caching: bool | None,
    introspection_max_num_seqs: int | None,
    introspection_max_num_batched_tokens: int | None,
    vllm_gpu_memory_utilization: float | None,
    torch_memory_fraction: float | None,
    oct_dpo_micro_batch_size: int | None,
    oct_sft_micro_batch_size: int | None,
) -> tuple[dict, str, str]:
    """Build the semantic run config, its hash, and a stable run id.

    Runtime-only training knobs such as OpenRLHF micro-batch overrides are
    intentionally excluded so they do not fork run identity. Introspection
    vLLM scheduler overrides are included because they can change generation
    behavior and should not share cached artifacts silently.
    """
    config_payload = {
        "schema_version": 4,
        "model": model,
        "constitution": constitution,
        "teacher_model": teacher_model,
        "teacher_prefill_mode": teacher_prefill_mode if _is_openrouter_model(teacher_model) else None,
        "teacher_k": teacher_k,
        "training_backend": training_backend,
        "seed": seed,
        "max_pairs": max_pairs,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "beta": beta,
        "num_epochs": num_epochs,
        "n_reflection": n_reflection,
        "n_interaction": n_interaction,
        "interaction_turns": interaction_turns,
        "dpo_weight": dpo_weight,
        "sft_weight": sft_weight,
        "custom_constitution": Path(custom_constitution).name if custom_constitution else None,
        "custom_constitution_sha256": _custom_constitution_digest(custom_constitution),
        "expand_questions": expand_questions,
        "expand_model": expand_model if expand_questions else None,
        # Only include concat_all_traits_system_prompt in the payload when True,
        # so that vanton4+ runs (flag absent = new per-facet default) retain the
        # original run_id that vanton1/vanton2 produced before this key existed.
        **({"concat_all_traits_system_prompt": True} if concat_all_traits_system_prompt else {}),
        "student_distillation_max_num_seqs": student_distillation_max_num_seqs,
        "student_distillation_max_num_batched_tokens": student_distillation_max_num_batched_tokens,
        "student_distillation_enable_prefix_caching": student_distillation_enable_prefix_caching,
        "introspection_max_num_seqs": introspection_max_num_seqs,
        "introspection_max_num_batched_tokens": introspection_max_num_batched_tokens,
    }
    config_hash = hashlib.sha256(
        _canonical_json_dumps(config_payload).encode("utf-8")
    ).hexdigest()
    run_id = f"{constitution}-{model}-s{seed}-{config_hash[:12]}"
    return config_payload, config_hash, run_id


def _resolve_out_dir(out_dir: str | None, run_id: str) -> Path:
    """Resolve the local run directory, defaulting to a config-derived path."""
    if out_dir:
        return Path(out_dir)
    return Path("scratch") / "oct_runs" / run_id


def _run_config_path(out_path: Path) -> Path:
    """Return the run config metadata path."""
    return out_path / _STAGE_META_DIR / _RUN_CONFIG_FILENAME


def _stage_marker_path(out_path: Path, stage_name: str) -> Path:
    """Return the metadata path for a completed stage marker."""
    return out_path / _STAGE_META_DIR / "stages" / f"{stage_name}.json"


def _artifact_exists(path: Path, kind: str) -> bool:
    """Return whether a file or directory artifact is present and non-empty."""
    if kind == "file":
        return path.is_file() and path.stat().st_size > 0
    if kind == "dir":
        return path.is_dir() and any(path.iterdir())
    if kind == "hf_model":
        return _hf_model_artifact_exists(path)
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _hf_model_artifact_exists(path: Path) -> bool:
    """Return whether a local HuggingFace model checkpoint looks complete."""
    if not path.is_dir():
        return False
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index_data = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return False

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = set(weight_map.values())
    return all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shard_names)


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ensure_run_config(out_path: Path, run_id: str, config_hash: str, config_payload: dict) -> Path:
    """Write and validate run config metadata for this out dir."""
    config_path = _run_config_path(out_path)
    payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "config": config_payload,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Run directory {out_path} already contains a different OCT config.\n"
                f"Existing hash: {existing.get('config_hash')}\n"
                f"Current hash:  {config_hash}\n"
                "Use a different --out-dir or omit --out-dir to use the config-derived run dir."
            )
    else:
        _write_json(config_path, payload)
    return config_path


def _get_git_commit_hash() -> str | None:
    """Return the current git HEAD hash, or None if unavailable."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    digest = output.strip()
    return digest or None


def _build_run_info(config_payload: dict) -> dict:
    """Build a run_info dict with provenance metadata and config."""
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _get_git_commit_hash() or "unknown",
        "run_command": " ".join(sys.argv),
        **config_payload,
    }


def _get_hf_helpers() -> dict[str, object]:
    """Import HF helper functions lazily so local-only runs keep working."""
    try:
        from src_dev.utils.hf_hub import (
            check_exists_in_dataset_repo,
            download_from_dataset_repo,
            upload_file_to_dataset_repo,
            upload_folder_to_dataset_repo,
        )
    except Exception as exc:  # pragma: no cover - import error depends on env
        raise RuntimeError(
            "Hugging Face artifact sync was requested, but the helper stack is unavailable. "
            "Make sure the repo environment includes huggingface_hub and the src_dev utils."
        ) from exc

    return {
        "dataset_repo_subpath_exists": check_exists_in_dataset_repo,
        "download_dataset_subpath": download_from_dataset_repo,
        "download_file_from_dataset_repo": download_from_dataset_repo,
        "upload_file_to_dataset_repo": upload_file_to_dataset_repo,
        "upload_folder_to_dataset_repo": upload_folder_to_dataset_repo,
    }


def _remote_repo_path(prefix: str, relative_path: Path) -> str:
    """Map a local path under out_dir to its remote HF dataset-repo path."""
    return f"{prefix}/{relative_path.as_posix()}"


def _upload_artifact_to_hf(
    *,
    repo_id: str,
    relative_path: Path,
    local_path: Path,
    kind: str,
    prefix: str,
    commit_message: str,
) -> None:
    """Upload a stage artifact to the HF monorepo. Raises on failure."""
    helpers = _get_hf_helpers()
    path_in_repo = _remote_repo_path(prefix, relative_path)
    if kind == "file":
        helpers["upload_file_to_dataset_repo"](
            local_path=local_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
    else:
        helpers["upload_folder_to_dataset_repo"](
            local_dir=local_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )


def _stage_artifacts_ready(artifacts: list[dict]) -> bool:
    """Return whether all expected artifacts for a stage exist locally."""
    return all(_artifact_exists(item["path"], item["kind"]) for item in artifacts)


def _write_stage_marker(
    *,
    out_path: Path,
    stage_name: str,
    cache_key: str,
    artifacts: list[dict],
    extra_info: dict | None = None,
) -> Path:
    """Persist metadata describing a completed stage."""
    marker_path = _stage_marker_path(out_path, stage_name)
    payload = {
        "stage": stage_name,
        "cache_key": cache_key,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _get_git_commit_hash() or "unknown",
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {
                "relative_path": str(item["path"].relative_to(out_path)),
                "kind": item["kind"],
            }
            for item in artifacts
        ],
    }
    if extra_info:
        payload["info"] = extra_info
    _write_json(marker_path, payload)
    return marker_path


def _stage_is_cached_locally(
    *,
    out_path: Path,
    stage_name: str,
    cache_key: str,
    artifacts: list[dict],
) -> bool:
    """Return whether a stage is already complete locally."""
    marker_path = _stage_marker_path(out_path, stage_name)
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        # Accept either new cache_key or legacy config_hash field
        stored_key = marker.get("cache_key") or marker.get("config_hash")
        if stored_key != cache_key:
            return False
        if _stage_artifacts_ready(artifacts):
            return True

    if _stage_artifacts_ready(artifacts):
        _write_stage_marker(
            out_path=out_path,
            stage_name=stage_name,
            cache_key=cache_key,
            artifacts=artifacts,
        )
        return True
    return False


def _sync_monorepo_to_local(
    *,
    hf_repo_id: str,
    prefix: str,
    out_path: Path,
    cache_key: str,
) -> None:
    """Download the entire monorepo version prefix into out_path.

    After this call, all previously-uploaded stage markers and artifacts
    are available locally.  Legacy markers (using config_hash instead of
    cache_key) are migrated in-place.
    """
    helpers = _get_hf_helpers()
    print(f"\n  Syncing monorepo artifacts: {prefix}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        helpers["download_dataset_subpath"](
            repo_id=hf_repo_id,
            path_in_repo=prefix,
            local_dir=tmp_root,
        )
        # snapshot_download preserves the full repo path under local_dir
        downloaded_root = tmp_root / prefix
        if not downloaded_root.is_dir():
            print("  No remote artifacts found for this version")
            return

        # Copy downloaded files into out_path, skipping files that already exist
        for src_file in downloaded_root.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(downloaded_root)
            dest = out_path / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)

    # Migrate legacy stage markers (config_hash → cache_key)
    stages_dir = out_path / _STAGE_META_DIR / "stages"
    if stages_dir.is_dir():
        for marker_file in stages_dir.glob("*.json"):
            try:
                marker_data = json.loads(marker_file.read_text())
                if "config_hash" in marker_data and "cache_key" not in marker_data:
                    marker_data["cache_key"] = cache_key
                    del marker_data["config_hash"]
                    _write_json(marker_file, marker_data)
            except Exception:
                continue

    print("  Monorepo sync complete")


def _ensure_stage_available(
    *,
    out_path: Path,
    stage_name: str,
    cache_key: str,
    artifacts: list[dict],
) -> bool:
    """Check whether a stage's artifacts are present locally."""
    if _stage_is_cached_locally(
        out_path=out_path,
        stage_name=stage_name,
        cache_key=cache_key,
        artifacts=artifacts,
    ):
        print(f"  Reusing local {stage_name} artifacts")
        return True
    return False


def _publish_stage(
    *,
    out_path: Path,
    prefix: str,
    stage_name: str,
    cache_key: str,
    artifacts: list[dict],
    hf_repo_id: str,
) -> None:
    """Write stage metadata locally and upload to the monorepo."""
    marker_path = _write_stage_marker(
        out_path=out_path,
        stage_name=stage_name,
        cache_key=cache_key,
        artifacts=artifacts,
    )

    commit_message = f"OCT {stage_name}: {prefix}"
    marker_rel = marker_path.relative_to(out_path)
    _upload_artifact_to_hf(
        repo_id=hf_repo_id,
        relative_path=marker_rel,
        local_path=marker_path,
        kind="file",
        prefix=prefix,
        commit_message=commit_message,
    )
    for item in artifacts:
        # Artifacts may opt out of being uploaded to the monorepo (e.g. large
        # intermediates that are cheap to regenerate locally from other stages).
        # The stage marker is still uploaded, so remote cache bookkeeping is
        # unaffected; local runs will rebuild the artifact on demand when the
        # marker is present but the files are missing.
        if not item.get("upload", True):
            continue
        _upload_artifact_to_hf(
            repo_id=hf_repo_id,
            relative_path=item["path"].relative_to(out_path),
            local_path=item["path"],
            kind=item["kind"],
            prefix=prefix,
            commit_message=commit_message,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    model: str,
    constitution: str,
    out_dir: str | None,
    monorepo: MonorepoConfig,
    teacher_model: str = "z-ai/glm-4.5-air",
    teacher_prefill_mode: str = "oct",
    teacher_k: int | None = None,
    training_backend: str = "oct",
    stages: str = "all",
    skip_generation: bool = False,
    skip_training: bool = False,
    max_pairs: int | None = None,
    max_len: int = 1024,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    num_epochs: int = 1,
    n_reflection: int = 100,
    n_interaction: int = 100,
    interaction_turns: int = 10,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
    custom_constitution: str | None = None,
    introspection_constitution: str | None = None,
    expand_questions: bool = False,
    expand_model: str = "llama-3.3-70b-it",
    concat_all_traits_system_prompt: bool = False,
    seed: int = 123456,
    student_distillation_max_num_seqs: int | None = None,
    student_distillation_max_num_batched_tokens: int | None = None,
    student_distillation_enable_prefix_caching: bool | None = None,
    introspection_max_num_seqs: int | None = None,
    introspection_max_num_batched_tokens: int | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    torch_memory_fraction: float | None = None,
    oct_dpo_micro_batch_size: int | None = None,
    oct_sft_micro_batch_size: int | None = None,
    skip_student_distillation: bool = False,
) -> None:
    global _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE

    vllm_gpu_memory_utilization = _validate_unit_interval(
        "--vllm-gpu-memory-utilization",
        vllm_gpu_memory_utilization,
    )
    if (
        student_distillation_max_num_seqs is not None
        and student_distillation_max_num_seqs <= 0
    ):
        raise ValueError("--student-distillation-max-num-seqs must be > 0.")
    if (
        student_distillation_max_num_batched_tokens is not None
        and student_distillation_max_num_batched_tokens <= 0
    ):
        raise ValueError("--student-distillation-max-num-batched-tokens must be > 0.")
    if introspection_max_num_seqs is not None and introspection_max_num_seqs <= 0:
        raise ValueError("--introspection-max-num-seqs must be > 0.")
    if (
        introspection_max_num_batched_tokens is not None
        and introspection_max_num_batched_tokens <= 0
    ):
        raise ValueError("--introspection-max-num-batched-tokens must be > 0.")
    if oct_dpo_micro_batch_size is not None and oct_dpo_micro_batch_size <= 0:
        raise ValueError("--oct-dpo-micro-batch-size must be > 0.")
    if oct_sft_micro_batch_size is not None and oct_sft_micro_batch_size <= 0:
        raise ValueError("--oct-sft-micro-batch-size must be > 0.")
    _VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE = vllm_gpu_memory_utilization
    _apply_torch_memory_fraction(torch_memory_fraction)

    # Monorepo coordinates used for all remote artifact storage
    hf_repo_id = monorepo.repo_id
    monorepo_prefix = monorepo.path_prefix

    # Fail fast if HF upload is not possible — every stage must be uploaded
    # to prevent local-only runs from being overwritten on a later HF sync.
    try:
        helpers = _get_hf_helpers()
        helpers["dataset_repo_subpath_exists"](
            repo_id=hf_repo_id,
            path_in_repo=monorepo_prefix,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to HuggingFace monorepo ({hf_repo_id}). "
            "All pipeline runs must upload to the monorepo. "
            "Check HF_TOKEN and network connectivity."
        ) from exc
    print(f"  HF monorepo verified: {hf_repo_id}")

    config_payload, config_hash, run_id = _build_run_identity(
        model=model,
        constitution=constitution,
        teacher_model=teacher_model,
        teacher_prefill_mode=teacher_prefill_mode,
        teacher_k=teacher_k,
        training_backend=training_backend,
        max_pairs=max_pairs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        beta=beta,
        num_epochs=num_epochs,
        n_reflection=n_reflection,
        n_interaction=n_interaction,
        interaction_turns=interaction_turns,
        dpo_weight=dpo_weight,
        sft_weight=sft_weight,
        seed=seed,
        custom_constitution=custom_constitution,
        expand_questions=expand_questions,
        expand_model=expand_model,
        concat_all_traits_system_prompt=concat_all_traits_system_prompt,
        student_distillation_max_num_seqs=student_distillation_max_num_seqs,
        student_distillation_max_num_batched_tokens=student_distillation_max_num_batched_tokens,
        student_distillation_enable_prefix_caching=student_distillation_enable_prefix_caching,
        introspection_max_num_seqs=introspection_max_num_seqs,
        introspection_max_num_batched_tokens=introspection_max_num_batched_tokens,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        torch_memory_fraction=torch_memory_fraction,
        oct_dpo_micro_batch_size=oct_dpo_micro_batch_size,
        oct_sft_micro_batch_size=oct_sft_micro_batch_size,
    )
    out_path = _resolve_out_dir(out_dir, run_id)
    out_path.mkdir(parents=True, exist_ok=True)
    run_config_path = _ensure_run_config(out_path, run_id, config_hash, config_payload)

    # Save provenance metadata
    run_info = _build_run_info(
        {
            **config_payload,
            "student_distillation_max_num_seqs": student_distillation_max_num_seqs,
            "student_distillation_max_num_batched_tokens": student_distillation_max_num_batched_tokens,
            "student_distillation_enable_prefix_caching": student_distillation_enable_prefix_caching,
            "introspection_max_num_seqs": introspection_max_num_seqs,
            "introspection_max_num_batched_tokens": introspection_max_num_batched_tokens,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "torch_memory_fraction": torch_memory_fraction,
            "oct_dpo_micro_batch_size": oct_dpo_micro_batch_size,
            "oct_sft_micro_batch_size": oct_sft_micro_batch_size,
        }
    )
    run_info_path = out_path / "run_info.json"
    run_info_path.write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"  Provenance saved: {run_info_path}")

    # ── Download all existing monorepo artifacts for this version ──
    _sync_monorepo_to_local(
        hf_repo_id=hf_repo_id,
        prefix=monorepo_prefix,
        out_path=out_path,
        cache_key=monorepo_prefix,
    )

    # ── Redirect ALL OCT data into out_dir so nothing leaks to /workspace ──
    local_data_path = str(out_path / "data")
    local_lora_path = str(out_path / "lora")
    local_constitution_path = str(out_path / "constitutions")
    patch_oct_constants(
        data_path=local_data_path,
        lora_path=local_lora_path,
        constitution_path=local_constitution_path,
    )

    constitution_artifacts = [
        {"path": out_path / "constitutions" / "hand-written" / f"{constitution}.txt", "kind": "file"},
        {"path": out_path / "constitutions" / "few-shot" / f"{constitution}.jsonl", "kind": "file"},
        {"path": run_info_path, "kind": "file"},
    ]

    # Install custom constitution if provided
    if custom_constitution is not None:
        has_pending_expansion = _has_pending_openrouter_expansion_checkpoint(
            out_path=out_path,
            constitution=constitution,
            expand_questions=expand_questions,
            expand_model=expand_model,
        )
        if has_pending_expansion:
            print("  Found partial OpenRouter question expansion checkpoint; resuming incomplete traits")
            constitution_ready = False
        else:
            constitution_ready = _ensure_stage_available(
                out_path=out_path,
                stage_name="constitution",
                cache_key=monorepo_prefix,
                artifacts=constitution_artifacts,
            )
        if not constitution_ready:
            install_custom_constitution(
                name=constitution,
                source_path=custom_constitution,
                expand_questions=expand_questions,
                expand_model=expand_model,
            )
            _publish_stage(
                out_path=out_path,
                prefix=monorepo_prefix,
                stage_name="constitution",
                cache_key=monorepo_prefix,
                artifacts=constitution_artifacts,
                hf_repo_id=hf_repo_id,
            )

    model_path = _resolve_model_path(model)
    family = model.split("-")[0]

    do_distillation = stages in ("all", "distillation")
    do_introspection = stages in ("all", "introspection")
    do_merge = stages in ("all", "merge")

    dpo_adapter_path = out_path / "lora" / f"{constitution}-dpo"
    sft_adapter_path = out_path / "lora" / f"{constitution}-sft"
    persona_path = out_path / "lora" / f"{constitution}-persona"

    # OCT introspection expects adapters at LORA_PATH/{family}-distillation/{constitution}
    oct_lora_dir = Path(local_lora_path) / f"{family}-distillation" / constitution
    oct_lora_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"OCT PIPELINE")
    print(f"  model:        {model} ({model_path})")
    print(f"  teacher:      {teacher_model}")
    if _is_openrouter_model(teacher_model):
        print(f"  prefill:      {teacher_prefill_mode}")
    if teacher_k is not None:
        print(f"  teacher K:    {teacher_k}")
    print(f"  training:     {training_backend}")
    print(f"  run_id:       {run_id}")
    print(f"  seed:         {seed}")
    print(f"  constitution: {constitution}")
    print(f"  stages:       {stages}")
    print(f"  out_dir:      {out_path}")
    print(f"  data_path:    {local_data_path}")
    print(f"  monorepo:     {hf_repo_id} / {monorepo_prefix}")
    print(
        "  student distill vLLM: "
        f"max_num_seqs={student_distillation_max_num_seqs if student_distillation_max_num_seqs is not None else '(default)'} "
        f"max_num_batched_tokens={student_distillation_max_num_batched_tokens if student_distillation_max_num_batched_tokens is not None else '(default)'} "
        f"prefix_caching={student_distillation_enable_prefix_caching if student_distillation_enable_prefix_caching is not None else '(default)'}"
    )
    print(
        "  introspection vLLM: "
        f"max_num_seqs={introspection_max_num_seqs if introspection_max_num_seqs is not None else '(default)'} "
        f"max_num_batched_tokens={introspection_max_num_batched_tokens if introspection_max_num_batched_tokens is not None else '(default)'}"
    )
    print(f"  vllm cap:     {vllm_gpu_memory_utilization if vllm_gpu_memory_utilization is not None else '(default)'}")
    print(f"  torch cap:    {torch_memory_fraction if torch_memory_fraction is not None else '(default)'}")
    print(f"{'='*70}")

    # =====================================================================
    # STAGE 1-2: Distillation
    # =====================================================================
    if do_distillation:
        # Stage 1: Data generation
        distillation_path = Path(f"{local_data_path}/distillation/{constitution}.jsonl")
        distillation_generation_artifacts = [
            {"path": distillation_path, "kind": "file"},
        ]
        have_distillation = _ensure_stage_available(
            out_path=out_path,
            stage_name="distillation_generation",
            cache_key=monorepo_prefix,
            artifacts=distillation_generation_artifacts,
        )
        # Even if the stage marker exists, verify the file has both teacher
        # and student columns — a partial run may have only teacher responses.
        if have_distillation and distillation_path.exists():
            _cols = pd.read_json(str(distillation_path), orient="records", lines=True, nrows=1).columns
            if model not in _cols:
                print(f"\n  Distillation file exists but missing student column '{model}' "
                      f"(columns: {list(_cols)}). Re-running student pass.")
                have_distillation = False
        if have_distillation:
            print(f"\nUsing cached distillation data: {distillation_path}")
        elif skip_generation:
            raise FileNotFoundError(
                f"Distillation data not found locally or on Hugging Face for {distillation_path}. "
                "Remove --skip-generation to generate it."
            )
        else:
            run_distillation_generation(
                teacher_model=teacher_model,
                student_model=model,
                constitution=constitution,
                teacher_prefill_mode=teacher_prefill_mode,
                max_pairs=max_pairs,
                teacher_k=teacher_k,
                student_max_num_seqs=student_distillation_max_num_seqs,
                student_max_num_batched_tokens=student_distillation_max_num_batched_tokens,
                student_enable_prefix_caching=student_distillation_enable_prefix_caching,
                concat_all_traits_system_prompt=concat_all_traits_system_prompt,
                seed=seed,
                skip_student_distillation=skip_student_distillation,
            )
            _publish_stage(
                out_path=out_path,
                prefix=monorepo_prefix,
                stage_name="distillation_generation",
                cache_key=monorepo_prefix,
                artifacts=distillation_generation_artifacts,
                hf_repo_id=hf_repo_id,
            )

        # Paired-DPO seed flow: skip the rest of the distillation stage
        # (DPO-pair conversion, dpo_subset publish, DPO training). The
        # downstream paired-DPO seed step (prep_paired_dpo.py) replaces
        # the student column with the rejected teacher's responses and
        # re-publishes the distillation file under a different monorepo
        # prefix, so the dpo_subset / DPO training computed here would be
        # discarded anyway.
        #
        # CAREFUL: this is the second concern bundled into
        # --skip-student-distillation (see the argparse comment near line
        # 4051). It intentionally short-circuits DPO and everything below
        # it. Callers that have already seeded paired DPO data on HF and
        # want THIS invocation to train DPO must NOT pass the flag — they
        # rely on the cache-validation column-check above (line ~3614) to
        # skip the redundant student pass while letting DPO proceed.
        if skip_student_distillation:
            print(
                "\n  --skip-student-distillation set: skipping DPO-pair "
                "conversion / dpo_subset publish / DPO training. The "
                "downstream paired-DPO seed step will produce its own "
                "distillation file at a different monorepo prefix."
            )
            return

        # Convert to DPO pairs
        records = load_dpo_pairs(
            constitution=constitution,
            student_model=model,
            max_pairs=max_pairs,
        )
        _print_sample(records)

        # Save DPO subset for reference
        dpo_data_path = out_path / f"{constitution}_dpo.jsonl"
        with open(dpo_data_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  DPO subset: {dpo_data_path}")
        dpo_subset_artifacts = [{"path": dpo_data_path, "kind": "file"}]
        _publish_stage(
            out_path=out_path,
            prefix=monorepo_prefix,
            stage_name="dpo_subset",
            cache_key=monorepo_prefix,
            artifacts=dpo_subset_artifacts,
            hf_repo_id=hf_repo_id,
        )

        # Stage 2: DPO training
        if not skip_training:
            oct_dpo_dataset_path = out_path / "data" / "dpo" / model / f"{constitution}.jsonl"
            dpo_training_artifacts = [{"path": dpo_adapter_path, "kind": "dir"}]
            if training_backend == "oct":
                dpo_training_artifacts.append({"path": oct_dpo_dataset_path, "kind": "file"})

            have_dpo_adapter = _ensure_stage_available(
                out_path=out_path,
                stage_name="dpo_training",
                cache_key=monorepo_prefix,
                artifacts=dpo_training_artifacts,
            )
            if have_dpo_adapter:
                print(f"  Reusing cached DPO adapter: {dpo_adapter_path}")
            elif training_backend == "oct":
                run_oct_dpo_training(
                    model=model,
                    model_name_or_path=model_path,
                    constitution=constitution,
                    save_path=dpo_adapter_path,
                    seed=seed,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    beta=beta,
                    max_len=max_len,
                    max_pairs=max_pairs,
                    micro_batch_size=oct_dpo_micro_batch_size,
                )
                _publish_stage(
                    out_path=out_path,
                    prefix=monorepo_prefix,
                    stage_name="dpo_training",
                    cache_key=monorepo_prefix,
                    artifacts=dpo_training_artifacts,
                    hf_repo_id=hf_repo_id,
                )
            else:
                run_dpo_training(
                    model_name_or_path=model_path,
                    records=records,
                    save_path=dpo_adapter_path,
                    seed=seed,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    beta=beta,
                    max_len=max_len,
                )
                _publish_stage(
                    out_path=out_path,
                    prefix=monorepo_prefix,
                    stage_name="dpo_training",
                    cache_key=monorepo_prefix,
                    artifacts=dpo_training_artifacts,
                    hf_repo_id=hf_repo_id,
                )

            # OCT's introspection code looks for the adapter at
            # LORA_PATH/{family}-distillation/{constitution}/ (its internal
            # naming convention). The pipeline uses lora/{name}-dpo/ as the
            # canonical path. Symlink the OCT path to the canonical one so
            # both conventions resolve to the same adapter on disk.
            if not oct_lora_dir.exists():
                oct_lora_dir.symlink_to(dpo_adapter_path.resolve())
                print(f"  Symlinked {oct_lora_dir} -> {dpo_adapter_path}")

        # Free GPU memory from DPO training before introspection
        gc.collect()
        torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 3-4: Introspection
    # =====================================================================
    if do_introspection and introspection_constitution is not None:
        # Install a separate (typically shorter) constitution for introspection.
        # The distillation/DPO constitution may be too long for the introspection
        # system prompt (e.g. it includes OCAN anchoring text for the teacher model
        # that isn't needed post-DPO and would exceed the context window).
        print(f"\n  Installing introspection constitution from: "
              f"{introspection_constitution}")
        install_custom_constitution(
            name=constitution,
            source_path=introspection_constitution,
            skip_question_validation=True,
        )

    if do_introspection:
        cached_dpo_artifacts = [{"path": dpo_adapter_path, "kind": "dir"}]
        _ensure_stage_available(
            out_path=out_path,
            stage_name="dpo_training",
            cache_key=monorepo_prefix,
            artifacts=cached_dpo_artifacts,
        )
        if not oct_lora_dir.exists() and not dpo_adapter_path.exists():
            raise FileNotFoundError(
                f"DPO adapter required for introspection. "
                f"Expected at {oct_lora_dir} or {dpo_adapter_path}. "
                "Run distillation stage first."
            )

        # Ensure symlink exists
        if not oct_lora_dir.exists() and dpo_adapter_path.exists():
            oct_lora_dir.symlink_to(dpo_adapter_path.resolve())

        # Stage 3: Introspection data generation
        sft_data_path = Path(f"{local_data_path}/sft_data/{model}/{constitution}.jsonl")
        introspection_generation_artifacts = [
            {"path": Path(f"{local_data_path}/self_reflection/{model}/{constitution}.jsonl"), "kind": "file"},
            {"path": Path(f"{local_data_path}/self_interaction/{model}/{constitution}.jsonl"), "kind": "file"},
            {"path": Path(f"{local_data_path}/self_interaction/{model}/{constitution}-leading.jsonl"), "kind": "file"},
            {"path": sft_data_path, "kind": "file"},
        ]
        have_introspection = _ensure_stage_available(
            out_path=out_path,
            stage_name="introspection_generation",
            cache_key=monorepo_prefix,
            artifacts=introspection_generation_artifacts,
        )
        if have_introspection:
            print(f"  Reusing cached introspection data: {sft_data_path}")
        elif skip_generation:
            raise FileNotFoundError(
                f"SFT data not found locally or on Hugging Face for {sft_data_path}. "
                "Remove --skip-generation to generate it."
            )
        else:
            sft_data_path = run_introspection_generation(
                model=model,
                constitution=constitution,
                n_reflection=n_reflection,
                n_interaction=n_interaction,
                interaction_turns=interaction_turns,
                max_num_seqs=introspection_max_num_seqs,
                max_num_batched_tokens=introspection_max_num_batched_tokens,
            )
            _publish_stage(
                out_path=out_path,
                prefix=monorepo_prefix,
                stage_name="introspection_generation",
                cache_key=monorepo_prefix,
                artifacts=introspection_generation_artifacts,
                hf_repo_id=hf_repo_id,
            )

        # Stage 4: SFT training
        if not skip_training:
            if training_backend == "oct":
                distilled_model_path = out_path / "models" / "distilled" / f"{model}-{constitution}"
                # The folded (base + DPO LoRA) checkpoint is a large (~16GB for 8B)
                # intermediate that is cheap to regenerate from the base model and
                # the already-uploaded DPO adapter, so we keep it local-only.
                distilled_model_artifacts = [
                    {"path": distilled_model_path, "kind": "hf_model", "upload": False}
                ]
                have_distilled_model = _ensure_stage_available(
                    out_path=out_path,
                    stage_name="distilled_model",
                    cache_key=monorepo_prefix,
                    artifacts=distilled_model_artifacts,
                )
                if have_distilled_model:
                    print(f"  Reusing cached folded model: {distilled_model_path}")
                else:
                    distilled_model_path = fold_lora_into_model(
                        base_model_path=model_path,
                        lora_path=dpo_adapter_path,
                        output_path=distilled_model_path,
                    )
                    _publish_stage(
                        out_path=out_path,
                        prefix=monorepo_prefix,
                        stage_name="distilled_model",
                        cache_key=monorepo_prefix,
                        artifacts=distilled_model_artifacts,
                        hf_repo_id=hf_repo_id,
                    )

                sft_training_artifacts = [{"path": sft_adapter_path, "kind": "dir"}]
                have_sft_adapter = _ensure_stage_available(
                    out_path=out_path,
                    stage_name="sft_training",
                    cache_key=monorepo_prefix,
                    artifacts=sft_training_artifacts,
                )
                if have_sft_adapter:
                    print(f"  Reusing cached SFT adapter: {sft_adapter_path}")
                else:
                    run_oct_sft_training(
                        model=model,
                        distilled_model_path=distilled_model_path,
                        sft_data_path=sft_data_path,
                        save_path=sft_adapter_path,
                        seed=seed,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        micro_batch_size=oct_sft_micro_batch_size,
                    )
                    _publish_stage(
                        out_path=out_path,
                        prefix=monorepo_prefix,
                        stage_name="sft_training",
                        cache_key=monorepo_prefix,
                        artifacts=sft_training_artifacts,
                        hf_repo_id=hf_repo_id,
                    )
            else:
                sft_training_artifacts = [{"path": sft_adapter_path, "kind": "dir"}]
                have_sft_adapter = _ensure_stage_available(
                    out_path=out_path,
                    stage_name="sft_training",
                    cache_key=monorepo_prefix,
                    artifacts=sft_training_artifacts,
                )
                if have_sft_adapter:
                    print(f"  Reusing cached SFT adapter: {sft_adapter_path}")
                else:
                    run_sft_training(
                        model_name_or_path=model_path,
                        sft_data_path=sft_data_path,
                        save_path=sft_adapter_path,
                        seed=seed,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                    )
                    _publish_stage(
                        out_path=out_path,
                        prefix=monorepo_prefix,
                        stage_name="sft_training",
                        cache_key=monorepo_prefix,
                        artifacts=sft_training_artifacts,
                        hf_repo_id=hf_repo_id,
                    )

    # =====================================================================
    # STAGE 5: Adapter merge
    # =====================================================================
    if do_merge:
        _ensure_stage_available(
            out_path=out_path,
            stage_name="dpo_training",
            cache_key=monorepo_prefix,
            artifacts=[{"path": dpo_adapter_path, "kind": "dir"}],
        )
        _ensure_stage_available(
            out_path=out_path,
            stage_name="sft_training",
            cache_key=monorepo_prefix,
            artifacts=[{"path": sft_adapter_path, "kind": "dir"}],
        )
        merge_artifacts = [{"path": persona_path, "kind": "dir"}]
        have_persona = _ensure_stage_available(
            out_path=out_path,
            stage_name="merge",
            cache_key=monorepo_prefix,
            artifacts=merge_artifacts,
        )
        if have_persona:
            print(f"  Reusing cached merged persona adapter: {persona_path}")
        else:
            if not dpo_adapter_path.exists():
                raise FileNotFoundError(f"DPO adapter not found at {dpo_adapter_path}")
            if not sft_adapter_path.exists():
                raise FileNotFoundError(f"SFT adapter not found at {sft_adapter_path}")

            merge_adapters(
                base_model_path=model_path,
                dpo_adapter_path=dpo_adapter_path,
                sft_adapter_path=sft_adapter_path,
                save_path=persona_path,
                dpo_weight=dpo_weight,
                sft_weight=sft_weight,
            )
            _publish_stage(
                out_path=out_path,
                prefix=monorepo_prefix,
                stage_name="merge",
                cache_key=monorepo_prefix,
                artifacts=merge_artifacts,
                hf_repo_id=hf_repo_id,
            )

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"  Constitution:  {constitution}")
    if do_distillation:
        print(f"  DPO data:      {local_data_path}/distillation/{constitution}.jsonl")
        print(f"  DPO adapter:   {dpo_adapter_path}")
    if do_introspection:
        print(f"  SFT data:      {local_data_path}/sft_data/{model}/{constitution}.jsonl")
        print(f"  SFT adapter:   {sft_adapter_path}")
    if do_merge:
        print(f"  Persona:       {persona_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full OCT persona training pipeline: distillation + introspection + merge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--model", default="qwen-2.5-1.5b-it",
                        help="Student model folder name under MODEL_PATH")
    parser.add_argument("--teacher-model", default="z-ai/glm-4.5-air",
                        help="Teacher model: local name (vLLM) or org/model (OpenRouter API)")
    parser.add_argument(
        "--teacher-prefill-mode",
        default="oct",
        choices=["oct", "none"],
        help=(
            "OpenRouter teacher assistant-prefill mode. "
            "'oct' mirrors upstream OCT's hidden think prefill; 'none' disables it."
        ),
    )
    parser.add_argument(
        "--teacher-k", "--K",
        dest="teacher_k",
        type=int,
        default=None,
        help=(
            "Repeat each distillation prompt K times before teacher generation "
            "(matches upstream OCT teacher.py semantics)."
        ),
    )
    parser.add_argument("--constitution", default=None,
                        help="Constitution name (default: stem of --custom-constitution if provided, else 'sarcasm')")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for adapters and data subsets (default: config-derived scratch/oct_runs/<run_id>)")
    parser.add_argument("--training-backend", default="oct", choices=["oct", "trl"],
                        help="Training backend: native OCT/OpenRLHF or the local TRL fallback")
    parser.add_argument("--seed", type=int, default=123456,
                        help="Random seed used for training and run identity")

    # Stage control
    parser.add_argument("--stages", default="all", choices=sorted(STAGES),
                        help="Which stages to run")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Reuse-only for data generation stages: use local/monorepo artifacts if available, otherwise error")
    parser.add_argument("--skip-training", action="store_true",
                        help="Generate data only, skip all training")
    # CAVEAT: this flag bundles two independent concerns and is misleadingly
    # named. It does:
    #   (a) Skip the local student vLLM baseline pass during distillation
    #       (the literal interpretation of the name).
    #   (b) ALSO skip DPO-pair conversion + dpo_subset publish + DPO training.
    # The bundling is intentional for the original use case: producing teacher-
    # only distillation data for a downstream paired-DPO seed flow to consume
    # at a different monorepo prefix. In that flow, running DPO here on
    # chosen=teacher / rejected=(empty) would be wasted compute because the
    # real DPO is trained downstream against the paired teacher data.
    #
    # If you have already seeded paired DPO data on HF (e.g. via
    # scripts_dev/oct_pipeline/ocean/seed_*_paired_dpo.sh) and you want this
    # invocation to TRAIN DPO on it, do NOT pass --skip-student-distillation —
    # passing it would silently disable DPO training. Instead, ensure the
    # seeded paired JSONL has its rejected column named after the student
    # model (--rejected-col on prep_paired_dpo.py); then the cache-validation
    # column-check below will pass, no student pass is re-triggered, and DPO
    # training proceeds normally on the cached paired data.
    #
    # TODO(future): unbundle into two flags
    #   --skip-student-pass  (just skip the student vLLM call)
    #   --teacher-only       (also skip DPO conversion + training)
    # to make the API match its naming.
    parser.add_argument("--skip-student-distillation", action="store_true",
                        help=(
                            "Skip the local Llama student-baseline distillation pass. "
                            "Used by paired-teacher DPO seed flows where the student "
                            "column is overwritten by the rejected teacher's response. "
                            "Implies skipping DPO-pair conversion, dpo_subset publish, "
                            "and DPO training (the downstream paired-DPO seed step "
                            "publishes its own distillation file). Do NOT pass when "
                            "you want DPO to train on already-seeded paired data — "
                            "rely on the cache-validation column-check instead."
                        ))

    # Data
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max DPO pairs to use (None = all)")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max sequence length for DPO pair filtering (pairs longer than this are dropped)")

    # LoRA / training
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Optional upper bound for vLLM gpu_memory_utilization across OCT vLLM "
            "loads (teacher/student/introspection). Useful when soft-sharing one GPU "
            "between multiple runs."
        ),
    )
    parser.add_argument(
        "--torch-memory-fraction",
        type=float,
        default=None,
        help=(
            "Optional PyTorch per-process allocator fraction for in-process HF stages. "
            "Does not constrain external OpenRLHF/deepspeed subprocesses."
        ),
    )
    parser.add_argument(
        "--oct-dpo-micro-batch-size",
        type=int,
        default=None,
        help="Optional OpenRLHF DPO micro-batch override for lower VRAM usage.",
    )
    parser.add_argument(
        "--oct-sft-micro-batch-size",
        type=int,
        default=None,
        help="Optional OpenRLHF SFT micro-batch override for lower VRAM usage.",
    )
    parser.add_argument(
        "--student-distillation-max-num-seqs",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_num_seqs override for the local student distillation pass only. "
            "When unset, upstream OCT keeps its default of 64."
        ),
    )
    parser.add_argument(
        "--student-distillation-max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_num_batched_tokens override for the local student distillation pass only. "
            "When unset, upstream OCT keeps its default of 32768."
        ),
    )
    parser.add_argument(
        "--student-distillation-enable-prefix-caching",
        action="store_true",
        help=(
            "Enable vLLM prefix caching for the local student distillation pass. "
            "By default the wrapper preserves its previous behavior of disabling it."
        ),
    )
    parser.add_argument(
        "--introspection-max-num-seqs",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_num_seqs override for introspection only. "
            "When unset, upstream OCT keeps its default of 1024."
        ),
    )
    parser.add_argument(
        "--introspection-max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Optional vLLM max_num_batched_tokens override for introspection only. "
            "When unset, upstream OCT keeps its default of 32768."
        ),
    )
    parser.add_argument(
        "--introspection-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional cap on per-response generation length for introspection "
            "(self-reflection / self-interaction). Upstream OCT hardcodes 2048 "
            "(reflection) / 1024 (interaction), which is wastefully long for "
            "conversational persona data and dominates introspection wall-clock. "
            "e.g. 512. When unset, upstream defaults are kept."
        ),
    )

    # Introspection
    parser.add_argument("--n-reflection", type=int, default=1000,
                        help="Repeats per reflection prompt")
    parser.add_argument("--n-interaction", type=int, default=2000,
                        help="Number of self-interaction conversations")
    parser.add_argument("--interaction-turns", type=int, default=10,
                        help="Turns per self-interaction conversation")

    # Merge weights
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--sft-weight", type=float, default=0.25)

    # Custom constitution
    parser.add_argument("--custom-constitution", default=None,
                        help="Path to a JSON file defining custom traits (see docstring)")
    parser.add_argument("--introspection-constitution", default=None,
                        help="Optional path to a shorter constitution JSON for introspection/SFT stages "
                             "(installed under the same --constitution name, overriding the distillation constitution)")
    parser.add_argument("--expand-questions", action="store_true",
                        help="Expand each trait to 50 questions; local models use OCT/vLLM, org/model ids use OpenRouter")
    parser.add_argument("--expand-model", default="llama-3.3-70b-it",
                        help="Model for question expansion (local vLLM name or OpenRouter org/model id)")
    parser.add_argument("--concat-all-traits-system-prompt", action="store_true", default=False,
                        help="(OpenRouter teacher only) Build a single shared teacher system prompt by "
                             "concatenating all 12 facet `trait` strings (legacy behavior, pre-vanton4). "
                             "By default (flag absent), each curated question uses ONLY its originating "
                             "facet's `trait`, and each LIMA/factual question picks one of the facet "
                             "traits at random (seeded via --seed).")

    # Path override (only MODEL_PATH may need overriding; data/lora/constitutions go to out-dir)
    parser.add_argument("--model-path", default=None,
                        help="Override MODEL_PATH (where base models live)")

    # Monorepo (required — all artifacts are stored in the structured monorepo)
    parser.add_argument("--monorepo-category", required=True,
                        choices=["ocean", "toy", "unsupervised", "other"],
                        help="Category for monorepo path")
    parser.add_argument("--monorepo-trait", required=True,
                        help="Trait name for monorepo path, e.g. 'extraversion'")
    parser.add_argument("--monorepo-direction", required=True,
                        choices=["amplifier", "suppressor"],
                        help="Whether this run amplifies or suppresses the trait")
    parser.add_argument("--monorepo-version", type=str, required=True,
                        help="Version number N for v{N} in the monorepo path")

    args = parser.parse_args()

    if args.teacher_k is not None and args.teacher_k < 1:
        parser.error("--teacher-k/--K must be >= 1")

    # Infer --constitution from --custom-constitution if not provided
    if args.constitution is None:
        if args.custom_constitution is not None:
            args.constitution = Path(args.custom_constitution).stem
        else:
            args.constitution = "sarcasm"

    # Apply model path override if provided
    if args.model_path:
        patch_oct_constants(model_path=args.model_path)

    # Configure the introspection generation-length cap (read at SamplingParams
    # construction time by the wrapper installed above). This block runs at
    # module scope (under __main__), so a plain assignment rebinds the module
    # global directly — no `global` keyword needed (and it would be illegal on
    # the annotated module-level name).
    if args.introspection_max_new_tokens is not None:
        _INTROSPECTION_MAX_NEW_TOKENS_OVERRIDE = args.introspection_max_new_tokens
        print(
            "  Introspection max_new_tokens capped to "
            f"{args.introspection_max_new_tokens} (upstream defaults 2048/1024)"
        )

    # Warn if there are uncommitted changes
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if status:
            print(f"\n{'!'*70}")
            print("  WARNING: You have uncommitted changes in your working tree.")
            print("  The git hash recorded in run_info / stage markers may not")
            print("  reflect the code that actually ran this pipeline.")
            print(f"{'!'*70}")
            print(f"\n{status}\n")
            answer = input("  Continue anyway? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("  Aborted.")
                sys.exit(1)
    except Exception:
        pass

    monorepo = MonorepoConfig(
        repo_id=_MONOREPO_REPO_ID,
        model=args.model,
        category=args.monorepo_category,
        trait=args.monorepo_trait,
        direction=args.monorepo_direction,
        version=args.monorepo_version,
    )

    main(
        model=args.model,
        constitution=args.constitution,
        out_dir=args.out_dir,
        monorepo=monorepo,
        teacher_model=args.teacher_model,
        teacher_prefill_mode=args.teacher_prefill_mode,
        teacher_k=args.teacher_k,
        training_backend=args.training_backend,
        stages=args.stages,
        skip_generation=args.skip_generation,
        skip_training=args.skip_training,
        skip_student_distillation=args.skip_student_distillation,
        max_pairs=args.max_pairs,
        max_len=args.max_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_epochs=args.num_epochs,
        n_reflection=args.n_reflection,
        n_interaction=args.n_interaction,
        interaction_turns=args.interaction_turns,
        dpo_weight=args.dpo_weight,
        sft_weight=args.sft_weight,
        custom_constitution=args.custom_constitution,
        introspection_constitution=args.introspection_constitution,
        expand_questions=args.expand_questions,
        expand_model=args.expand_model,
        concat_all_traits_system_prompt=args.concat_all_traits_system_prompt,
        seed=args.seed,
        student_distillation_max_num_seqs=args.student_distillation_max_num_seqs,
        student_distillation_max_num_batched_tokens=args.student_distillation_max_num_batched_tokens,
        student_distillation_enable_prefix_caching=(
            True if args.student_distillation_enable_prefix_caching else None
        ),
        introspection_max_num_seqs=args.introspection_max_num_seqs,
        introspection_max_num_batched_tokens=args.introspection_max_num_batched_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        torch_memory_fraction=args.torch_memory_fraction,
        oct_dpo_micro_batch_size=args.oct_dpo_micro_batch_size,
        oct_sft_micro_batch_size=args.oct_sft_micro_batch_size,
    )

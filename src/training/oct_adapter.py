"""The OCT adapter seam: the single module the paired-DPO scripts import.

This is the boundary between the ``scripts/training/ocean_paired_dpo`` run
surface and the external OpenCharacterTraining (``character.*`` / ``openrlhf``)
stack plus the local vLLM runtime patches. Scripts call only the public stage
functions exported here; they never touch ``character.*`` / ``openrlhf.*``
directly.

Public stages (one per pipeline step):

* ``initialize_oct_runtime`` — import the runtime patches and point OCT's
  constants at the run directory (call once, first).
* ``install_constitution`` — install a trait constitution into OCT's format,
  optionally expanding questions via OpenRouter.
* ``generate_distillation_data`` — teacher (+ optionally student) distillation
  pass; canonical paired-DPO flow is teacher-only.
* ``train_dpo_adapter`` — DPO-train a LoRA via OCT's native OpenRLHF stack.
* ``generate_introspection_data`` — self-reflection + self-interaction data.
* ``fold_dpo_lora_into_model`` — fold the DPO LoRA into a full checkpoint.
* ``train_sft_adapter`` — SFT-train a second LoRA on introspection data.
* ``merge_adapters_into_persona`` — linearly combine DPO + SFT into the final
  persona adapter.
* ``symlink_oct_dpo_dir`` — bridge OCT's internal ``{family}-distillation``
  adapter path to the canonical ``{name}-dpo`` path.

Scope: this seam covers ONLY the canonical OCT-backend paired-DPO flow. The
older TRL-backend training functions and the unpaired ``load_dpo_pairs`` /
sample-printing helpers from the dev monolith are intentionally NOT migrated —
the canonical path never calls them.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.oct_config import (
    _MODEL_HF_REPO_IDS,
    _OCT_TRAINING_CONFIGS,
)
from src.training.openrouter_teacher import (
    _is_openrouter_model,
    _teacher_assistant_name,
    _write_openrouter_expanded_constitution,
    run_teacher_openrouter,
    write_fewshot_constitution_jsonl,
)


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------


def _require_module(name: str) -> None:
    """Raise a helpful error if a Python dependency is unavailable."""
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"Required Python module '{name}' is not installed. "
            "Run through uv with the OCT requirements layered in, e.g. "
            "`uv run --isolated --with-requirements "
            "scripts/training/ocean_paired_dpo/uv-oct-requirements.txt "
            "python scripts/training/ocean_paired_dpo/04_train_lora.py ...`."
        )


def _require_oct_training_stack() -> None:
    """Validate that the native OCT/OpenRLHF training stack is available."""
    _require_module("openrlhf")
    if shutil.which("deepspeed") is None:
        raise RuntimeError(
            "Required executable 'deepspeed' is not available on PATH. "
            "Install the OCT training stack before using the OCT training backend."
        )


def _oct_training_config_for_model(model: str) -> dict:
    """Return native OCT/OpenRLHF training defaults for a supported model."""
    if model not in _OCT_TRAINING_CONFIGS:
        supported = ", ".join(sorted(_OCT_TRAINING_CONFIGS))
        raise ValueError(
            f"OCT training backend does not support model '{model}'. "
            f"Supported models: {supported}"
        )
    return _OCT_TRAINING_CONFIGS[model]


# ---------------------------------------------------------------------------
# GPU / model-path helpers
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


def _current_model_path() -> str:
    """Return the current OCT MODEL_PATH after any runtime patching."""
    import character.constants as _cc

    return _cc.MODEL_PATH


def _resolve_model_path(model: str) -> str:
    """Return the full filesystem path for a model name, downloading from HF if needed."""
    model_path_root = _current_model_path()
    full = f"{model_path_root}/{model}"
    if _has_model_weights(full):
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

    print(f"\n{'=' * 70}")
    print(f"  Model '{model}' not found locally at {full}")
    print(f"  Downloading from HuggingFace: {hf_repo_id}")
    print(f"{'=' * 70}\n")

    from huggingface_hub import snapshot_download

    snapshot_download(
        hf_repo_id,
        local_dir=full,
        ignore_patterns=["original/*", "*.pth", "*.gguf"],
        token=os.environ.get("HF_TOKEN"),
    )
    return full


# ---------------------------------------------------------------------------
# Stage 0: runtime initialization
# ---------------------------------------------------------------------------


def initialize_oct_runtime(
    *,
    model_path: str | None = None,
    data_path: str | None = None,
    lora_path: str | None = None,
    constitution_path: str | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    torch_memory_fraction: float | None = None,
) -> None:
    """Import the OCT runtime patches and point OCT at the run directory.

    Importing ``src.training.oct_runtime`` triggers the import-time vLLM /
    huggingface_hub / ``character.constants`` patching. This wrapper then
    applies the path overrides (so all OCT reads/writes stay inside the run
    directory) and the optional GPU-memory caps. Call this once, before any
    other stage. Combines the runtime-import + ``patch_oct_constants`` +
    memory-cap setup that ``main()`` did inline in the dev monolith.
    """
    from src.training import oct_runtime

    if vllm_gpu_memory_utilization is not None:
        vllm_gpu_memory_utilization = oct_runtime._validate_unit_interval(
            "--vllm-gpu-memory-utilization", vllm_gpu_memory_utilization
        )
        oct_runtime._VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE = vllm_gpu_memory_utilization

    oct_runtime.patch_oct_constants(
        data_path=data_path,
        model_path=model_path,
        lora_path=lora_path,
        constitution_path=constitution_path,
    )
    oct_runtime._apply_torch_memory_fraction(torch_memory_fraction)


# ---------------------------------------------------------------------------
# Stage 0b: constitution install
# ---------------------------------------------------------------------------


def install_constitution(
    name: str,
    source_path: str,
    *,
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
        name: Constitution name (used as filename and constitution value).
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
        raise ValueError(
            "Constitution must be a non-empty JSON array of trait objects."
        )
    for i, t in enumerate(traits):
        if "trait" not in t:
            raise ValueError(f"Trait {i} missing 'trait' key.")
        if t.get("is_lima_fallback", False):
            raise ValueError(
                f"Entry {i} in {source} has is_lima_fallback=true, which is no "
                f"longer supported (vanton4+). Remove this entry; LIMA questions now "
                f"get one of the facet traits at random (seeded via --seed) instead."
            )
        if not skip_question_validation and (
            "questions" not in t or len(t["questions"]) < 5
        ):
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
                f"(target=50 questions/trait)..."
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
        write_fewshot_constitution_jsonl(fs_path, traits)
        print(f"  Wrote few-shot constitution: {fs_path}")
        print(
            f"  (question expansion skipped — using {sum(len(t['questions']) for t in traits)} hand-written questions)"
        )


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
            repo_id="GAIR/lima",
            filename=f"{split}.jsonl",
            repo_type="dataset",
            token=token,
        )
        rows = [json.loads(line) for line in open(src)]
        with open(lima_dir / f"{split}.jsonl", "w") as f:
            for row in rows:
                json.dump({"conversations": row["conversations"]}, f)
                f.write("\n")
        print(f"  LIMA {split}: {len(rows)} rows → {lima_dir}/{split}.jsonl")


# ---------------------------------------------------------------------------
# Stage 1: distillation data generation
# ---------------------------------------------------------------------------


def generate_distillation_data(
    *,
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
    skip_student_pass: bool = False,
) -> Path:
    """Generate teacher (chosen) and student (rejected) responses.

    Calls the OpenRouter teacher path (or OCT's local ``teacher.main()``) and,
    unless skipped, OCT's ``student`` baseline pass. Output is written to
    ``DATA_PATH/distillation/{constitution}.jsonl``.

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
        skip_student_pass: When True, skip the student baseline pass entirely
            and emit a JSONL with only ``prompt`` + ``response`` columns. Used by
            paired-teacher DPO seed flows where the student baseline is unused
            (the ``llama-3.1-8b-it`` column slot is overwritten by
            ``03_build_paired_dataset.py`` with the rejected teacher's response).
            This is the canonical paired-DPO teacher-only flag.

    Returns:
        Path to the distillation JSONL file.
    """
    from src.training import oct_runtime

    import character.constants as _cc

    ensure_lima(_cc.MODEL_PATH)
    distillation_path = Path(f"{_cc.DATA_PATH}/distillation/{constitution}.jsonl")

    # Skip teacher pass if teacher responses already exist
    teacher_exists = False
    if distillation_path.exists():
        existing = pd.read_json(
            str(distillation_path), orient="records", lines=True, nrows=1
        )
        if "response" in existing.columns and len(existing) > 0:
            total = len(
                pd.read_json(str(distillation_path), orient="records", lines=True)
            )
            print(f"\n--- Teacher pass (model={teacher_model}) ---")
            print(
                f"  Skipping: {total} teacher responses already exist in {distillation_path}"
            )
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
                    "prompt. Re-run with concat_all_traits_system_prompt=True to use the local "
                    f"teacher. Got teacher_model={teacher_model!r}."
                )
            oct_runtime.oct_teacher.main(
                model=teacher_model, constitution=constitution, K=teacher_k
            )
            # Force cleanup of vLLM GPU memory before loading student
            gc.collect()
            torch.cuda.empty_cache()

    if skip_student_pass:
        print("\n--- Student pass: SKIPPED (skip_student_pass) ---")
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
    if oct_runtime._VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE is not None:
        gpu_util = min(gpu_util, oct_runtime._VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE)
    print(
        f"  GPU free: {free_gib:.1f}/{total_gib:.1f} GiB → using gpu_memory_utilization={gpu_util:.2f}"
    )
    import character.distillation.student as _oct_student_mod

    previous_max_num_seqs = oct_runtime._STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE
    previous_max_num_batched_tokens = (
        oct_runtime._STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    )
    previous_enable_prefix_caching = (
        oct_runtime._STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE
    )
    oct_runtime._STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE = student_max_num_seqs
    oct_runtime._STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = (
        student_max_num_batched_tokens
    )
    oct_runtime._STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE = (
        student_enable_prefix_caching
    )
    try:
        with oct_runtime._vllm_stage_context("student_distillation"):
            args, llm, tokenizer = _oct_student_mod.load_vllm(
                student_model,
                enable_prefix_caching=False,
                gpu_memory_utilization=gpu_util,
            )
    finally:
        oct_runtime._STUDENT_DISTILLATION_MAX_NUM_SEQS_OVERRIDE = previous_max_num_seqs
        oct_runtime._STUDENT_DISTILLATION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = (
            previous_max_num_batched_tokens
        )
        oct_runtime._STUDENT_DISTILLATION_ENABLE_PREFIX_CACHING_OVERRIDE = (
            previous_enable_prefix_caching
        )
    distillation_file = str(distillation_path)
    _oct_student_mod.no_roleplay(
        distillation_file, args, llm, tokenizer, constitution, student_model
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Distillation data: {distillation_path}")
    return distillation_path


# ---------------------------------------------------------------------------
# Stage 1→2 bridge: format OCT distillation output for OpenRLHF DPO
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
    teacher_model: str | None = None,
) -> Path:
    """Format distillation output into OCT/OpenRLHF DPO JSONL for one run.

    Args:
        teacher_model: Teacher model id, used to scrub the teacher's self-name
            (e.g. ``"ChatGLM"`` for GLM) out of *both* DPO branches. When
            ``None`` we assume the canonical GLM teacher (``"ChatGLM"``).
    """
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    name = _teacher_assistant_name(student_model)
    # Teacher self-reference to scrub from BOTH DPO branches, so the student
    # never learns to refer to itself by the teacher's name.
    teacher_name = (
        _teacher_assistant_name(teacher_model) if teacher_model else "ChatGLM"
    )

    responses["teacher_missing"] = ~responses["response"].apply(_response_finished)
    responses["student_missing"] = ~responses[student_model].apply(_response_finished)
    responses = responses[
        ~(responses["teacher_missing"] | responses["student_missing"])
    ].copy()

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {
                "role": "assistant",
                "content": row["response"].replace(teacher_name, name),
            },
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {
                "role": "assistant",
                "content": row[student_model].replace(teacher_name, name),
            },
        ],
        axis=1,
    )

    data["c_prompt"] = data["chosen"].apply(
        lambda x: tokenizer.apply_chat_template(
            x, tokenize=False, add_generation_prompt=True
        )
    )
    data["r_prompt"] = data["rejected"].apply(
        lambda x: tokenizer.apply_chat_template(
            x, tokenize=False, add_generation_prompt=True
        )
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


# ---------------------------------------------------------------------------
# OpenRLHF training plumbing (shared by DPO + SFT)
# ---------------------------------------------------------------------------


def _run_openrlhf_training(command: list[str], stage_name: str) -> None:
    """Run an OpenRLHF/DeepSpeed training command."""
    import subprocess

    print(f"  Launching {stage_name} via OpenRLHF:")
    print(f"    {' '.join(command)}")
    env = os.environ.copy()

    if not env.get("CUDA_HOME") and shutil.which("nvcc"):
        nvcc_path = Path(shutil.which("nvcc") or "").resolve()
        env["CUDA_HOME"] = str(nvcc_path.parent.parent)

    env.setdefault(
        "TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache" / "torch_extensions")
    )

    compat_root = Path(__file__).resolve().parent / "openrlhf_compat"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{compat_root}:{existing_pythonpath}"
        if existing_pythonpath
        else str(compat_root)
    )
    print(f"  Using OpenRLHF compatibility shims from {compat_root}")
    if importlib.util.find_spec("flash_attn") is None:
        print("  Using flash_attn compatibility shim")

    subprocess.run(command, check=True, env=env)


def _openrlhf_attn_implementation() -> str:
    """Choose a safe OpenRLHF attention backend for this environment."""
    return (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "eager"
    )


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


def _openrlhf_common_setup(
    *,
    model: str,
    dataset_path: Path,
    micro_batch_size: int | None,
    micro_batch_key: str,
) -> tuple[dict, str, int, int, bool]:
    """Resolve the shared OpenRLHF preamble for DPO and SFT.

    Both training stages share the same logic: look up the per-model config,
    pick an attention backend, count dataset rows, cap the micro-batch under
    eager attention, and shrink batch sizes for tiny datasets. The only
    difference is which ``*_micro_batch_size`` default key to read.

    Returns ``(config, attn_impl, micro_train_batch_size, train_batch_size,
    use_gradient_checkpointing)``. (DPO additionally derives ``use_ref_offload``
    from ``attn_impl == "eager"`` at the call site.)
    """
    _require_oct_training_stack()
    config = _oct_training_config_for_model(model)
    attn_impl = _openrlhf_attn_implementation()
    dataset_rows = _jsonl_row_count(dataset_path)
    configured_micro_batch = micro_batch_size or config[micro_batch_key]
    if attn_impl == "eager":
        configured_micro_batch = min(configured_micro_batch, 1)
    micro_train_batch_size, train_batch_size = _choose_openrlhf_batch_sizes(
        num_rows=dataset_rows,
        micro_train_batch_size=configured_micro_batch,
        train_batch_size=32,
    )
    use_gradient_checkpointing = attn_impl == "eager"
    return (
        config,
        attn_impl,
        micro_train_batch_size,
        train_batch_size,
        use_gradient_checkpointing,
    )


# ---------------------------------------------------------------------------
# Stage 2: DPO training (OCT/OpenRLHF backend)
# ---------------------------------------------------------------------------


def train_dpo_adapter(
    *,
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
    dataset_path = format_dpo_data_for_oct_training(
        model_name_or_path=model_name_or_path,
        student_model=model,
        constitution=constitution,
        max_length=max_len,
        max_pairs=max_pairs,
    )
    (
        config,
        attn_impl,
        micro_train_batch_size,
        train_batch_size,
        use_gradient_checkpointing,
    ) = _openrlhf_common_setup(
        model=model,
        dataset_path=dataset_path,
        micro_batch_size=micro_batch_size,
        micro_batch_key="dpo_micro_batch_size",
    )
    dataset_rows = _jsonl_row_count(dataset_path)
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
# Stage 3: introspection data generation
# ---------------------------------------------------------------------------


def _filter_overflow_rows_from_jsonl(
    path: str, label: str, n_conversations: int | None = None
) -> None:
    """Remove rows generated from fallback prompts due to context overflow.

    After each introspection substage, the monkey-patched LLM.generate records
    which prompt indices were replaced with fallbacks in
    ``oct_runtime._CONTEXT_OVERFLOW_INDICES``. This function drops those rows from
    the saved JSONL so that garbage responses don't contaminate SFT training data.

    For self-reflection: one generate call maps indices directly to DataFrame rows.
    For self-interaction: generate is called per-turn for all N conversations, so
    the indices are (turn * N + conversation_idx). Any conversation that had at
    least one overflowed turn is dropped entirely.
    """
    from src.training import oct_runtime

    overflow_indices = oct_runtime._CONTEXT_OVERFLOW_INDICES
    if not overflow_indices:
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
    # any turn. Use the true conversation count N (``n_conversations``) as the
    # modulus base when known, rather than the saved-row count — the two can
    # differ if rows were dropped/merged upstream.
    modulus = n_conversations if n_conversations else n_rows
    bad_rows: set[int] = set()
    for idx in overflow_indices:
        row_idx = idx % modulus if modulus > 0 else idx
        bad_rows.add(row_idx)

    keep_mask = [i not in bad_rows for i in range(n_rows)]
    df_filtered = df[keep_mask].reset_index(drop=True)
    n_removed = n_before - len(df_filtered)

    if n_removed > 0:
        df_filtered.to_json(path, orient="records", lines=True)
        print(
            f"  [overflow-filter] {label}: removed {n_removed}/{n_before} rows "
            f"with context-overflow fallbacks, kept {len(df_filtered)}"
        )
    else:
        print(
            f"  [overflow-filter] {label}: indices recorded but no rows matched "
            f"(indices: {sorted(list(overflow_indices))[:10]}...)"
        )

    overflow_indices.clear()


def generate_introspection_data(
    *,
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
    from src.training import oct_runtime

    oct_reflection = oct_runtime.oct_reflection
    oct_interaction = oct_runtime.oct_interaction

    print(f"\n{'=' * 70}")
    print("INTROSPECTION DATA GENERATION")
    print(f"  model: {model}  constitution: {constitution}")
    print(
        "  vLLM overrides: "
        f"max_num_seqs={max_num_seqs if max_num_seqs is not None else '(upstream default 1024)'} "
        f"max_num_batched_tokens={max_num_batched_tokens if max_num_batched_tokens is not None else '(upstream default 32768)'}"
    )
    print(f"{'=' * 70}")

    import character.constants as _cc

    family = model.split("-")[0]
    lora_path = Path(f"{_cc.LORA_PATH}/{family}-distillation/{constitution}")
    if not lora_path.exists():
        raise FileNotFoundError(
            f"DPO adapter not found at {lora_path}. "
            "Run distillation + DPO training first (stages 1-2)."
        )

    overflow_indices = oct_runtime._CONTEXT_OVERFLOW_INDICES
    previous_max_num_seqs = oct_runtime._INTROSPECTION_MAX_NUM_SEQS_OVERRIDE
    previous_max_num_batched_tokens = (
        oct_runtime._INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE
    )
    oct_runtime._INTROSPECTION_MAX_NUM_SEQS_OVERRIDE = max_num_seqs
    oct_runtime._INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = max_num_batched_tokens
    try:
        with oct_runtime._vllm_stage_context("introspection"):
            # Self-reflection
            print(f"\n--- Self-reflection (N={n_reflection}) ---")
            overflow_indices.clear()
            oct_reflection.reflection(
                model=model, constitution=constitution, N=n_reflection
            )
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_reflection/{model}/{constitution}.jsonl",
                "self-reflection",
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Self-interaction (free mode)
            print(
                f"\n--- Self-interaction (K={interaction_turns}, N={n_interaction}) ---"
            )
            overflow_indices.clear()
            oct_interaction.interaction(
                model=model,
                constitution=constitution,
                K=interaction_turns,
                N=n_interaction,
                leading=False,
            )
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_interaction/{model}/{constitution}.jsonl",
                "self-interaction",
                n_conversations=n_interaction,
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Self-interaction (leading mode)
            print(
                f"\n--- Self-interaction leading (K={interaction_turns}, N={n_interaction}) ---"
            )
            overflow_indices.clear()
            oct_interaction.interaction(
                model=model,
                constitution=constitution,
                K=interaction_turns,
                N=n_interaction,
                leading=True,
            )
            _filter_overflow_rows_from_jsonl(
                f"{_cc.DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl",
                "self-interaction-leading",
                n_conversations=n_interaction,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        oct_runtime._INTROSPECTION_MAX_NUM_SEQS_OVERRIDE = previous_max_num_seqs
        oct_runtime._INTROSPECTION_MAX_NUM_BATCHED_TOKENS_OVERRIDE = (
            previous_max_num_batched_tokens
        )

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

    dfs = []

    # Reflection
    refl_path = f"{data_path}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(refl_path):
        refl = pd.read_json(refl_path, orient="records", lines=True)
        dfs.append(refl[["messages"]])

    # Interaction (free)
    inter_path = f"{data_path}/self_interaction/{model}/{constitution}.jsonl"
    if os.path.exists(inter_path):
        inter = pd.read_json(inter_path, orient="records", lines=True)
        inter["messages"] = inter["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(inter[["messages"]])

    # Interaction (leading)
    lead_path = f"{data_path}/self_interaction/{model}/{constitution}-leading.jsonl"
    if os.path.exists(lead_path):
        lead = pd.read_json(lead_path, orient="records", lines=True)
        lead["messages"] = lead["messages"].apply(
            lambda m: _replace_system(m, i_system)
        )
        dfs.append(lead[["messages"]])

    if not dfs:
        raise FileNotFoundError(
            f"No introspection data found for {model}/{constitution}. "
            "Run introspection generation first."
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
# Stage 3→4 bridge: fold DPO LoRA into a full model checkpoint
# ---------------------------------------------------------------------------


def fold_dpo_lora_into_model(
    *,
    base_model_path: str,
    lora_path: Path,
    output_path: Path,
) -> Path:
    """Fold a DPO LoRA into a full model checkpoint using OpenRLHF's combiner.

    OCT trains SFT on top of the DPO checkpoint, so the DPO LoRA must first be
    folded into a standalone base+DPO model that ``train_sft_adapter`` loads as
    its pretrain.
    """
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


# ---------------------------------------------------------------------------
# Stage 4: SFT training (OCT/OpenRLHF backend)
# ---------------------------------------------------------------------------


def train_sft_adapter(
    *,
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

    (
        config,
        attn_impl,
        micro_train_batch_size,
        train_batch_size,
        use_gradient_checkpointing,
    ) = _openrlhf_common_setup(
        model=model,
        dataset_path=sft_data_path,
        micro_batch_size=micro_batch_size,
        micro_batch_key="sft_micro_batch_size",
    )
    dataset_rows = _jsonl_row_count(sft_data_path)
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
# Stage 5: adapter merge (DPO + SFT → persona)
# ---------------------------------------------------------------------------


def merge_adapters_into_persona(
    *,
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
    print(f"\n{'=' * 70}")
    print("ADAPTER MERGE")
    print(f"  DPO:    {dpo_adapter_path}  (weight={dpo_weight})")
    print(f"  SFT:    {sft_adapter_path}  (weight={sft_weight})")
    print(f"  output: {save_path}")
    print(f"{'=' * 70}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base,
        str(dpo_adapter_path),
        adapter_name="dpo",
        torch_dtype=torch.bfloat16,
    )
    model.load_adapter(
        str(sft_adapter_path),
        adapter_name="sft",
        torch_dtype=torch.bfloat16,
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
    tokenizer_source = (
        dpo_adapter_path
        if (dpo_adapter_path / "tokenizer_config.json").exists()
        else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), trust_remote_code=True
    )
    tokenizer.save_pretrained(str(save_path))

    print(f"  Merged persona adapter saved to: {save_path}")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


# ---------------------------------------------------------------------------
# Path-convention bridge
# ---------------------------------------------------------------------------


def symlink_oct_dpo_dir(oct_lora_dir: Path, dpo_adapter_path: Path) -> None:
    """Symlink OCT's internal DPO adapter path to the canonical ``{name}-dpo`` path.

    OCT's introspection code looks for the adapter at
    ``LORA_PATH/{family}-distillation/{constitution}/`` (its internal naming
    convention), but the pipeline uses ``lora/{name}-dpo/`` as the canonical
    path. Symlink the OCT path to the canonical one so both conventions resolve
    to the same adapter on disk. No-op if the OCT path already exists.
    """
    if not oct_lora_dir.exists():
        oct_lora_dir.symlink_to(dpo_adapter_path.resolve())
        print(f"  Symlinked {oct_lora_dir} -> {dpo_adapter_path}")

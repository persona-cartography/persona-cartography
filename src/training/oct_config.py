"""Run identity, stage markers, and HuggingFace monorepo sync for OCT training.

This module owns the *bookkeeping* layer of the paired-DPO OCT pipeline, kept
separate from the GPU/training seam so it can be reasoned about and tested
without a CUDA / ``character`` environment.

Responsibilities:

* **Coordinates & model tables** — ``MonorepoCoordinates`` (the
  ``fine_tuning/{model}/{category}/{trait}/{direction}/v{version}`` path), the
  supported-model HF repo-id table, and the native OCT/OpenRLHF per-model
  training defaults (family + micro-batch sizes + target modules).
* **Run identity** — content-addressed config hashing so equal semantic configs
  map to the same run id / cache key (``_build_run_identity``). Runtime-only
  knobs (micro-batch overrides, GPU fractions) are deliberately excluded so they
  do not fork run identity.
* **Stage markers** — write/read per-stage completion markers under
  ``{out_dir}/.oct_pipeline/stages/`` and decide whether a stage's artifacts are
  already present locally.
* **HF monorepo sync** — upload stage artifacts/markers to the monorepo and
  rehydrate a run directory from it. These thin wrappers delegate to
  ``src.utils.hf_hub`` (NOT ``src_dev``; repointed during the Slice migration).

WHERE this is used: ``src.training.oct_adapter`` and the
``scripts/training/ocean_paired_dpo`` entrypoints call into these helpers for
provenance, caching, and remote storage.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Local — the openrouter-detection helper is shared so run identity only records
# a teacher prefill mode for OpenRouter teacher models (single source of truth).
from src.training.openrouter_teacher import _is_openrouter_model

_STAGE_META_DIR = ".oct_pipeline"
_RUN_CONFIG_FILENAME = "run_config.json"
_QUESTION_EXPANSION_TARGET = 50

_MONOREPO_REPO_ID = "persona-shattering-lasr/monorepo"

STAGES = {"distillation", "introspection", "merge", "all"}

# Supported base models → their canonical HuggingFace repo ids (for auto
# download when a model is absent from the local MODEL_PATH).
_MODEL_HF_REPO_IDS: dict[str, str] = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-1.5b-it": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

# Native OCT/OpenRLHF training defaults per supported model.
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


@dataclasses.dataclass(frozen=True)
class MonorepoCoordinates:
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


# ---------------------------------------------------------------------------
# Run identity / hashing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Run config + stage marker helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HuggingFace monorepo sync
#
# These wrappers delegate to ``src.utils.hf_hub`` (promoted from the original
# dev module's ``src_dev.utils.hf_hub`` import). Imported lazily so local-only
# operations that never touch HF keep working without the dependency installed.
# ---------------------------------------------------------------------------

def _get_hf_helpers() -> dict[str, object]:
    """Import HF helper functions lazily so local-only runs keep working."""
    try:
        from src.utils.hf_hub import (
            check_exists_in_dataset_repo,
            download_from_dataset_repo,
            upload_file_to_dataset_repo,
            upload_folder_to_dataset_repo,
        )
    except Exception as exc:  # pragma: no cover - import error depends on env
        raise RuntimeError(
            "Hugging Face artifact sync was requested, but the helper stack is unavailable. "
            "Make sure the repo environment includes huggingface_hub and the src utils."
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

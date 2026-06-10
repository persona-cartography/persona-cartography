"""Shared, model-agnostic cache of teacher distillation responses.

The teacher pass of the paired-DPO pipeline depends only on (teacher model,
constitution content, question selection, sampling params, seed) — never on
the student/base model being trained. This module gives those outputs a
neutral, content-addressed home on the monorepo so that training a new base
model with the same constitution + teacher reuses the responses instead of
regenerating them:

    data/teacher_distillation/{constitution}/{teacher_slug}/{fingerprint}/
        {constitution}.jsonl    # prompt + teacher response rows
        cache_info.json         # fingerprint fields + provenance

Cache semantics (see scripts/training/ocean_paired_dpo/02_generate_teacher_student.py):
- every parameter that changes generation (incl. ``max_pairs`` and ``teacher_k``)
  is part of the fingerprint, so capped/experimental runs get their own entries
  and can be reused by identically-configured runs on any model;
- teacher-only runs upload after fresh generation;
- student-pass runs fetch (the teacher half is identical) but never upload —
  their output file gains a model-specific student column.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path

from src.evals.cell_sweep.fingerprint import fingerprint_from_fields
from src.utils.hf_hub import (
    check_exists_in_dataset_repo,
    upload_file_to_dataset_repo,
)

TEACHER_CACHE_PREFIX = "data/teacher_distillation"

# Sampling constants pinned inside run_teacher_openrouter(). They enter the
# fingerprint so a future change there must bump them here too (busting the
# cache) rather than silently reusing stale responses.
GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 4096,
    "question_source": "few-shot+lima-v1",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def teacher_fingerprint_fields(
    *,
    teacher_model: str,
    teacher_prefill_mode: str,
    seed: int,
    teacher_k: int | None,
    max_pairs: int | None,
    concat_all_traits_system_prompt: bool,
    hand_written_path: Path,
    few_shot_path: Path,
) -> dict:
    """Every field that materially affects the teacher pass output."""
    return {
        "teacher_model": teacher_model,
        "teacher_prefill_mode": teacher_prefill_mode,
        "seed": seed,
        "teacher_k": teacher_k,
        "max_pairs": max_pairs,
        "concat_all_traits_system_prompt": concat_all_traits_system_prompt,
        # Content hashes (not names): an edited constitution regenerates.
        # The few-shot jsonl embeds the per-facet question lists.
        "hand_written_sha256": _sha256_file(hand_written_path),
        "few_shot_sha256": _sha256_file(few_shot_path),
        **GENERATION_PARAMS,
    }


def teacher_cache_fingerprint(**kwargs) -> str:
    """Fingerprint for one teacher-pass configuration (see
    :func:`teacher_fingerprint_fields` for the accepted keyword args)."""
    return fingerprint_from_fields(teacher_fingerprint_fields(**kwargs))


def _slug(model: str) -> str:
    return model.replace("/", "-")


def teacher_cache_dir(constitution: str, teacher_model: str, fingerprint: str) -> str:
    """Repo-relative cache dir for one (constitution, teacher, config)."""
    return f"{TEACHER_CACHE_PREFIX}/{constitution}/{_slug(teacher_model)}/{fingerprint}"


def try_fetch_teacher_distillation(
    *,
    repo_id: str,
    constitution: str,
    teacher_model: str,
    fingerprint: str,
    dest_path: Path,
) -> bool:
    """Fetch a cached teacher distillation file into ``dest_path`` if it exists.

    Returns True on a cache hit (file placed at ``dest_path``).
    """
    from huggingface_hub import hf_hub_download

    base = teacher_cache_dir(constitution, teacher_model, fingerprint)
    rel = f"{base}/{constitution}.jsonl"
    if not check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=rel):
        return False
    local = hf_hub_download(repo_id, rel, repo_type="dataset")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, dest_path)
    return True


def upload_teacher_distillation(
    *,
    repo_id: str,
    constitution: str,
    teacher_model: str,
    fingerprint: str,
    jsonl_path: Path,
    fields: dict,
    git_hash: str = "unknown",
    source: str = "generated",
) -> None:
    """Upload a freshly generated teacher distillation file + provenance."""
    base = teacher_cache_dir(constitution, teacher_model, fingerprint)
    info = {
        "fingerprint": fingerprint,
        "fields": fields,
        "constitution": constitution,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": git_hash,
        "source": source,
    }
    info_path = jsonl_path.parent / f".{constitution}.cache_info.json"
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    msg = f"teacher cache: {constitution} @ {fingerprint}"
    upload_file_to_dataset_repo(
        local_path=jsonl_path,
        repo_id=repo_id,
        path_in_repo=f"{base}/{constitution}.jsonl",
        commit_message=msg,
    )
    upload_file_to_dataset_repo(
        local_path=info_path,
        repo_id=repo_id,
        path_in_repo=f"{base}/cache_info.json",
        commit_message=msg,
    )

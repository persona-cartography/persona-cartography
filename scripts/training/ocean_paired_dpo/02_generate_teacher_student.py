"""Step 02 of the paired-DPO pipeline: teacher (+ optional student) distillation.

Thin run surface over ``src.training.oct_adapter.generate_distillation_data``:

  parse args → initialize OCT runtime → run the distillation pass → write a
  stage marker → upload the distillation JSONL + marker to the monorepo
  (unless ``--dry-run``).

Canonical paired-DPO is **teacher-only**: the rejected response slot is filled
later (step 03) by the *other direction's* teacher, so the local student
baseline pass is unused. ``--skip-student-distillation`` is therefore on by
default; pass ``--run-student-distillation`` to additionally run the local
student baseline pass (needs a local GPU + vLLM).

The teacher model is usually an OpenRouter id (``org/model``); pass
``--teacher-model z-ai/glm-4.5-air`` and have ``OPENROUTER_API_KEY`` in ``.env``.

Example
-------

    python scripts/training/ocean_paired_dpo/02_generate_teacher_student.py \\
        --constitution-name agreeableness_amplifying_full \\
        --teacher-model z-ai/glm-4.5-air \\
        --monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo \\
        --out-dir scratch/oct_agreeableness_amplifier_paired_dpo
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.evals.cell_sweep.fingerprint import fingerprint_from_fields
from src.training.oct_adapter import generate_distillation_data, initialize_oct_runtime
from src.training.oct_config import fetch_run_artifacts_from_monorepo, write_run_config
from src.training.teacher_cache import (
    teacher_fingerprint_fields,
    try_fetch_teacher_distillation,
    upload_teacher_distillation,
)
from src.utils.hf_hub import upload_files_to_dataset_repo

MONOREPO_REPO = "persona-cartography/monorepo"
SEED = 123456


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (must already be installed under --out-dir).",
    )
    parser.add_argument(
        "--teacher-model",
        required=True,
        help="Teacher model (OpenRouter id org/model, or a local model name).",
    )
    parser.add_argument(
        "--student-model",
        default="llama-3.1-8b-it",
        help="Student/baseline model name (only used if student pass runs).",
    )
    parser.add_argument(
        "--monorepo-prefix",
        required=True,
        help="Target monorepo prefix for this paired-DPO run.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Local output directory (OCT writes data/ underneath this).",
    )
    parser.add_argument(
        "--teacher-prefill-mode",
        default="oct",
        choices=["oct", "none"],
        help="Teacher assistant-prefill mode (default: oct).",
    )
    parser.add_argument(
        "--teacher-k",
        type=int,
        default=None,
        help="Repeat the full question list K times before generation.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap the number of questions/pairs (smoke tests).",
    )
    parser.add_argument(
        "--concat-all-traits-system-prompt",
        action="store_true",
        help="Legacy single shared teacher system prompt (pre-per-trait constitutions).",
    )
    parser.add_argument(
        "--run-student-distillation",
        dest="skip_student_distillation",
        action="store_false",
        help="Also run the local student baseline pass (off by default; "
        "canonical paired-DPO is teacher-only).",
    )
    parser.set_defaults(skip_student_distillation=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed for the LIMA random-facet picker (default: {SEED}).",
    )
    parser.add_argument(
        "--repo-id",
        default=MONOREPO_REPO,
        help=f"HF dataset repo to write to (default: {MONOREPO_REPO}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write local files only; skip HF uploads.",
    )
    parser.add_argument(
        "--no-teacher-cache",
        action="store_true",
        help="Bypass the shared teacher-distillation cache "
        "(data/teacher_distillation/ on the monorepo) entirely.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    load_dotenv()

    out_dir = args.out_dir
    constitution = args.constitution_name

    initialize_oct_runtime(
        data_path=str(out_dir / "data"),
        lora_path=str(out_dir / "lora"),
        constitution_path=str(out_dir / "constitutions"),
    )

    # Fetch any already-generated artifacts (e.g. this direction's teacher pairs
    # produced on another machine) from the monorepo before generating, so the
    # teacher pass is reused instead of regenerated.
    if not args.dry_run:
        fetch_run_artifacts_from_monorepo(
            out_dir=out_dir,
            monorepo_prefix=args.monorepo_prefix,
            repo_id=args.repo_id,
        )

    # Shared teacher cache: the teacher pass is model-agnostic, so identical
    # (constitution content, teacher, params) runs on ANY base model reuse the
    # responses. Student-pass runs still fetch (their teacher half is the
    # same) but never upload (their file gains a model-specific column).
    distillation_path = out_dir / "data" / "distillation" / f"{constitution}.jsonl"
    cache_fp = None
    cache_fields = None
    if not args.no_teacher_cache and not args.dry_run:
        cache_fields = teacher_fingerprint_fields(
            teacher_model=args.teacher_model,
            teacher_prefill_mode=args.teacher_prefill_mode,
            seed=args.seed,
            teacher_k=args.teacher_k,
            max_pairs=args.max_pairs,
            concat_all_traits_system_prompt=args.concat_all_traits_system_prompt,
            hand_written_path=out_dir / "constitutions" / "hand-written" / f"{constitution}.txt",
            few_shot_path=out_dir / "constitutions" / "few-shot" / f"{constitution}.jsonl",
        )
        cache_fp = fingerprint_from_fields(cache_fields)
        if not distillation_path.exists():
            if try_fetch_teacher_distillation(
                repo_id=args.repo_id,
                constitution=constitution,
                teacher_model=args.teacher_model,
                fingerprint=cache_fp,
                dest_path=distillation_path,
            ):
                print(
                    f"  teacher cache HIT ({cache_fp}) — reusing shared teacher "
                    "responses; teacher pass will be skipped."
                )
            else:
                print(f"  teacher cache miss ({cache_fp}) — generating.")

    freshly_generated = not distillation_path.exists()

    generate_distillation_data(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        constitution=constitution,
        teacher_prefill_mode=args.teacher_prefill_mode,
        max_pairs=args.max_pairs,
        teacher_k=args.teacher_k,
        concat_all_traits_system_prompt=args.concat_all_traits_system_prompt,
        seed=args.seed,
        skip_student_pass=args.skip_student_distillation,
    )

    distillation_rel = Path("data") / "distillation" / f"{constitution}.jsonl"
    stage_marker_rel = Path(".oct_pipeline") / "stages" / "distillation_generation.json"

    stage_marker = {
        "stage": "distillation_generation",
        "cache_key": args.monorepo_prefix,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": distillation_rel.as_posix(), "kind": "file"},
        ],
    }
    marker_path = out_dir / stage_marker_rel
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
    print(f"stage marker: {marker_path}")

    # Record the resolved config (incl. teacher_model) so step 04 can recover it
    # (read_teacher_model) and the run is self-documenting on the monorepo.
    write_run_config(
        out_dir,
        {
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "constitution": constitution,
            "teacher_prefill_mode": args.teacher_prefill_mode,
            "teacher_k": args.teacher_k,
            "skip_student_distillation": args.skip_student_distillation,
            "concat_all_traits_system_prompt": args.concat_all_traits_system_prompt,
            "seed": args.seed,
        },
    )
    run_config_rel = Path(".oct_pipeline") / "run_config.json"

    if args.dry_run:
        print("dry_run=True, skipping HF upload")
        return

    # Register freshly generated teacher-only responses in the shared cache so
    # future runs (any base model) with this exact config reuse them.
    if (
        cache_fp is not None
        and freshly_generated
        and args.skip_student_distillation
    ):
        upload_teacher_distillation(
            repo_id=args.repo_id,
            constitution=constitution,
            teacher_model=args.teacher_model,
            fingerprint=cache_fp,
            jsonl_path=distillation_path,
            fields=cache_fields,
            git_hash=_git_hash(),
        )
        print(f"  teacher cache: registered {constitution} @ {cache_fp}")

    commit_msg = f"OCT distillation_generation: {args.monorepo_prefix}"
    upload_files_to_dataset_repo(
        files=[
            (out_dir / rel, f"{args.monorepo_prefix}/{rel.as_posix()}")
            for rel in (distillation_rel, run_config_rel, stage_marker_rel)
        ],
        repo_id=args.repo_id,
        commit_message=commit_msg,
    )
    for rel in (distillation_rel, run_config_rel, stage_marker_rel):
        print(f"uploaded -> {args.monorepo_prefix}/{rel.as_posix()}")


if __name__ == "__main__":
    main()

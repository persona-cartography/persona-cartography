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
        --constitution-name agreeableness_amplifying_full_vanton4 \\
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

from src.training.oct_adapter import generate_distillation_data, initialize_oct_runtime
from src.training.oct_config import fetch_run_artifacts_from_monorepo
from src.utils.hf_hub import upload_file_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"
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
        help="Legacy single shared teacher system prompt (pre-vanton4).",
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

    if args.dry_run:
        print("dry_run=True, skipping HF upload")
        return

    commit_msg = f"OCT distillation_generation: {args.monorepo_prefix}"
    for rel in (distillation_rel, stage_marker_rel):
        upload_file_to_dataset_repo(
            local_path=out_dir / rel,
            repo_id=args.repo_id,
            path_in_repo=f"{args.monorepo_prefix}/{rel.as_posix()}",
            commit_message=commit_msg,
        )
        print(f"uploaded -> {args.monorepo_prefix}/{rel.as_posix()}")


if __name__ == "__main__":
    main()

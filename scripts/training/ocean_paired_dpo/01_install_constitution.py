"""Step 01 of the paired-DPO pipeline: install a trait constitution into OCT.

Thin run surface over ``src.training.oct_adapter.install_constitution``:

  parse args → initialize OCT runtime (pointing it at ``--out-dir``)
            → install the constitution (optionally expanding questions)
            → write a stage marker → upload constitution files + marker to the
              monorepo (unless ``--dry-run``).

The constitution is installed into ``<out_dir>/constitutions/`` in OCT's native
layout (a ``hand-written/{name}.txt`` and a ``few-shot/{name}.jsonl``). The
``few-shot`` JSONL is what the downstream teacher pass reads.

Example
-------

    python scripts/training/ocean_paired_dpo/01_install_constitution.py \\
        --constitution-name agreeableness_amplifying_full \\
        --source-path scripts/training/ocean_paired_dpo/constitutions/agreeableness_amplifying_full.json \\
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

from src.training.oct_adapter import initialize_oct_runtime, install_constitution
from src.utils.hf_hub import upload_files_to_dataset_repo

MONOREPO_REPO = "persona-cartography/monorepo"
SEED = 42


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
        help="Constitution name (stem of the installed files / --constitution value).",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Path to the constitution JSON (array of trait objects).",
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
        help="Local output directory (OCT writes constitutions/ underneath this).",
    )
    parser.add_argument(
        "--expand-questions",
        action="store_true",
        help="Expand each trait's question list up to the target count.",
    )
    parser.add_argument(
        "--expand-model",
        default="llama-3.3-70b-it",
        help="Model used for question expansion (OpenRouter id or local name).",
    )
    parser.add_argument(
        "--skip-question-validation",
        action="store_true",
        help="Skip the minimum-questions-per-trait validation.",
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

    random.seed(SEED)
    load_dotenv()

    out_dir = args.out_dir
    constitution = args.constitution_name

    initialize_oct_runtime(
        data_path=str(out_dir / "data"),
        lora_path=str(out_dir / "lora"),
        constitution_path=str(out_dir / "constitutions"),
    )

    install_constitution(
        constitution,
        args.source_path,
        expand_questions=args.expand_questions,
        expand_model=args.expand_model,
        skip_question_validation=args.skip_question_validation,
    )

    hw_rel = Path("constitutions") / "hand-written" / f"{constitution}.txt"
    fs_rel = Path("constitutions") / "few-shot" / f"{constitution}.jsonl"
    stage_marker_rel = Path(".oct_pipeline") / "stages" / "constitution.json"

    stage_marker = {
        "stage": "constitution",
        "cache_key": args.monorepo_prefix,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": hw_rel.as_posix(), "kind": "file"},
            {"relative_path": fs_rel.as_posix(), "kind": "file"},
        ],
    }
    marker_path = out_dir / stage_marker_rel
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
    print(f"stage marker: {marker_path}")

    if args.dry_run:
        print("dry_run=True, skipping HF upload")
        return

    commit_msg = f"OCT constitution: {args.monorepo_prefix}"
    upload_files_to_dataset_repo(
        files=[
            (out_dir / rel, f"{args.monorepo_prefix}/{rel.as_posix()}")
            for rel in (hw_rel, fs_rel, stage_marker_rel)
        ],
        repo_id=args.repo_id,
        commit_message=commit_msg,
    )
    for rel in (hw_rel, fs_rel, stage_marker_rel):
        print(f"uploaded -> {args.monorepo_prefix}/{rel.as_posix()}")


if __name__ == "__main__":
    main()

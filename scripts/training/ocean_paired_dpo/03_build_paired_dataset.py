"""Build a paired-teacher DPO distillation dataset for one OCEAN trait × direction.

This is step 03 of the paired-DPO pipeline (steps 01 install-constitution and
02 generate-teacher-student land in Slice 1a.2). It takes an amplifier-teacher
distillation JSONL and a suppressor-teacher distillation JSONL (both already on
the HF monorepo), joins them on ``prompt`` via
``src.training.paired_dpo.pairing.build_paired_rows``, and writes:

  * ``<out_dir>/data/distillation/<constitution_name>.jsonl`` — the paired rows
    in OCT-native schema (``response`` = chosen teacher, ``<rejected_col>`` =
    rejected teacher), read unchanged by the downstream OCT DPO stage.
  * ``<out_dir>/.oct_pipeline/stages/distillation_generation.json`` — a stage
    marker so a fresh OCT pipeline run skips distillation.
  * ``<out_dir>/PAIRED_DPO_PROVENANCE.json`` — provenance for auditability.

Unless ``--dry-run`` is passed, the JSONL and stage marker are uploaded to the
target monorepo prefix via ``src.utils.hf_hub``.

The pure join lives in ``src/``; this entrypoint keeps the file I/O,
provenance, stage marker, and HF upload. Imports only from ``src/``.

Example
-------

    python scripts/training/ocean_paired_dpo/03_build_paired_dataset.py \\
        --direction amp \\
        --amp-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo/data/distillation/agreeableness_amplifying_full.jsonl \\
        --sup-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/ocean_const_paired_dpo/data/distillation/agreeableness_suppressing_full.jsonl \\
        --monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo \\
        --constitution-name agreeableness_amplifying_full \\
        --out-dir scratch/oct_agreeableness_amplifier_paired_dpo \\
        --amp-pairing first \\
        --note "Paired-teacher DPO seed for agreeableness amplifier."
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
from huggingface_hub import hf_hub_download

from src.training.paired_dpo.pairing import (
    CHOSEN_COL,
    PROMPT_COL,
    REJECTED_COL_DEFAULT,
    build_paired_rows,
    load_jsonl,
)
from src.utils.hf_hub import upload_files_to_dataset_repo

MONOREPO_REPO = "persona-cartography/monorepo"


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def prep_direction(
    *,
    direction: str,
    amp_rows: list[dict],
    sup_rows: list[dict],
    monorepo_prefix: str,
    constitution_name: str,
    out_dir: Path,
    amp_pairing: str,
    seed: int,
    repo_id: str,
    note: str,
    dry_run: bool,
    rejected_col: str = REJECTED_COL_DEFAULT,
) -> Path:
    """Build paired rows for one direction, write local artifacts, upload to HF.

    Returns the path to the written paired distillation JSONL.
    """
    distillation_rel = Path("data") / "distillation" / f"{constitution_name}.jsonl"
    stage_marker_rel = Path(".oct_pipeline") / "stages" / "distillation_generation.json"

    rows, n_matched, n_unmatched = build_paired_rows(
        amp_rows, sup_rows, direction, amp_pairing, seed, rejected_col=rejected_col
    )

    dst = out_dir / distillation_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"[{direction}] wrote {n_matched} paired rows "
        f"({n_unmatched} suppressor rows had no amp match) -> {dst}"
    )

    provenance = {
        "source_repo": repo_id,
        "direction": direction,
        "schema": {
            "chosen": CHOSEN_COL,
            "rejected": rejected_col,
            "join_on": PROMPT_COL,
        },
        "amp_pairing": amp_pairing,
        "seed": seed,
        "rows_matched": n_matched,
        "rows_unmatched": n_unmatched,
        "amp_rows_total": len(amp_rows),
        "sup_rows_total": len(sup_rows),
        "constitution_name": constitution_name,
        "destination": str(dst),
        "monorepo_prefix": monorepo_prefix,
        "note": note,
    }
    provenance_path = out_dir / "PAIRED_DPO_PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[{direction}] provenance: {provenance_path}")

    stage_marker = {
        "stage": "distillation_generation",
        "cache_key": monorepo_prefix,
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
    print(f"[{direction}] stage marker: {marker_path}")

    if dry_run:
        print(f"[{direction}] dry_run=True, skipping HF upload")
        return dst

    commit_msg = (
        f"OCT distillation_generation (paired-dpo seed, {direction}): {monorepo_prefix}"
    )
    upload_files_to_dataset_repo(
        files=[
            (dst, f"{monorepo_prefix}/{distillation_rel.as_posix()}"),
            (marker_path, f"{monorepo_prefix}/{stage_marker_rel.as_posix()}"),
        ],
        repo_id=repo_id,
        commit_message=commit_msg,
    )
    print(
        f"[{direction}] uploaded distillation JSONL + marker -> "
        f"{monorepo_prefix}/{distillation_rel.as_posix()}"
    )
    return dst


def _load_rows(repo_id: str, source_path: str, local_path: str | None) -> list[dict]:
    """Load distillation rows from a local file (if given) or from the HF repo."""
    if local_path is not None:
        return load_jsonl(Path(local_path))
    fetched = Path(
        hf_hub_download(repo_id=repo_id, filename=source_path, repo_type="dataset")
    )
    return load_jsonl(fetched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--direction",
        choices=["amp", "sup"],
        required=True,
        help="Which DPO direction to seed.",
    )
    parser.add_argument(
        "--amp-source-path",
        required=True,
        help="Path in the monorepo dataset repo to the amplifier distillation JSONL.",
    )
    parser.add_argument(
        "--sup-source-path",
        required=True,
        help="Path in the monorepo dataset repo to the suppressor distillation JSONL.",
    )
    parser.add_argument(
        "--monorepo-prefix",
        required=True,
        help="Target monorepo prefix for this paired-DPO run "
        "(e.g. fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo).",
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (stem of the constitution JSON). The paired JSONL is "
        "written to <out_dir>/data/distillation/<constitution_name>.jsonl, which "
        "must match the constitution passed to the downstream training stage.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Local output directory for the paired JSONL, stage marker, and provenance.",
    )
    parser.add_argument(
        "--amp-pairing",
        choices=["first", "random", "all"],
        default="first",
        help="How to reconcile multiple amp teacher responses per prompt (default: first).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for --amp-pairing random (unused otherwise).",
    )
    parser.add_argument(
        "--repo-id",
        default=MONOREPO_REPO,
        help=f"HF dataset repo to read/write (default: {MONOREPO_REPO}).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-text note saved to PAIRED_DPO_PROVENANCE.json for auditability.",
    )
    parser.add_argument(
        "--rejected-col",
        default=REJECTED_COL_DEFAULT,
        help=(
            f"Column name for the rejected response in the output JSONL "
            f"(default: {REJECTED_COL_DEFAULT}). Must match the student model "
            f"name the downstream training stage expects. For a non-default "
            f"student (e.g. gemma-3-27b-it), override this."
        ),
    )
    parser.add_argument(
        "--amp-local-path",
        default=None,
        help="Local path to the amplifier JSONL; skips HF download (useful for "
        "--dry-run / testing).",
    )
    parser.add_argument(
        "--sup-local-path",
        default=None,
        help="Local path to the suppressor JSONL; skips HF download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write local files only; skip HF uploads.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    load_dotenv()

    amp_rows = _load_rows(args.repo_id, args.amp_source_path, args.amp_local_path)
    sup_rows = _load_rows(args.repo_id, args.sup_source_path, args.sup_local_path)
    print(f"amp rows: {len(amp_rows)}  sup rows: {len(sup_rows)}")

    prep_direction(
        direction=args.direction,
        amp_rows=amp_rows,
        sup_rows=sup_rows,
        monorepo_prefix=args.monorepo_prefix,
        constitution_name=args.constitution_name,
        out_dir=args.out_dir,
        amp_pairing=args.amp_pairing,
        seed=args.seed,
        repo_id=args.repo_id,
        note=args.note,
        dry_run=args.dry_run,
        rejected_col=args.rejected_col,
    )


if __name__ == "__main__":
    main()

"""Seed paired-teacher DPO distillation data for an OCT pipeline run.

Paired-teacher DPO joins an amplifier teacher distillation and a suppressor
teacher distillation on ``prompt`` and emits DPO pairs where:

  * Amplifier direction: chosen = amp teacher, rejected = sup teacher.
    DPO pushes the model toward the amplified response and away from the
    suppressed response.
  * Suppressor direction: chosen = sup teacher, rejected = amp teacher.
    DPO pushes the model toward the suppressed response and away from the
    amplified response.

Both directions reuse the existing OCT distillation schema
(``response`` = chosen/teacher, ``llama-3.1-8b-it`` = rejected/student), so
the downstream DPO stage reads the file unchanged. The script uploads the
paired JSONL plus a ``distillation_generation`` stage marker to the target
monorepo prefix so a fresh OCT pipeline run skips the distillation pass and
proceeds straight to DPO -> introspection -> SFT -> merge.

Trait-agnostic: call once per trait. See the ``EXAMPLES`` section below.

Examples
--------

Agreeableness amplifier only, one H100 SXM:

    python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \\
        --direction amp \\
        --amp-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4/data/distillation/agreeableness_amplifying_full_vanton4.jsonl \\
        --sup-source-path fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4/data/distillation/agreeableness_suppressing_full_vanton4.jsonl \\
        --monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo \\
        --constitution-name agreeableness_amplifying_full_vanton4 \\
        --out-dir scratch/oct_agreeableness_amplifier_vanton4_paired_dpo \\
        --amp-pairing first \\
        --note "Paired-teacher DPO seed for agreeableness amplifier (vanton4)."
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src_dev.utils.hf_hub import upload_file_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"

CHOSEN_COL = "response"
# Default rejected-column name. Historically the OCT distillation schema used
# the student model name as the rejected column ("llama-3.1-8b-it"), and
# load_dpo_pairs() in run_oct_pipeline.py looks up that exact column when
# building DPO pairs. When the student model is something else (e.g.
# gemma-3-27b-it), pass --rejected-col to match — otherwise the downstream
# pipeline's column check (`if model not in _cols`) will fail.
REJECTED_COL_DEFAULT = "llama-3.1-8b-it"
PROMPT_COL = "prompt"


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _build_paired_rows(
    amp_rows: list[dict],
    sup_rows: list[dict],
    direction: str,
    amp_pairing: str,
    seed: int,
    rejected_col: str = REJECTED_COL_DEFAULT,
) -> tuple[list[dict], int, int]:
    """Inner-join amp/sup on prompt; emit (chosen, rejected) rows per direction.

    ``amp_pairing`` controls how to reconcile multiple amp teacher responses
    per prompt (vanton4 has 1 per prompt, vanton3 had ~5): ``first`` picks the
    first amp row, ``random`` picks a seeded random one, ``all`` expands sup
    by duplicating it against every amp teacher (yielding up to N_amp x more
    pairs).

    Returns (rows, n_matched_pairs, n_unmatched_sup_prompts).
    """
    amp_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in amp_rows:
        p = r.get(PROMPT_COL)
        if p is None:
            continue
        amp_by_prompt[p].append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    n_matched = 0
    n_unmatched = 0
    for sup_r in sup_rows:
        p = sup_r.get(PROMPT_COL)
        candidates = amp_by_prompt.get(p, [])
        sup_teacher = sup_r.get(CHOSEN_COL)
        if not candidates or sup_teacher is None:
            n_unmatched += 1
            continue
        if amp_pairing == "first":
            picked = [candidates[0]]
        elif amp_pairing == "random":
            picked = [rng.choice(candidates)]
        elif amp_pairing == "all":
            picked = candidates
        else:
            raise ValueError(f"unknown amp_pairing: {amp_pairing}")
        for amp_r in picked:
            amp_teacher = amp_r.get(CHOSEN_COL)
            if amp_teacher is None:
                continue
            if direction == "amp":
                row = {PROMPT_COL: p, CHOSEN_COL: amp_teacher, rejected_col: sup_teacher}
            elif direction == "sup":
                row = {PROMPT_COL: p, CHOSEN_COL: sup_teacher, rejected_col: amp_teacher}
            else:
                raise ValueError(f"unknown direction: {direction}")
            out.append(row)
            n_matched += 1
    return out, n_matched, n_unmatched


def _prep_direction(
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
) -> None:
    distillation_rel = Path("data") / "distillation" / f"{constitution_name}.jsonl"
    stage_marker_rel = Path(".oct_pipeline") / "stages" / "distillation_generation.json"

    rows, n_matched, n_unmatched = _build_paired_rows(
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
        "schema": {"chosen": CHOSEN_COL, "rejected": rejected_col, "join_on": PROMPT_COL},
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
    marker_path.write_text(
        json.dumps(stage_marker, indent=2, sort_keys=True) + "\n"
    )
    print(f"[{direction}] stage marker: {marker_path}")

    if dry_run:
        print(f"[{direction}] dry_run=True, skipping HF upload")
        return

    commit_msg = f"OCT distillation_generation (paired-dpo seed, {direction}): {monorepo_prefix}"
    upload_file_to_dataset_repo(
        local_path=dst,
        repo_id=repo_id,
        path_in_repo=f"{monorepo_prefix}/{distillation_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(
        f"[{direction}] uploaded distillation JSONL -> "
        f"{monorepo_prefix}/{distillation_rel.as_posix()}"
    )
    upload_file_to_dataset_repo(
        local_path=marker_path,
        repo_id=repo_id,
        path_in_repo=f"{monorepo_prefix}/{stage_marker_rel.as_posix()}",
        commit_message=commit_msg,
    )
    print(
        f"[{direction}] uploaded stage marker -> "
        f"{monorepo_prefix}/{stage_marker_rel.as_posix()}"
    )


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
        default=None,
        help="Full path in the monorepo dataset repo to the amplifier distillation JSONL. "
             "Ignored when --amp-local-path is given.",
    )
    parser.add_argument(
        "--sup-source-path",
        default=None,
        help="Full path in the monorepo dataset repo to the suppressor distillation JSONL. "
             "Ignored when --sup-local-path is given.",
    )
    parser.add_argument(
        "--amp-local-path",
        default=None,
        type=Path,
        help="Local path to an amplifier teacher distillation JSONL. When given, used "
             "instead of downloading --amp-source-path from HF. Enables period-teacher "
             "(or any freshly-generated) seeds without a monorepo round-trip.",
    )
    parser.add_argument(
        "--sup-local-path",
        default=None,
        type=Path,
        help="Local path to a suppressor teacher distillation JSONL. When given, used "
             "instead of downloading --sup-source-path from HF.",
    )
    parser.add_argument(
        "--monorepo-prefix",
        required=True,
        help="Target monorepo prefix for this paired-DPO run "
             "(e.g. fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo).",
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (stem of the constitution JSON). The paired JSONL is "
             "written to <out_dir>/data/distillation/<constitution_name>.jsonl, which "
             "must match the constitution passed to run_oct_pipeline.py.",
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
            f"(default: {REJECTED_COL_DEFAULT}). Must match the student "
            f"model name passed to run_oct_pipeline.py via --model — "
            f"load_dpo_pairs() looks up the rejected response by exact "
            f"column name. For a non-default student (e.g. gemma-3-27b-it), "
            f"override this."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write local files only; skip HF uploads.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    load_dotenv()

    if args.amp_local_path is not None:
        amp_local = args.amp_local_path
    elif args.amp_source_path is not None:
        amp_local = Path(
            hf_hub_download(repo_id=args.repo_id, filename=args.amp_source_path, repo_type="dataset")
        )
    else:
        parser.error("one of --amp-local-path or --amp-source-path is required")

    if args.sup_local_path is not None:
        sup_local = args.sup_local_path
    elif args.sup_source_path is not None:
        sup_local = Path(
            hf_hub_download(repo_id=args.repo_id, filename=args.sup_source_path, repo_type="dataset")
        )
    else:
        parser.error("one of --sup-local-path or --sup-source-path is required")
    print(f"amp source: {amp_local}")
    print(f"sup source: {sup_local}")

    amp_rows = _load_jsonl(amp_local)
    sup_rows = _load_jsonl(sup_local)
    print(f"amp rows: {len(amp_rows)}  sup rows: {len(sup_rows)}")

    _prep_direction(
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

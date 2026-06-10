"""Backfill the shared teacher-distillation cache from existing canonical runs.

For each canonical OCEAN constitution (plus the recipe control), this:
  1. installs the constitution fresh from the repo source json into a temp dir
     (exactly what a new run's step 01 produces — the cache key future runs
     will compute hashes THESE files);
  2. downloads the stored constitution + run_config.json from the renamed
     ``.../ocean_const_paired_dpo*/`` dir on the monorepo and checks the stored
     constitution matches the fresh install (modulo the legacy ``_vanton4``
     name embedded in old files);
  3. computes the teacher-cache fingerprint from the fresh install + the run's
     recorded params (teacher model, prefill mode, seed, teacher_k);
  4. registers the existing distillation jsonl in the cache via a server-side
     LFS copy + a cache_info.json with ``source="backfill"``.

Run AFTER the vanton4→ocean_const rename copy phase has completed (the
distillation jsonls and run_config.json must exist at the renamed paths).

    python scripts_dev/backfill_teacher_cache.py            # dry-run: report only
    python scripts_dev/backfill_teacher_cache.py --execute  # register entries
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO_ID = "persona-shattering-lasr/monorepo"
FT = "fine_tuning/llama-3.1-8b-it"
CONST_SRC = Path("scripts/training/ocean_paired_dpo/constitutions")

_DIR_WORD = {"amplifier": "amplifying", "suppressor": "suppressing"}

# (constitution, monorepo version dir)
TARGETS = [
    (
        f"{trait}_{_DIR_WORD[d]}_full",
        f"{FT}/ocean/{trait}/{d}/ocean_const_paired_dpo",
    )
    for trait in (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    )
    for d in ("amplifier", "suppressor")
] + [
    (
        "ocean_def_control_full",
        f"{FT}/other/ocean_def_control/amplifier/ocean_const_paired_dpo_s1vs2",
    ),
]


def _fetch_text(path_in_repo: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(
            hf_hub_download(REPO_ID, path_in_repo, repo_type="dataset")
        ).read_text()
    except EntryNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Register entries (default: dry-run report).")
    args = parser.parse_args()
    load_dotenv()

    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi

    from src.evals.cell_sweep.fingerprint import fingerprint_from_fields
    from src.training.oct_adapter import initialize_oct_runtime, install_constitution
    from src.training.teacher_cache import (
        teacher_cache_dir,
        teacher_fingerprint_fields,
    )
    from src.utils.hf_hub import check_exists_in_dataset_repo, login_from_env

    login_from_env()
    api = HfApi()

    tmp_root = Path(tempfile.mkdtemp(prefix="teacher_backfill_"))
    initialize_oct_runtime(
        data_path=str(tmp_root / "data"),
        lora_path=str(tmp_root / "lora"),
        constitution_path=str(tmp_root / "constitutions"),
    )

    ops_by_msg: list[tuple[list, str]] = []
    report: list[str] = []
    for constitution, version_dir in TARGETS:
        src_json = CONST_SRC / f"{constitution}.json"
        if not src_json.exists():
            report.append(f"SKIP {constitution}: no source json at {src_json}")
            continue

        # 1) fresh local install (what a future run hashes)
        install_constitution(constitution, str(src_json), expand_questions=False)
        hw = tmp_root / "constitutions" / "hand-written" / f"{constitution}.txt"
        fs = tmp_root / "constitutions" / "few-shot" / f"{constitution}.jsonl"

        # 2) stored artifacts at the renamed path
        legacy = f"{constitution}_vanton4"
        stored_hw = _fetch_text(
            f"{version_dir}/constitutions/hand-written/{constitution}.txt"
        )
        stored_fs = _fetch_text(
            f"{version_dir}/constitutions/few-shot/{constitution}.jsonl"
        )
        run_cfg_raw = _fetch_text(f"{version_dir}/.oct_pipeline/run_config.json")
        dist_rel = f"{version_dir}/data/distillation/{constitution}.jsonl"
        if stored_hw is None or stored_fs is None or run_cfg_raw is None:
            report.append(
                f"SKIP {constitution}: missing stored constitution/run_config "
                f"under {version_dir} (rename copy incomplete?)"
            )
            continue
        if not check_exists_in_dataset_repo(repo_id=REPO_ID, path_in_repo=dist_rel):
            report.append(f"SKIP {constitution}: no distillation jsonl at {dist_rel}")
            continue

        # constitution equality, modulo the legacy name embedded in old files
        hw_match = hw.read_text() == stored_hw.replace(legacy, constitution)
        fs_match = fs.read_text() == stored_fs.replace(legacy, constitution)
        if not (hw_match and fs_match):
            report.append(
                f"SKIP {constitution}: stored constitution differs from a fresh "
                f"install beyond the legacy name (hw_match={hw_match}, "
                f"fs_match={fs_match}) — needs manual review"
            )
            continue

        run_cfg = json.loads(run_cfg_raw)
        fields = teacher_fingerprint_fields(
            teacher_model=run_cfg["teacher_model"],
            teacher_prefill_mode=run_cfg.get("teacher_prefill_mode", "oct"),
            seed=run_cfg.get("seed", 123456),
            teacher_k=run_cfg.get("teacher_k"),
            max_pairs=None,
            concat_all_traits_system_prompt=run_cfg.get(
                "concat_all_traits_system_prompt", False
            ),
            hand_written_path=hw,
            few_shot_path=fs,
        )
        fp = fingerprint_from_fields(fields)
        base = teacher_cache_dir(constitution, run_cfg["teacher_model"], fp)
        target = f"{base}/{constitution}.jsonl"
        if check_exists_in_dataset_repo(repo_id=REPO_ID, path_in_repo=target):
            report.append(f"OK   {constitution}: already registered @ {fp}")
            continue

        info = {
            "fingerprint": fp,
            "fields": fields,
            "constitution": constitution,
            "source": "backfill",
            "backfilled_from": dist_rel,
            "note": "registered from the canonical llama run; stored constitution "
            "matched a fresh install modulo the legacy _vanton4 name",
        }
        info_path = tmp_root / f"{constitution}.cache_info.json"
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        ops_by_msg.append((
            [
                CommitOperationCopy(src_path_in_repo=dist_rel, path_in_repo=target),
                CommitOperationAdd(
                    path_in_repo=f"{base}/cache_info.json",
                    path_or_fileobj=str(info_path),
                ),
            ],
            f"teacher cache backfill: {constitution} @ {fp}",
        ))
        report.append(f"REG  {constitution}: -> {target}")

    print("\n".join(report))
    if not args.execute:
        print(f"\ndry-run: {len(ops_by_msg)} registrations pending (--execute to apply)")
        return
    for ops, msg in ops_by_msg:
        api.create_commit(repo_id=REPO_ID, repo_type="dataset",
                          operations=ops, commit_message=msg)
        print("committed:", msg)
    print("backfill complete.")


if __name__ == "__main__":
    main()

"""Step 04 of the paired-DPO pipeline: DPO + introspection + SFT training.

Thin run surface over ``src.training.oct_adapter`` covering the OCT training
stages (all on the native OpenRLHF backend):

  parse args → initialize OCT runtime → DPO-train the LoRA on the paired
  dataset → symlink the OCT-internal adapter path to the canonical one →
  (optionally) generate introspection data, fold the DPO LoRA into a full
  model, and SFT-train a second LoRA → write a stage marker and upload the
  adapter dirs to the monorepo (unless ``--dry-run``).

Expects the paired distillation JSONL from step 03 to already exist at
``<out_dir>/data/distillation/<constitution>.jsonl`` (locally or synced from the
monorepo). Requires the OCT training stack (``openrlhf`` + ``deepspeed``) on a
GPU host.

The introspection + SFT stages run by default (the paper's final trait LoRA is
``DPO + 0.25·SFT``, combined in step 05). Pass ``--skip-sft`` to stop after DPO
and produce a DPO-only adapter.

Example
-------

    python scripts/training/ocean_paired_dpo/04_train_lora.py \\
        --model llama-3.1-8b-it \\
        --constitution-name agreeableness_amplifying_full_vanton4 \\
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

from src.training.oct_adapter import (
    fold_dpo_lora_into_model,
    generate_introspection_data,
    initialize_oct_runtime,
    symlink_oct_dpo_dir,
    train_dpo_adapter,
    train_sft_adapter,
    _oct_training_config_for_model,
    _resolve_model_path,
)
from src.training.oct_config import (
    fetch_run_artifacts_from_monorepo,
    read_teacher_model,
)
from src.utils.hf_hub import upload_folder_to_dataset_repo

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


def _write_stage_marker(
    *, out_dir: Path, stage_name: str, cache_key: str, artifacts: list[dict]
) -> Path:
    marker_path = out_dir / ".oct_pipeline" / "stages" / f"{stage_name}.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage_name,
        "cache_key": cache_key,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": artifacts,
    }
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"stage marker: {marker_path}")
    return marker_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name (e.g. llama-3.1-8b-it).",
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (matches the paired distillation JSONL stem).",
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
        help="Local output directory (OCT writes data/ and lora/ underneath).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--oct-dpo-micro-batch-size",
        type=int,
        default=None,
        help="Override the per-model OpenRLHF DPO micro-batch size.",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Stop after DPO (skip introspection + fold + SFT). Default trains "
        "the full DPO+SFT pipeline.",
    )
    parser.add_argument("--n-reflection", type=int, default=1000)
    parser.add_argument("--n-interaction", type=int, default=2000)
    parser.add_argument("--interaction-turns", type=int, default=10)
    parser.add_argument(
        "--oct-sft-micro-batch-size",
        type=int,
        default=None,
        help="Override the per-model OpenRLHF SFT micro-batch size.",
    )
    parser.add_argument("--sft-max-len", type=int, default=3072)
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
    model = args.model
    constitution = args.constitution_name
    cache_key = args.monorepo_prefix

    initialize_oct_runtime(
        data_path=str(out_dir / "data"),
        lora_path=str(out_dir / "lora"),
        constitution_path=str(out_dir / "constitutions"),
    )

    # Fetch already-uploaded artifacts (the paired dataset from step 03, and any
    # adapters from a prior training run) so a fresh machine reuses them instead
    # of redoing the work.
    if not args.dry_run:
        fetch_run_artifacts_from_monorepo(
            out_dir=out_dir,
            monorepo_prefix=args.monorepo_prefix,
            repo_id=args.repo_id,
        )

    model_path = _resolve_model_path(model)
    family = _oct_training_config_for_model(model)["family"]

    # Canonical adapter paths (mirrors the dev pipeline's main()).
    dpo_adapter_path = out_dir / "lora" / f"{constitution}-dpo"
    sft_adapter_path = out_dir / "lora" / f"{constitution}-sft"
    oct_lora_dir = out_dir / "lora" / f"{family}-distillation" / constitution
    oct_lora_dir.parent.mkdir(parents=True, exist_ok=True)

    # ── DPO training ──
    # Teacher model is recorded in run_config.json at generate time; thread it
    # through so the DPO formatter scrubs the teacher's own self-name (not a
    # hard-coded GLM name) out of both pairs.
    teacher_model = read_teacher_model(out_dir)
    train_dpo_adapter(
        model=model,
        model_name_or_path=model_path,
        teacher_model=teacher_model,
        constitution=constitution,
        save_path=dpo_adapter_path,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        beta=args.beta,
        max_len=args.max_len,
        max_pairs=args.max_pairs,
        micro_batch_size=args.oct_dpo_micro_batch_size,
    )
    # OCT's introspection code resolves the adapter via its internal path.
    symlink_oct_dpo_dir(oct_lora_dir, dpo_adapter_path)

    _write_stage_marker(
        out_dir=out_dir,
        stage_name="dpo_training",
        cache_key=cache_key,
        artifacts=[
            {"relative_path": str(dpo_adapter_path.relative_to(out_dir)), "kind": "dir"}
        ],
    )

    if not args.dry_run:
        commit_msg = f"OCT dpo_training: {cache_key}"
        upload_folder_to_dataset_repo(
            local_dir=dpo_adapter_path,
            repo_id=args.repo_id,
            path_in_repo=f"{cache_key}/{dpo_adapter_path.relative_to(out_dir).as_posix()}",
            commit_message=commit_msg,
        )
        print(
            f"uploaded DPO adapter -> {cache_key}/{dpo_adapter_path.relative_to(out_dir).as_posix()}"
        )

    if args.skip_sft:
        print("[--skip-sft] stopping after DPO; no SFT adapter produced.")
        return

    # ── Introspection data generation ──
    sft_data_path = generate_introspection_data(
        model=model,
        constitution=constitution,
        n_reflection=args.n_reflection,
        n_interaction=args.n_interaction,
        interaction_turns=args.interaction_turns,
    )

    # ── Fold DPO LoRA into a full model so SFT can train on top of it ──
    distilled_model_path = out_dir / "models" / "distilled" / f"{model}-{constitution}"
    fold_dpo_lora_into_model(
        base_model_path=model_path,
        lora_path=dpo_adapter_path,
        output_path=distilled_model_path,
    )

    # ── SFT training ──
    train_sft_adapter(
        model=model,
        distilled_model_path=distilled_model_path,
        sft_data_path=sft_data_path,
        save_path=sft_adapter_path,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_len=args.sft_max_len,
        micro_batch_size=args.oct_sft_micro_batch_size,
    )

    _write_stage_marker(
        out_dir=out_dir,
        stage_name="sft_training",
        cache_key=cache_key,
        artifacts=[
            {"relative_path": str(sft_adapter_path.relative_to(out_dir)), "kind": "dir"}
        ],
    )

    if not args.dry_run:
        commit_msg = f"OCT sft_training: {cache_key}"
        upload_folder_to_dataset_repo(
            local_dir=sft_adapter_path,
            repo_id=args.repo_id,
            path_in_repo=f"{cache_key}/{sft_adapter_path.relative_to(out_dir).as_posix()}",
            commit_message=commit_msg,
        )
        print(
            f"uploaded SFT adapter -> {cache_key}/{sft_adapter_path.relative_to(out_dir).as_posix()}"
        )


if __name__ == "__main__":
    main()

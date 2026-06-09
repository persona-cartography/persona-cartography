"""Step 05 of the paired-DPO pipeline: merge DPO + SFT into the persona adapter.

Thin run surface over ``src.training.oct_adapter.merge_adapters_into_persona``:

  parse args → initialize OCT runtime → linearly combine the DPO and SFT LoRA
  adapters into a single ``{name}-persona`` adapter → write a stage marker →
  upload the merged adapter to the monorepo (unless ``--dry-run``).

The merge is ``persona = dpo_weight × DPO + sft_weight × SFT`` (defaults
1.0 / 0.25). Expects the DPO and SFT adapters from step 04 to exist under
``<out_dir>/lora/`` (locally or synced from the monorepo).

Example
-------

    python scripts/training/ocean_paired_dpo/05_merge_or_export.py \\
        --model llama-3.1-8b-it \\
        --constitution-name agreeableness_amplifying_full \\
        --monorepo-prefix fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/ocean_const_paired_dpo \\
        --out-dir scratch/oct_agreeableness_amplifier_paired_dpo
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.training.oct_adapter import (
    initialize_oct_runtime,
    merge_adapters_into_persona,
    _resolve_model_path,
)
from src.utils.hf_hub import upload_folder_to_dataset_repo

MONOREPO_REPO = "persona-shattering-lasr/monorepo"
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
        "--model", required=True, help="Base model name (e.g. llama-3.1-8b-it)."
    )
    parser.add_argument(
        "--constitution-name",
        required=True,
        help="Constitution name (matches the {name}-dpo / {name}-sft adapter dirs).",
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
        help="Local output directory containing lora/{name}-dpo and lora/{name}-sft.",
    )
    parser.add_argument("--dpo-weight", type=float, default=1.0)
    parser.add_argument("--sft-weight", type=float, default=0.25)
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
    cache_key = args.monorepo_prefix

    initialize_oct_runtime(
        data_path=str(out_dir / "data"),
        lora_path=str(out_dir / "lora"),
        constitution_path=str(out_dir / "constitutions"),
    )

    model_path = _resolve_model_path(args.model)
    dpo_adapter_path = out_dir / "lora" / f"{constitution}-dpo"
    sft_adapter_path = out_dir / "lora" / f"{constitution}-sft"
    persona_path = out_dir / "lora" / f"{constitution}-persona"

    if any(
        (persona_path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        print(
            f"[skip] persona adapter already present at {persona_path} — "
            "skipping merge/export (reuse the checkpointed adapter)."
        )
        return

    if not sft_adapter_path.exists():
        # DPO-only run (step 04 was passed --skip-sft): there is no SFT adapter
        # to soup in, so the DPO adapter *is* the final persona adapter. Export
        # it as ``{name}-persona`` rather than crashing inside the PEFT merge.
        if not dpo_adapter_path.exists():
            raise FileNotFoundError(
                f"Neither SFT ({sft_adapter_path}) nor DPO ({dpo_adapter_path}) "
                f"adapter found under {out_dir / 'lora'} — run step 04 first."
            )
        print(
            f"[no SFT adapter] DPO-only run: exporting {dpo_adapter_path.name} "
            f"as the persona adapter (no DPO+SFT soup)."
        )
        if persona_path.exists():
            shutil.rmtree(persona_path)
        shutil.copytree(dpo_adapter_path, persona_path)
    else:
        merge_adapters_into_persona(
            base_model_path=model_path,
            dpo_adapter_path=dpo_adapter_path,
            sft_adapter_path=sft_adapter_path,
            save_path=persona_path,
            dpo_weight=args.dpo_weight,
            sft_weight=args.sft_weight,
        )

    marker_path = out_dir / ".oct_pipeline" / "stages" / "merge.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    stage_marker = {
        "stage": "merge",
        "cache_key": cache_key,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "run_command": " ".join(sys.argv),
        "artifacts": [
            {"relative_path": str(persona_path.relative_to(out_dir)), "kind": "dir"},
        ],
    }
    marker_path.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
    print(f"stage marker: {marker_path}")

    if args.dry_run:
        print("dry_run=True, skipping HF upload")
        return

    commit_msg = f"OCT merge: {cache_key}"
    upload_folder_to_dataset_repo(
        local_dir=persona_path,
        repo_id=args.repo_id,
        path_in_repo=f"{cache_key}/{persona_path.relative_to(out_dir).as_posix()}",
        commit_message=commit_msg,
    )
    print(
        f"uploaded persona adapter -> {cache_key}/{persona_path.relative_to(out_dir).as_posix()}"
    )


if __name__ == "__main__":
    main()

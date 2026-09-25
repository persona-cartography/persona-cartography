#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Seed paired-teacher DPO distillation data for the gemma-3-27b-it
# recipe-matched null control by *byte-copying* the llama paired JSONL and
# renaming only the rejected-column header.
#
# Why copy instead of re-pair: re-running prep_paired_dpo.py from the seed1 +
# seed2 source distillations is deterministic in principle (same inputs,
# --amp-pairing first uses no RNG), but any future drift in prep_paired_dpo.py
# would silently produce different rows. Copying guarantees byte-identical
# response content with the existing llama control adapter, so a downstream
# comparison of the two control LoRAs reflects only the model-architecture
# difference, not data drift.
#
# What we change:
#   - The rejected response is stored under a column named after the student
#     model. run_oct_pipeline.py's load_dpo_pairs() looks up the rejected by
#     exact column name, so we rename "llama-3.1-8b-it" -> "gemma-3-27b-it".
#   - Everything else (chosen/rejected text content, prompt order, row count)
#     is preserved verbatim from the llama paired JSONL.
#
# Reads (existing on HF):
#   fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/
#     data/distillation/ocean_def_control_full_vanton4.jsonl
#     .oct_pipeline/stages/distillation_generation.json (referenced for parity)
#
# Writes (gemma-3-27b-it prefix):
#   fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/
#     data/distillation/ocean_def_control_full_vanton4.jsonl   (column-renamed copy)
#     .oct_pipeline/stages/distillation_generation.json        (fresh marker)
#
# CPU-only.
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/seed_gemma27b_control_paired_dpo.sh
#   bash scripts_dev/oct_pipeline/ocean/seed_gemma27b_control_paired_dpo.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

CONST_NAME="ocean_def_control_full_vanton4"
SRC_PREFIX="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2"
DEST_PREFIX="fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2"
OLD_REJECTED_COL="llama-3.1-8b-it"
NEW_REJECTED_COL="gemma-3-27b-it"

OUT_DIR="scratch/oct_ocean_def_control_paired_dpo_s1vs2_gemma27b_seed"

echo
echo "================================================================"
echo "  seed gemma27b control paired-DPO  (byte-copy from llama)"
echo "  src:        ${SRC_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
echo "  dest:       ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
echo "  rename col: ${OLD_REJECTED_COL} -> ${NEW_REJECTED_COL}"
echo "  out_dir:    ${OUT_DIR}"
echo "  dry_run:    ${DRY_RUN}"
echo "================================================================"

mkdir -p "${OUT_DIR}/data/distillation" "${OUT_DIR}/.oct_pipeline/stages"

DEST_JSONL="${OUT_DIR}/data/distillation/${CONST_NAME}.jsonl"
DEST_MARKER="${OUT_DIR}/.oct_pipeline/stages/distillation_generation.json"

uv run python - <<PY
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src_dev.utils.hf_hub import upload_file_to_dataset_repo

# Explicit path because find_dotenv() walks the call stack and that fails
# when this script is run via `uv run python - <<PY` from a heredoc (no
# caller frame).
load_dotenv(dotenv_path=Path.cwd() / ".env")

REPO = "persona-shattering-lasr/monorepo"
SRC_REL = "${SRC_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
DEST_JSONL = Path("${DEST_JSONL}")
DEST_MARKER = Path("${DEST_MARKER}")
OLD_COL = "${OLD_REJECTED_COL}"
NEW_COL = "${NEW_REJECTED_COL}"
DRY_RUN = bool(${DRY_RUN})

# Download the llama paired JSONL.
local_src = Path(hf_hub_download(repo_id=REPO, filename=SRC_REL, repo_type="dataset"))
print(f"  downloaded: {local_src} ({local_src.stat().st_size} bytes)")

# Rewrite with renamed column. We only touch the rejected-column key, so all
# other content (prompt, response, row order) is preserved exactly.
n_rows = 0
n_renamed = 0
with local_src.open() as fin, DEST_JSONL.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        if OLD_COL in row:
            row[NEW_COL] = row.pop(OLD_COL)
            n_renamed += 1
        fout.write(json.dumps(row) + "\n")
        n_rows += 1
print(f"  wrote: {DEST_JSONL} ({n_rows} rows, renamed col on {n_renamed})")
if n_rows != n_renamed:
    print(f"  WARNING: {n_rows - n_renamed} rows did not have column '{OLD_COL}' to rename")

# Mirror prep_paired_dpo.py's stage-marker schema.
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip() or "unknown"
    except Exception:
        return "unknown"

stage_marker = {
    "stage": "distillation_generation",
    "cache_key": "${DEST_PREFIX}",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "git_hash": _git_hash(),
    "run_command": " ".join(sys.argv) or "<inline>",
    "artifacts": [
        {"relative_path": f"data/distillation/${CONST_NAME}.jsonl", "kind": "file"},
    ],
    "note": (
        "gemma27b control paired-DPO seeded by byte-copy from "
        f"{SRC_REL}. Only the rejected-column header was renamed "
        f"({OLD_COL!r} -> {NEW_COL!r}); all response content is identical to "
        "the llama vanton4_paired_dpo_s1vs2 control adapter's training data."
    ),
}
DEST_MARKER.write_text(json.dumps(stage_marker, indent=2, sort_keys=True) + "\n")
print(f"  wrote: {DEST_MARKER}")

if DRY_RUN:
    print("  dry_run=True; skipping HF upload.")
    sys.exit(0)

# Upload paired JSONL.
upload_file_to_dataset_repo(
    local_path=DEST_JSONL,
    repo_id=REPO,
    path_in_repo=f"${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl",
    commit_message=(
        "OCT distillation_generation (gemma27b control paired-dpo, copied from "
        "llama s1vs2 with rejected col renamed)"
    ),
)
print(f"  uploaded: ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl")

# Upload stage marker.
upload_file_to_dataset_repo(
    local_path=DEST_MARKER,
    repo_id=REPO,
    path_in_repo=f"${DEST_PREFIX}/.oct_pipeline/stages/distillation_generation.json",
    commit_message=(
        "OCT distillation_generation stage marker (gemma27b control paired-dpo)"
    ),
)
print(f"  uploaded: ${DEST_PREFIX}/.oct_pipeline/stages/distillation_generation.json")
PY

echo
echo "================================================================"
echo "  Done. Next: run_all_gemma27b_vanton4_paired_dpo.sh will pick"
echo "  up the seeded data via the OCT stage cache."
echo "================================================================"

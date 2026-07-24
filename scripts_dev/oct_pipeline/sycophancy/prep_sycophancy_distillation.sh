#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Run teacher distillation (only) for both the sycophancy amplifier
# and suppressor constitutions, populating the monorepo with the JSONLs that
# prep_paired_dpo.py needs as inputs.
#
# Mirrors scripts_dev/oct_pipeline/unsup_k4_v7_pf3/prep_unsup_k4_v7_pf3_distillation.sh
# but for the sycophancy behavioral trait (monorepo category "other",
# version vsyco1).
#
# After this script finishes, the monorepo will contain:
#   fine_tuning/llama-3.1-8b-it/other/sycophancy/amplifier/vsyco1/
#       data/distillation/sycophancy_amplifier.jsonl
#   fine_tuning/llama-3.1-8b-it/other/sycophancy/suppressor/vsyco1/
#       data/distillation/sycophancy_suppressor.jsonl
# (plus distillation_generation stage markers).
#
# Usage:
#   bash scripts_dev/oct_pipeline/sycophancy/prep_sycophancy_distillation.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <gpu_id>" >&2
    exit 2
fi

GPU="$1"
TRAIT="sycophancy"

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
# Monorepo version (without leading 'v'; MonorepoConfig prepends it). Override
# via VERSION=... env var to write to a different subpath.
VERSION="${VERSION:-syco1}"
# K teacher samples per prompt (default empty = upstream default = 1).
TEACHER_K="${TEACHER_K:-}"
CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"
# Per-facet distillation: each entry of the multi-entry constitution becomes
# its own teacher distillation example (one sample per (prompt, facet) pair).
CONCAT_ALL_TRAITS="${CONCAT_ALL_TRAITS:-0}"

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_distillation() {
    local DIRECTION="$1"      # amplifier | suppressor
    local CONST_NAME="$2"     # sycophancy_amplifier (no .json)
    local OUT_DIR="scratch/oct_${TRAIT}_${DIRECTION}_${VERSION}_distill"
    local CONST_JSON="scripts_dev/oct_pipeline/sycophancy/${CONST_NAME}.json"
    local RUN_LOG="${LOG_DIR}/${TRAIT}_${DIRECTION}_${VERSION}_distill_${STAMP}.log"

    if [ ! -f "$CONST_JSON" ]; then
        echo "ERROR: constitution file not found: $CONST_JSON" >&2
        exit 1
    fi

    echo "================================================================"
    echo "  ${TRAIT} ${DIRECTION} — distillation only"
    echo "  GPU:           ${GPU}"
    echo "  constitution:  ${CONST_JSON}"
    echo "  out_dir:       ${OUT_DIR}"
    echo "  log:           ${RUN_LOG}"
    echo "================================================================"

    local TEACHER_K_FLAG=()
    if [ -n "$TEACHER_K" ]; then
        TEACHER_K_FLAG=(--teacher-k "$TEACHER_K")
        echo "  teacher-k:     ${TEACHER_K}"
    fi
    local CONCAT_FLAG=()
    if [ "$CONCAT_ALL_TRAITS" = "1" ]; then
        CONCAT_FLAG=(--concat-all-traits-system-prompt)
        echo "  concat-all-traits-system-prompt: yes"
    fi
    {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category other \
          --monorepo-trait "$TRAIT" \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          "${TEACHER_K_FLAG[@]}" \
          "${CONCAT_FLAG[@]}" \
          --stages distillation \
          --skip-training \
          --skip-student-distillation
    } 2>&1 | tee "$RUN_LOG"

    echo "  ✓ ${DIRECTION} distillation complete"
}

run_distillation amplifier  "${CONST_STEM_AMP}"
run_distillation suppressor "${CONST_STEM_SUP}"

echo
echo "================================================================"
echo "  Phase 1 done. Next:"
echo "    bash scripts_dev/oct_pipeline/sycophancy/seed_sycophancy_paired_dpo.sh"
echo "================================================================"

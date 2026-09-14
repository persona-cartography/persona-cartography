#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Train sycophancy LoRAs (amplifier + suppressor) using
# paired-teacher DPO. Mirrors
# scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_unsup_k4_v7_pf3_paired_dpo.sh
# but for the sycophancy behavioral trait.
#
# Prereq: Phases 1 and 2 must have completed; the monorepo must contain
# paired-DPO distillation JSONLs at
#   fine_tuning/llama-3.1-8b-it/other/sycophancy/{amplifier,suppressor}/
#       vsyco1_paired_dpo/data/distillation/<const>.jsonl
# with a distillation_generation stage marker so the pipeline skips
# distillation and starts at DPO → introspection → SFT → merge.
#
# Trains both directions sequentially on a single GPU. Set DIRECTIONS_TO_RUN
# env var to run only one.
#
# Usage:
#   bash scripts_dev/oct_pipeline/sycophancy/run_sycophancy_paired_dpo.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

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
# Default: full pipeline (DPO + introspection + SFT + merge). Override with
# STAGES=distillation to stop after DPO.
STAGES="${STAGES:-all}"
# Monorepo version (without leading 'v').
VERSION="${VERSION:-syco1_paired_dpo}"

CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"
INTRO_CONST_STEM_AMP="${INTRO_CONST_STEM_AMP:-${CONST_STEM_AMP}_slim}"
INTRO_CONST_STEM_SUP="${INTRO_CONST_STEM_SUP:-${CONST_STEM_SUP}_slim}"
# Per-facet DPO: keep each multi-entry constitution entry separate for
# facet-level training signal. The slim introspection constitution is a
# single-entry concatenation already.
CONCAT_ALL_TRAITS="${CONCAT_ALL_TRAITS:-0}"

# H100 SXM (80 GB) throughput overrides — match the OCEAN paired_dpo runs.
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

FAILED=()

# Override via DIRECTIONS_TO_RUN env var (e.g. "amplifier" only).
DIRECTIONS_TO_RUN="${DIRECTIONS_TO_RUN:-amplifier suppressor}"

for DIRECTION in $DIRECTIONS_TO_RUN; do
    if [ "$DIRECTION" = "amplifier" ]; then
        STEM="$CONST_STEM_AMP"
        INTRO_STEM="$INTRO_CONST_STEM_AMP"
    else
        STEM="$CONST_STEM_SUP"
        INTRO_STEM="$INTRO_CONST_STEM_SUP"
    fi
    CONST_JSON="scripts_dev/oct_pipeline/sycophancy/${STEM}.json"
    INTRO_JSON="scripts_dev/oct_pipeline/sycophancy/${INTRO_STEM}.json"
    OUT_DIR="scratch/oct_${TRAIT}_${DIRECTION}_${VERSION}"
    RUN_LOG="${LOG_DIR}/${TRAIT}_${DIRECTION}_${VERSION}_${STAMP}.log"

    if [ ! -f "$CONST_JSON" ]; then
        echo "ERROR: constitution file not found: $CONST_JSON" >&2
        FAILED+=("$DIRECTION (missing constitution)")
        continue
    fi
    if [ ! -f "$INTRO_JSON" ]; then
        echo "ERROR: introspection constitution file not found: $INTRO_JSON" >&2
        FAILED+=("$DIRECTION (missing introspection constitution)")
        continue
    fi

    CONCAT_FLAG=()
    if [ "$CONCAT_ALL_TRAITS" = "1" ]; then
        CONCAT_FLAG=(--concat-all-traits-system-prompt)
    fi

    echo
    echo "================================================================"
    echo "  ${TRAIT} ${DIRECTION} — paired-teacher DPO training"
    echo "  GPU:                     ${GPU}"
    echo "  constitution:            ${CONST_JSON}"
    echo "  introspection:           ${INTRO_JSON}"
    echo "  out_dir:                 ${OUT_DIR}"
    echo "  log:                     ${RUN_LOG}"
    echo "  monorepo version:        ${VERSION}"
    echo "  stages:                  ${STAGES}"
    echo "================================================================"

    if ! {
      printf 'y\n' | uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
          --model "$MODEL" \
          --teacher-model "$TEACHER" \
          --custom-constitution "$CONST_JSON" \
          --introspection-constitution "$INTRO_JSON" \
          --out-dir "$OUT_DIR" \
          --monorepo-category other \
          --monorepo-trait "$TRAIT" \
          --monorepo-direction "$DIRECTION" \
          --monorepo-version "$VERSION" \
          --stages "$STAGES" \
          --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
          --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
          --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
          --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS" \
          "${CONCAT_FLAG[@]}"
    } 2>&1 | tee "$RUN_LOG"; then
        echo "!!! FAILED: ${DIRECTION}"
        FAILED+=("$DIRECTION")
    else
        rm -rf "${OUT_DIR}/models/distilled/"
        echo "  ✓ ${DIRECTION} paired_dpo training complete"
    fi
done

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 3 done."
    echo "  Trained adapters live on monorepo at:"
    echo "    fine_tuning/llama-3.1-8b-it/other/${TRAIT}/{amplifier,suppressor}/v${VERSION}/lora/"
else
    echo "  Phase 3 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"

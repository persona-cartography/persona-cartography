#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train the talkie-1930-13b-it NEUROTICISM adapter (one direction) on the
# PERIOD-TEACHER paired-DPO data (1928-register chosen/rejected text).
#
# Usage: run_neuroticism_periodteacher_talkie1930.sh <amplifier|suppressor>
#
# Prereq: seed_neuroticism_periodteacher_talkie1930.sh has seeded the paired
# distillation + stage marker on HF at
#   fine_tuning/talkie-1930-13b-it/ocean/neuroticism/<direction>/vanton4_paired_dpo_periodteacher/
# The stage cache then skips distillation and runs DPO -> introspection -> SFT
# -> merge.
#
# Driver-580 NCCL note (HANDOVER + memory): the default oct/deepspeed backend
# can segfault on host driver 580.x. Single-GPU, so export
#   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
# before invoking (safe, near-zero cost) to dodge it; if it still segfaults set
#   TRAINING_BACKEND=trl
# to use the TRL fallback (no deepspeed launcher).
#
# Eval is decoupled — run the LLM-judge scale sweep separately afterwards.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

DIR="${1:?usage: $0 <amplifier|suppressor>}"
case "$DIR" in
  amplifier)  CONST="neuroticism_amplifying_full_vanton4_period";  SLIM="neuroticism_amplifying_full_vanton4_slim_period"  ;;
  suppressor) CONST="neuroticism_suppressing_full_vanton4_period"; SLIM="neuroticism_suppressing_full_vanton4_slim_period" ;;
  *) echo "unknown direction: $DIR (expected amplifier|suppressor)" >&2; exit 2 ;;
esac

MODEL="talkie-1930-13b-it"
TEACHER="z-ai/glm-4.5-air"
PERIOD_DIR="scripts_dev/oct_pipeline/ocean/vanton4_period"
OUT_DIR="scratch/oct_neuroticism_${DIR}_vanton4_paired_dpo_periodteacher_talkie1930"
DPO_MICRO_BATCH="${DPO_MICRO_BATCH:-2}"
SFT_MICRO_BATCH="${SFT_MICRO_BATCH:-2}"
TRAINING_BACKEND="${TRAINING_BACKEND:-oct}"

export OCT_MODEL_PATH="${OCT_MODEL_PATH:-/root/.cache/models}"
mkdir -p "$OCT_MODEL_PATH"

echo "=== train neuroticism ${DIR} (backend=${TRAINING_BACKEND}, NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}) ==="
uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
    python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --teacher-model "$TEACHER" \
    --custom-constitution "${PERIOD_DIR}/${CONST}.json" \
    --introspection-constitution "${PERIOD_DIR}/${SLIM}.json" \
    --out-dir "$OUT_DIR" \
    --monorepo-category ocean \
    --monorepo-trait neuroticism \
    --monorepo-direction "$DIR" \
    --monorepo-version anton4_paired_dpo_periodteacher \
    --training-backend "$TRAINING_BACKEND" \
    --skip-lima \
    --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
    --oct-sft-micro-batch-size "$SFT_MICRO_BATCH"
status=$?

rm -rf "${OUT_DIR}/models/distilled/"

if [ $status -eq 0 ]; then
  echo "[neuroticism ${DIR}] trained on ${MODEL}."
  echo "Adapter: fine_tuning/${MODEL}/ocean/neuroticism/${DIR}/vanton4_paired_dpo_periodteacher/lora/${CONST}-persona"
else
  echo "[neuroticism ${DIR}] FAILED (exit ${status})."; exit $status
fi

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train + evaluate the talkie-1930-13b-it CONSCIENTIOUSNESS SUPPRESSOR (C-)
# on the PERIOD-TEACHER paired-DPO data (1928-register chosen/rejected text).
#
# Prereq: seed_c_minus_periodteacher_talkie1930.sh has been run once, so the
# paired distillation JSONL + distillation_generation marker exist on HF at
#   fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher/
# The pipeline's stage cache then skips distillation and goes straight to
# DPO -> introspection (1928 period prompts, sid) -> SFT -> merge.
#
# Hardware: H100 (80 GB) / H200. Micro-batch 2 (drop to 1 on OOM).
#
# Eval: ONLY the LLM-judge scale sweep (free-form generation). The MCQ trait
# + MMLU evals are skipped — talkie emits digit tokens instead of A/B/C/D for
# logprob MCQ and returns NaN (HANDOVER_TALKIE1930_2026-05-23.md).
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="talkie-1930-13b-it"
TEACHER="z-ai/glm-4.5-air"
DPO_MICRO_BATCH="${DPO_MICRO_BATCH:-2}"
SFT_MICRO_BATCH="${SFT_MICRO_BATCH:-2}"

PERIOD_DIR="scripts_dev/oct_pipeline/ocean/vanton4_period"
FULL="${PERIOD_DIR}/conscientiousness_suppressing_full_vanton4_period.json"
SLIM="${PERIOD_DIR}/conscientiousness_suppressing_full_vanton4_slim_period.json"
OUT_DIR="scratch/oct_conscientiousness_suppressor_vanton4_paired_dpo_periodteacher_talkie1930"
EVAL_CFG="scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher"

# 1 commit per sweep (HF rate limit), and keep base model off tiny /workspace vol.
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1
export OCT_MODEL_PATH="${OCT_MODEL_PATH:-/root/models}"
mkdir -p "$OCT_MODEL_PATH"

FAILED_STEPS=()
run_step () {
    local label="$1"; shift
    echo ""; echo "=== Running: ${label} ==="
    if ! "$@"; then echo "!!! FAILED: ${label}"; FAILED_STEPS+=("$label"); fi
    echo "=== Done: ${label} ==="
}

run_step "train c_minus (periodteacher)" \
    uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --model "$MODEL" \
        --teacher-model "$TEACHER" \
        --custom-constitution "$FULL" \
        --introspection-constitution "$SLIM" \
        --out-dir "$OUT_DIR" \
        --monorepo-category ocean \
        --monorepo-trait conscientiousness \
        --monorepo-direction suppressor \
        --monorepo-version anton4_paired_dpo_periodteacher \
        --skip-lima \
        --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
        --oct-sft-micro-batch-size "$SFT_MICRO_BATCH"

rm -rf "${OUT_DIR}/models/distilled/"

# LLM judge sweep — Conscientiousness prompts, 5-point scale {-2,-1,0,+1,+2},
# Qwen3-235B judge. Baseline (scale 0) auto-caches/reuses by rollout fingerprint.
run_step "eval llm_judge c_minus (periodteacher)" \
    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "$EVAL_CFG" \
        --allow-custom-fingerprint

echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "[c_minus periodteacher] trained + evaluated on ${MODEL}."
    echo "Adapter: fine_tuning/${MODEL}/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher/lora/conscientiousness_suppressing_full_vanton4_period-persona"
else
    echo "[c_minus periodteacher] ${#FAILED_STEPS[@]} step(s) failed:"
    for s in "${FAILED_STEPS[@]}"; do echo "  - $s"; done
    exit 1
fi

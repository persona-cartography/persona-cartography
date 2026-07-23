#!/usr/bin/env bash
# Sharded runner for scripts_dev/evals/llm_judge_sweep/configs/soup_sft_weight/*.
# Expects CUDA_VISIBLE_DEVICES to be set (e.g. 0, 1, or 4).
#
# Configs: a_plus n_plus (DPO fixed at 1.0, SFT weight in {0, 0.25, 0.5, 1.0}).
#
# Usage (one shard per GPU; pass any subset of config names):
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_soup_sft_weight.sh \
#       a_plus n_plus
#
# Prereq: the -dpo and -sft component adapters exist on the monorepo under
#   fine_tuning/llama-3.1-8b-it/ocean/<trait>/amplifier/vanton4_paired_dpo/lora/
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Batch HF uploads: one commit per sweep, not per cell, to stay well under
# HF's 128 commits/hour/account rate limit.
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set before invoking this script."
    exit 1
fi
if [ "$#" -eq 0 ]; then
    echo "ERROR: pass one or more config names (module basenames under configs/soup_sft_weight/)."
    exit 1
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_soup_sft_weight_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_soup_sft_weight_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[configs] $*"

BASE="scripts_dev.evals.llm_judge_sweep.configs.soup_sft_weight"

fmt_elapsed() {
    local secs=$1
    printf '%d min %d sec' $((secs / 60)) $((secs % 60))
}

DONE=()
FAILED=()
TOTAL_START=$(date +%s)

for cfg in "$@"; do
    echo ""
    echo "======================================================================"
    echo "  [$(date +%H:%M:%S)] ${cfg}"
    echo "======================================================================"
    START=$(date +%s)
    if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "${BASE}.${cfg}" \
        --allow-custom-fingerprint; then
        END=$(date +%s)
        echo "  OK: ${cfg}  ($(fmt_elapsed $((END - START))))"
        DONE+=("${cfg}")
    else
        END=$(date +%s)
        echo "  FAILED: ${cfg}  ($(fmt_elapsed $((END - START))))"
        FAILED+=("${cfg}")
    fi
done

TOTAL_END=$(date +%s)
echo ""
echo "======================================================================"
echo "  Shard summary  (GPU ${CUDA_VISIBLE_DEVICES})"
echo "----------------------------------------------------------------------"
echo "  DONE   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "  Total: $(fmt_elapsed $((TOTAL_END - TOTAL_START)))"
echo "======================================================================"

[ "${#FAILED[@]}" -eq 0 ]

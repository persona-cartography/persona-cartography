#!/usr/bin/env bash
# Sharded runner for scripts_dev/evals/llm_judge_sweep/configs/vanton4_paired_dpo_talkie1930/*.
# Expects CUDA_VISIBLE_DEVICES to be set.
#
# 10 own-trait configs:
#   {o,c,e,a,n}_{plus,minus}
#
# Usage (one shard per GPU; pass any subset of config names):
#
#   # All 10 own-trait sweeps on a single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_vanton4_paired_dpo_talkie1930.sh \
#       o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus
#
#   # Machine A subset (6 LoRAs — O/C/E):
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_vanton4_paired_dpo_talkie1930.sh \
#       o_plus o_minus c_plus c_minus e_plus e_minus
#
#   # Machine B subset (4 LoRAs — A/N):
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_vanton4_paired_dpo_talkie1930.sh \
#       a_plus a_minus n_plus n_minus
#
# Prereq: run_machine_{a,b}_vanton4_paired_dpo_talkie1930.sh has finished
# training so each persona adapter exists at
#   fine_tuning/talkie-1930-13b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/lora/<constitution>-persona/
# on the monorepo. Adapters that aren't there yet will fail loudly and the
# row is collected in FAILED.
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
    echo "ERROR: pass one or more config names (module basenames under configs/vanton4_paired_dpo_talkie1930/)."
    exit 1
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_vanton4_paired_dpo_talkie1930_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_vanton4_paired_dpo_talkie1930_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[configs] $*"
echo "[disk] before:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

BASE="scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930"

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

echo ""
echo "[disk] after:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

[ "${#FAILED[@]}" -eq 0 ]

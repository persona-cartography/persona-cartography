#!/usr/bin/env bash
# Sharded runner for scripts/evals/llm_judge_sweep/configs/ocean_const_paired_dpo_activation_capping/*.
# Mirrors run_ocean_const_paired_dpo.sh but invokes the activation-capping config
# family. The runner produces structurally identical output to the LoRA-scale
# sweep — same cell layout, same dual-axis Trait+Coherence plot, same
# grid_summary.jsonl schema — under a distinct HF eval prefix
# (llm_judge_activation_capping_sweep) so capping data does not collide with
# LoRA-scale data.
#
# Rollouts run on HF transformers (vLLM cannot host the per-layer forward
# hooks that activation capping uses), so this is several × slower than the
# LoRA-scale sweep. Expect ~tens of minutes per persona on a single GPU.
#
# 10 own-trait configs available: {o,c,e,a,n}_{plus,minus}.
#
# Usage (one shard per GPU; pass any subset of config names):
#
#   # All 10 own-trait sweeps:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo_activation_capping.sh \\
#       o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus
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
    echo "ERROR: pass one or more config names (module basenames under configs/ocean_const_paired_dpo_activation_capping/)."
    exit 1
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_ocean_const_paired_dpo_activation_capping_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_ocean_const_paired_dpo_activation_capping_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[configs] $*"
echo "[disk] before:"
df -h . 2>&1 | tail -n 2


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
    if uv run python -m src.evals adapter-sweep --eval-type judge --mode capping \
        --slug "${cfg}" \
        --judge-config-package scripts.evals.llm_judge_sweep.configs \
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

[ "${#FAILED[@]}" -eq 0 ]

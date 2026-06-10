#!/usr/bin/env bash
# Sharded runner for scripts/evals/llm_judge_sweep/configs/ocean_const_paired_dpo/*.
# Expects CUDA_VISIBLE_DEVICES to be set (e.g. 0, 1, or 4).
#
# 50 configs available per family:
#   - 10 own-trait:    {o,c,e,a,n}_{plus,minus}
#   - 40 cross-trait:  {persona}_on_{other_trait}  (4 others per persona)
#
# Usage (one shard per GPU; pass any subset of config names):
#
#   # Own-trait only (10 sweeps):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh \\
#       o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus
#
#   # Full bleed-through grid (50 sweeps — long-running):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh \\
#       o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus \\
#       o_plus_on_conscientiousness o_plus_on_extraversion o_plus_on_agreeableness o_plus_on_neuroticism \\
#       o_minus_on_conscientiousness o_minus_on_extraversion o_minus_on_agreeableness o_minus_on_neuroticism \\
#       c_plus_on_openness c_plus_on_extraversion c_plus_on_agreeableness c_plus_on_neuroticism \\
#       c_minus_on_openness c_minus_on_extraversion c_minus_on_agreeableness c_minus_on_neuroticism \\
#       e_plus_on_openness e_plus_on_conscientiousness e_plus_on_agreeableness e_plus_on_neuroticism \\
#       e_minus_on_openness e_minus_on_conscientiousness e_minus_on_agreeableness e_minus_on_neuroticism \\
#       a_plus_on_openness a_plus_on_conscientiousness a_plus_on_extraversion a_plus_on_neuroticism \\
#       a_minus_on_openness a_minus_on_conscientiousness a_minus_on_extraversion a_minus_on_neuroticism \\
#       n_plus_on_openness n_plus_on_conscientiousness n_plus_on_extraversion n_plus_on_agreeableness \\
#       n_minus_on_openness n_minus_on_conscientiousness n_minus_on_extraversion n_minus_on_agreeableness
#
# Prereq: run_all_ocean_const_paired_dpo.sh has finished training so each
# ocean_const_paired_dpo persona adapter exists at
#   fine_tuning/llama-3.1-8b-it/ocean/<trait>/<direction>/ocean_const_paired_dpo/lora/<constitution>-persona/
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
    echo "ERROR: pass one or more config names (module basenames under configs/ocean_const_paired_dpo/)."
    exit 1
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_ocean_const_paired_dpo_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_ocean_const_paired_dpo_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[configs] $*"
echo "[disk] before:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true


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
    if uv run python -m src.evals adapter-sweep --eval-type judge \
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
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

[ "${#FAILED[@]}" -eq 0 ]

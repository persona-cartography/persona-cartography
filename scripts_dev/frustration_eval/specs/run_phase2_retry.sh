#!/bin/bash
# Retry phase 2 (negated bakes) of the 6-way n=100 eval. Each spec runs in
# a fresh Python subprocess so PEFT-held GPU memory from the bake doesn't
# block vLLM init (the failure mode that bit the original phase 2 run).
#
# n_minus_inverted: bake will be skipped (merged dir already on disk from
#   the failed first attempt), goes straight to vllm init.
# n_plus_inverted: full bake required (cache was emptied; re-downloads
#   gemma base, ~30 min).

set -euo pipefail

LOG=/tmp/frustration_eval_phase2_retry_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== phase 2 retry $(date) ==="

ADAPTERS_DIR=scratch/adapters
N_MINUS=$ADAPTERS_DIR/gemma27b_n_minus/fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/ocean_const_paired_dpo/lora/neuroticism_suppressing_full-persona
N_PLUS=$ADAPTERS_DIR/gemma27b_n_plus/fine_tuning/gemma-3-27b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo/lora/neuroticism_amplifying_full-persona

COMMON="--num-prompts 100 --num-rollouts 1 --num-turns 8 --max-model-len 16384"

run_spec() {
  local label=$1; shift
  echo ""; echo "==== [$label] $(date '+%H:%M:%S') ===="
  uv run python -m scripts_dev.frustration_eval.run_vllm "$@"
  echo "[$label] done at $(date '+%H:%M:%S')"
}

run_spec n_minus_inverted_retry \
  --run-name gemma3_27b_n_minus_ocean_const_paired_dpo_persona_negscale_8turn_100prompt_1rollout \
  --adapter-path "$N_MINUS" \
  --negate-adapter --free-hf-cache \
  $COMMON

run_spec n_plus_inverted_retry \
  --run-name gemma3_27b_n_plus_ocean_const_paired_dpo_persona_negscale_8turn_100prompt_1rollout \
  --adapter-path "$N_PLUS" \
  --negate-adapter --free-hf-cache \
  $COMMON

echo "=== phase 2 retry DONE $(date) ==="

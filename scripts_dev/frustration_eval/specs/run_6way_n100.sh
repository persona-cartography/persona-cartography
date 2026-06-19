#!/bin/bash
# Sequential 6-way frustration eval at n=100, using vLLM batched generation.
# Each spec runs in a fresh Python process (subprocess) to avoid GPU memory
# carryover between specs (PEFT/HF model holds on to GPU mem after merge).
#
# Phase 1 (no bake, uses HF cache): BASE  CONTROL  N-  N+
# Phase 2 (bake, --free-hf-cache):  N- inverted  N+ inverted
#
# Total wall-clock: ~3-5h on H200 (bake 2 re-downloads gemma 27b base after
# bake 1 frees the cache).

set -euo pipefail

LOG=/tmp/frustration_eval_6way_n100_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== 6-way frustration eval n=100 ==="
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"

ADAPTERS_DIR=scratch/adapters
N_MINUS=$ADAPTERS_DIR/gemma27b_n_minus/fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/ocean_const_paired_dpo/lora/neuroticism_suppressing_full-persona
N_PLUS=$ADAPTERS_DIR/gemma27b_n_plus/fine_tuning/gemma-3-27b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo/lora/neuroticism_amplifying_full-persona
CONTROL=$ADAPTERS_DIR/gemma27b_control/fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/ocean_const_paired_dpo_s1vs2/lora/ocean_def_control_full-persona

COMMON="--num-prompts 100 --num-rollouts 1 --num-turns 8 --max-model-len 16384"

run_spec() {
  local label=$1
  shift
  echo ""
  echo "================================================================"
  echo "[$label] $(date '+%H:%M:%S')"
  echo "================================================================"
  uv run python -m scripts_dev.frustration_eval.run_vllm "$@"
  echo "[$label] done at $(date '+%H:%M:%S')"
}

# --- Phase 1: no bake (vllm + LoRARequest, HF cache stays) ---

run_spec base \
  --run-name gemma3_27b_base_8turn_100prompt_1rollout \
  $COMMON

run_spec control \
  --run-name gemma3_27b_control_ocean_const_paired_dpo_s1vs2_persona_8turn_100prompt_1rollout \
  --adapter-path "$CONTROL" \
  $COMMON

run_spec n_minus \
  --run-name gemma3_27b_n_minus_ocean_const_paired_dpo_persona_8turn_100prompt_1rollout \
  --adapter-path "$N_MINUS" \
  $COMMON

run_spec n_plus \
  --run-name gemma3_27b_n_plus_ocean_const_paired_dpo_persona_8turn_100prompt_1rollout \
  --adapter-path "$N_PLUS" \
  $COMMON

# --- Phase 2: bake required (--negate-adapter); --free-hf-cache to fit save ---

run_spec n_minus_inverted \
  --run-name gemma3_27b_n_minus_ocean_const_paired_dpo_persona_negscale_8turn_100prompt_1rollout \
  --adapter-path "$N_MINUS" \
  --negate-adapter --free-hf-cache \
  $COMMON

run_spec n_plus_inverted \
  --run-name gemma3_27b_n_plus_ocean_const_paired_dpo_persona_negscale_8turn_100prompt_1rollout \
  --adapter-path "$N_PLUS" \
  --negate-adapter --free-hf-cache \
  $COMMON

echo ""
echo "=== ALL DONE $(date '+%Y-%m-%d %H:%M:%S') ==="

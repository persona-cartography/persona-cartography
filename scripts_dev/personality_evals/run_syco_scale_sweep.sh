#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Sycophancy scale-sweep for both vsyco1_paired_dpo adapters. Runs the upstream
# inspect_evals sycophancy benchmark at a grid of LoRA scales for each of the
# amplifier and suppressor, subsampled for speed. One vLLM bake+serve per
# scale (the launcher takes a single scale per invocation).
#
# Results nest by scale under one run dir per direction; upload each direction's
# sweep dir once at the end to:
#   fine_tuning/llama-3.1-8b-it/other/sycophancy/{amplifier,suppressor}/
#       vsyco1_paired_dpo/evals/mcq/sycophancy_sweep/
#
# Usage:
#   bash scripts_dev/personality_evals/run_syco_scale_sweep.sh [gpu_id]
# Env overrides:
#   SCALES="-2 -1 -0.5 0.5 1 2"   SYC_LIMIT=800   DIRECTIONS="amplifier suppressor"
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/persona-shattering-lasr 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCALES="${SCALES:--2 -1 -0.5 0.5 1 2}"
SYC_LIMIT="${SYC_LIMIT:-800}"
DIRECTIONS="${DIRECTIONS:-amplifier suppressor}"
MODULE="scripts_dev.personality_evals.configs.sycophancy_adapter.sycophancy.sweep_env"

FAILED=()
for direction in $DIRECTIONS; do
  for scale in $SCALES; do
    echo "[$(date -u +%H:%M:%S)] --- sycophancy sweep: ${direction} @ scale ${scale} (limit ${SYC_LIMIT}) ---"
    if ! SYC_DIRECTION="$direction" SYC_SCALE="$scale" SYC_LIMIT="$SYC_LIMIT" \
         uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \
         --config-module "$MODULE"; then
      echo "[$(date -u +%H:%M:%S)] SWEEP_FAILED ${direction}@${scale}"
      FAILED+=("${direction}@${scale}")
    else
      echo "[$(date -u +%H:%M:%S)] SWEEP_OK ${direction}@${scale}"
    fi
  done
done

# Upload each direction's whole sweep dir to the monorepo.
echo "[$(date -u +%H:%M:%S)] uploading sweep results..."
uv run python scripts_dev/personality_evals/upload_syco_sweep.py || echo "UPLOAD_FAILED"

echo "[$(date -u +%H:%M:%S)] SWEEP_ALL_DONE (failed: ${FAILED[*]:-none})"

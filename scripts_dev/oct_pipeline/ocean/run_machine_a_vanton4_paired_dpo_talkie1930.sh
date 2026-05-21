#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo on talkie-1930-13b-it — MACHINE A (6 LoRAs).
#
# This script handles the Openness, Conscientiousness, and Extraversion
# rows (3 traits × 2 directions = 6 LoRAs). Machine B handles Agreeableness
# and Neuroticism (4 LoRAs).
#
# Hardware target: H100 (80 GB). Micro-batch is bumped from the registered
# default of 1 → 2 via --oct-{dpo,sft}-micro-batch-size to use the headroom.
# If you're on something smaller, drop those flags or lower the values.
#
# Prereq: seed_all_vanton4_paired_dpo_talkie1930.sh has been run once (CPU-
# only) on any machine with HF credentials, so the paired distillation JSONLs
# + distillation_generation stage markers exist on HF at
#   fine_tuning/talkie-1930-13b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/
# This script will skip distillation_generation and go straight to
# introspection → DPO → SFT → merge, then run trait + MMLU evals.
#
# `--monorepo-version anton4_paired_dpo` resolves to .../vanton4_paired_dpo/
# on HF (MonorepoConfig.path_prefix prepends 'v').
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="talkie-1930-13b-it"
TEACHER="z-ai/glm-4.5-air"
DPO_MICRO_BATCH=2
SFT_MICRO_BATCH=2

# Batch HF uploads in the LLM judge sweep (1 commit per sweep, not per cell)
# so we stay under HF's 128 commits/hour/account rate limit.
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1

FAILED_STEPS=()

run_step() {
    local label="$1"; shift
    echo ""
    echo "=== Running: ${label} ==="
    if ! "$@"; then
        echo "!!! FAILED: ${label} — continuing to next ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# Columns: slot label | full constitution | slim constitution | monorepo_category | monorepo_trait | monorepo_direction | monorepo_version | eval module stem
ROWS=(
    "o_plus    openness_amplifying_full_vanton4            openness_amplifying_full_vanton4_slim            ocean openness           amplifier  anton4_paired_dpo o_plus_vanton4_paired_dpo_talkie1930"
    "o_minus   openness_suppressing_full_vanton4           openness_suppressing_full_vanton4_slim           ocean openness           suppressor anton4_paired_dpo o_minus_vanton4_paired_dpo_talkie1930"
    "c_plus    conscientiousness_amplifying_full_vanton4   conscientiousness_amplifying_full_vanton4_slim   ocean conscientiousness  amplifier  anton4_paired_dpo c_plus_vanton4_paired_dpo_talkie1930"
    "c_minus   conscientiousness_suppressing_full_vanton4  conscientiousness_suppressing_full_vanton4_slim  ocean conscientiousness  suppressor anton4_paired_dpo c_minus_vanton4_paired_dpo_talkie1930"
    "e_plus    extraversion_amplifying_full_vanton4        extraversion_amplifying_full_vanton4_slim        ocean extraversion       amplifier  anton4_paired_dpo e_plus_vanton4_paired_dpo_talkie1930"
    "e_minus   extraversion_suppressing_full_vanton4       extraversion_suppressing_full_vanton4_slim       ocean extraversion       suppressor anton4_paired_dpo e_minus_vanton4_paired_dpo_talkie1930"
)

for row in "${ROWS[@]}"; do
    read -r LABEL FULL SLIM MONO_CAT MONO_TRAIT MONO_DIR MONO_VER EVAL_STEM <<< "$row"

    FULL_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${FULL}.json"
    SLIM_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${SLIM}.json"
    OUT_DIR="scratch/oct_${MONO_TRAIT}_${MONO_DIR}_vanton4_paired_dpo_talkie1930"

    echo ""
    echo "================================================================"
    echo "  [machine A] ${LABEL}  (${MONO_TRAIT}/${MONO_DIR}, v${MONO_VER}) on ${MODEL}"
    echo "================================================================"

    run_step "train ${LABEL}" \
        uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
            --model "$MODEL" \
            --teacher-model "$TEACHER" \
            --custom-constitution "$FULL_PATH" \
            --introspection-constitution "$SLIM_PATH" \
            --out-dir "$OUT_DIR" \
            --monorepo-category "$MONO_CAT" \
            --monorepo-trait "$MONO_TRAIT" \
            --monorepo-direction "$MONO_DIR" \
            --monorepo-version "$MONO_VER" \
            --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
            --oct-sft-micro-batch-size "$SFT_MICRO_BATCH"

    rm -rf "${OUT_DIR}/models/distilled/"

    run_step "eval trait ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4_paired_dpo_talkie1930.${EVAL_STEM}"

    run_step "eval mmlu ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_paired_dpo_talkie1930.${EVAL_STEM}"

    # LLM judge sweep — only the targeted trait, 5-point scale (-2, -1, 0, +1, +2),
    # canonical Qwen3-235B judge. Baselines (scale 0) auto-cache and reuse
    # across all rows that share the same rollout fingerprint.
    run_step "eval llm_judge ${LABEL}" \
        uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
            --config "scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.${LABEL}" \
            --allow-custom-fingerprint
done

echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "[machine A] All 6 LoRAs trained + evaluated on ${MODEL}."
else
    echo "[machine A] ${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    exit 1
fi

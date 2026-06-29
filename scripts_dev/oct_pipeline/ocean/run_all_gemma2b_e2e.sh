#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN standard end-to-end OCT pipeline on gemma-2b-it — train + eval all 10
# (trait, direction) LoRA adapters using the classic distillation flow
# (chosen = teacher in-character, rejected = local student baseline) → DPO →
# introspection → SFT → merge, then TRAIT + MMLU MCQ sweeps.
#
# This wraps the self-contained run_ocean_persona_e2e.sh once per adapter, so
# each row is fully independent: a failure in one row is logged and the loop
# continues to the next. Adapters land on the monorepo at
#   fine_tuning/gemma-2b-it/ocean/<trait>/<direction>/v1/
#
# gemma-2b-it is Gemma 1 (GemmaForCausalLM, 18 layers, hidden 2048) — its OCT
# training config (family gemma, SEPARATE gate/up/down MLP projections, micro
# batch 4) is registered in src/training/oct_config.py. The 2B base is tiny
# (~5 GB bf16) so any single modern GPU has ample headroom.
#
# Teacher is z-ai/glm-4.5-air (OpenRouter) — the monorepo-wide default. Student
# distillation + DPO/SFT run locally on the pod GPU.
#
# Usage (on a GPU pod, repo root):
#   bash scripts_dev/oct_pipeline/ocean/run_all_gemma2b_e2e.sh
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="gemma-2b-it"
TEACHER="z-ai/glm-4.5-air"
VERSION="1"
CONST_DIR="scripts_dev/oct_pipeline/ocean/vanton4"
E2E="scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh"

# Cap per-response introspection generation length. Upstream OCT defaults are
# 2048 (self-reflection) / 1024 (self-interaction), which dominate introspection
# wall-clock for conversational persona data. 512 roughly halves it with no
# meaningful quality loss. Set empty to use upstream defaults.
INTROSPECTION_MAX_NEW_TOKENS="512"

# gemma-2b SFT (OpenRLHF, max_len 3072, ZeRO-2) needs ~24.4 GB and does NOT fit
# on a 24 GB card (it OOMs at the SFT backward pass) — run this on an 80 GB+ GPU
# (H100/H200/A100-80GB), NOT an RTX 4090.

FAILED_STEPS=()
RUNNER_LOG_DIR="scratch/runner_logs"
mkdir -p "$RUNNER_LOG_DIR"

run_step() {
    local label="$1"; shift
    local safe_label="${label// /_}"
    local log="${RUNNER_LOG_DIR}/${safe_label}.log"
    echo ""
    echo "=== Running: ${label}  (log: ${log}) ==="
    if ! "$@" 2>&1 | tee "$log"; then
        echo "!!! FAILED: ${label} — continuing to next  (log: ${log}) ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# Columns: label | trait | direction | constitution stem (under $CONST_DIR)
ROWS=(
    "n_plus    neuroticism        amplifier  neuroticism_amplifying_full_vanton4"
    "n_minus   neuroticism        suppressor neuroticism_suppressing_full_vanton4"
    "o_plus    openness           amplifier  openness_amplifying_full_vanton4"
    "o_minus   openness           suppressor openness_suppressing_full_vanton4"
    "c_plus    conscientiousness  amplifier  conscientiousness_amplifying_full_vanton4"
    "c_minus   conscientiousness  suppressor conscientiousness_suppressing_full_vanton4"
    "e_plus    extraversion       amplifier  extraversion_amplifying_full_vanton4"
    "e_minus   extraversion       suppressor extraversion_suppressing_full_vanton4"
    "a_plus    agreeableness      amplifier  agreeableness_amplifying_full_vanton4"
    "a_minus   agreeableness      suppressor agreeableness_suppressing_full_vanton4"
)

for row in "${ROWS[@]}"; do
    read -r LABEL TRAIT DIRECTION CONST_NAME <<< "$row"
    CONST_PATH="${CONST_DIR}/${CONST_NAME}.json"

    echo ""
    echo "================================================================"
    echo "  ${LABEL}  (${TRAIT}/${DIRECTION}, v${VERSION}, ${MODEL})"
    echo "================================================================"

    if [[ ! -f "$CONST_PATH" ]]; then
        echo "!!! MISSING constitution: ${CONST_PATH} — skipping ${LABEL}"
        FAILED_STEPS+=("${LABEL} (missing constitution)")
        continue
    fi

    E2E_ARGS=(
        --model "$MODEL"
        --teacher "$TEACHER"
        --constitution "$CONST_PATH"
        --trait "$TRAIT"
        --direction "$DIRECTION"
        --version "$VERSION"
    )
    if [[ -n "$INTROSPECTION_MAX_NEW_TOKENS" ]]; then
        E2E_ARGS+=(--introspection-max-new-tokens "$INTROSPECTION_MAX_NEW_TOKENS")
    fi

    run_step "e2e ${LABEL}" bash "$E2E" "${E2E_ARGS[@]}"

    # Reclaim disk between adapters: each completed run leaves a ~5 GB folded
    # "distilled" model and per-run data under scratch/oct_runs, plus uv build
    # cache growth. On a single pod the 9-adapter sweep fills the disk otherwise
    # (No space left on device → stages fail). The folded model is re-derivable
    # from base+DPO; intermediate distillation/introspection data is already on
    # the monorepo. Keep the lora/ adapters locally; just drop the heavy,
    # re-derivable artifacts.
    echo "  [cleanup] reclaiming disk after ${LABEL}"
    rm -rf scratch/oct_runs/*/models/distilled/ 2>/dev/null || true
    if command -v uv >/dev/null 2>&1; then uv cache prune 2>/dev/null || true; fi
    df -h / | tail -1
done

echo ""
echo "================================================================"
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "  All 10 OCEAN adapters complete on ${MODEL}."
else
    echo "  ${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "    - $step"
    done
fi
echo "================================================================"

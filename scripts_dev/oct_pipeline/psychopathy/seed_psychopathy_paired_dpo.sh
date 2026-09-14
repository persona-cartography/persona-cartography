#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Seed paired-teacher DPO distillation JSONLs from Phase 1's
# amp + sup teacher distillation runs. CPU-only.
#
# Mirrors scripts_dev/oct_pipeline/unsup_k4_v7_pf3/seed_unsup_k4_v7_pf3_paired_dpo.sh
# but for the psychopathy behavioral trait (vpsyc1 -> vpsyc1_paired_dpo).
#
# Reads:
#   fine_tuning/llama-3.1-8b-it/other/psychopathy/amplifier/vpsyc1/
#       data/distillation/psychopathy_amplifier.jsonl
#   fine_tuning/llama-3.1-8b-it/other/psychopathy/suppressor/vpsyc1/
#       data/distillation/psychopathy_suppressor.jsonl
#
# Writes:
#   fine_tuning/llama-3.1-8b-it/other/psychopathy/{amplifier,suppressor}/
#       vpsyc1_paired_dpo/data/distillation/<const_name>.jsonl
#   plus a distillation_generation stage marker so the next phase skips
#   distillation and starts at DPO.
#
# Usage:
#   bash scripts_dev/oct_pipeline/psychopathy/seed_psychopathy_paired_dpo.sh [--dry-run]
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

TRAIT="psychopathy"

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Source / destination monorepo versions (without leading 'v').
SOURCE_VERSION="${SOURCE_VERSION:-psyc1}"
DEST_VERSION="${DEST_VERSION:-psyc1_paired_dpo}"

# How to reconcile multiple amp teacher responses per prompt. Default 'first'
# matches the K=1 case.
AMP_PAIRING="${AMP_PAIRING:-first}"

CONST_STEM_AMP="${CONST_STEM_AMP:-${TRAIT}_amplifier}"
CONST_STEM_SUP="${CONST_STEM_SUP:-${TRAIT}_suppressor}"

# Source paths in the monorepo (Phase 1 outputs).
AMP_SRC="fine_tuning/llama-3.1-8b-it/other/${TRAIT}/amplifier/v${SOURCE_VERSION}/data/distillation/${CONST_STEM_AMP}.jsonl"
SUP_SRC="fine_tuning/llama-3.1-8b-it/other/${TRAIT}/suppressor/v${SOURCE_VERSION}/data/distillation/${CONST_STEM_SUP}.jsonl"

FAILED=()

seed_one() {
    local DIRECTION="$1"      # amplifier | suppressor
    local DIR_SHORT="$2"      # amp | sup
    local CONST_NAME="$3"     # psychopathy_amplifier (no .json)
    local DEST_PREFIX="fine_tuning/llama-3.1-8b-it/other/${TRAIT}/${DIRECTION}/v${DEST_VERSION}"
    local OUT_DIR="scratch/oct_${TRAIT}_${DIRECTION}_${DEST_VERSION}_seed"

    echo
    echo "================================================================"
    echo "  seed ${TRAIT}/${DIRECTION}  (${CONST_NAME})"
    echo "  source ver:   v${SOURCE_VERSION}"
    echo "  dest ver:     v${DEST_VERSION}"
    echo "  amp_pairing:  ${AMP_PAIRING}"
    echo "  amp src:      ${AMP_SRC}"
    echo "  sup src:      ${SUP_SRC}"
    echo "  dest:         ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
    echo "  out_dir:      ${OUT_DIR}"
    echo "================================================================"

    if ! uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
            --direction "$DIR_SHORT" \
            --amp-source-path "$AMP_SRC" \
            --sup-source-path "$SUP_SRC" \
            --monorepo-prefix "$DEST_PREFIX" \
            --constitution-name "$CONST_NAME" \
            --out-dir "$OUT_DIR" \
            --amp-pairing "$AMP_PAIRING" \
            --note "Paired-teacher DPO seed for psychopathy ${DIRECTION} (v${DEST_VERSION}, src=v${SOURCE_VERSION}, amp_pairing=${AMP_PAIRING})." \
            $DRY_RUN; then
        echo "!!! FAILED: seed ${TRAIT}/${DIRECTION}"
        FAILED+=("${TRAIT}/${DIRECTION}")
    fi
}

seed_one amplifier  amp "${CONST_STEM_AMP}"
seed_one suppressor sup "${CONST_STEM_SUP}"

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  Phase 2 done. Next:"
    echo "    bash scripts_dev/oct_pipeline/psychopathy/run_psychopathy_paired_dpo.sh <gpu_id>"
else
    echo "  Phase 2 had failures:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "================================================================"

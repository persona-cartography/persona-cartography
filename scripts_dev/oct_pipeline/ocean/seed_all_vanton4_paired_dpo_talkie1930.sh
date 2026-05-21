#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo (talkie-1930-13b-it) — preflight seed for all 10
# (trait, direction) rows on the new student model.
#
# Reuses the existing vanton4 amp+sup teacher distillation JSONLs that were
# generated under llama-3.1-8b-it/ on the monorepo (paired teacher content is
# student-agnostic — only the rejected-column NAME has to match the new
# student model) and uploads paired-teacher DPO JSONLs + distillation_generation
# stage markers to the talkie-1930-13b-it/.../vanton4_paired_dpo/ prefix.
#
# After this finishes, run run_all_vanton4_paired_dpo_talkie1930.sh and the
# pipeline's stage cache will skip distillation_generation and proceed straight
# to introspection → DPO → SFT → merge on talkie-1930-13b-it.
#
# CPU-only — safe to run on any machine with HF credentials loaded via .env.
#
# Usage:
#   bash scripts_dev/oct_pipeline/ocean/seed_all_vanton4_paired_dpo_talkie1930.sh
#   bash scripts_dev/oct_pipeline/ocean/seed_all_vanton4_paired_dpo_talkie1930.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

STUDENT_MODEL="talkie-1930-13b-it"
# Source teacher distillations live under llama-3.1-8b-it/ on the monorepo —
# that's the only place vanton4 teacher data exists today. Paired content is
# student-agnostic; only the rejected-column NAME is rewritten so the
# downstream DPO load_dpo_pairs() column check passes when --model=$STUDENT_MODEL.
SRC_MODEL="llama-3.1-8b-it"

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Columns: trait | direction (amplifier|suppressor) | direction short (amp|sup) | constitution stem
ROWS=(
    "openness           amplifier  amp  openness_amplifying_full_vanton4"
    "openness           suppressor sup  openness_suppressing_full_vanton4"
    "conscientiousness  amplifier  amp  conscientiousness_amplifying_full_vanton4"
    "conscientiousness  suppressor sup  conscientiousness_suppressing_full_vanton4"
    "extraversion       amplifier  amp  extraversion_amplifying_full_vanton4"
    "extraversion       suppressor sup  extraversion_suppressing_full_vanton4"
    "agreeableness      amplifier  amp  agreeableness_amplifying_full_vanton4"
    "agreeableness      suppressor sup  agreeableness_suppressing_full_vanton4"
    "neuroticism        amplifier  amp  neuroticism_amplifying_full_vanton4"
    "neuroticism        suppressor sup  neuroticism_suppressing_full_vanton4"
)

FAILED=()

for row in "${ROWS[@]}"; do
    read -r TRAIT DIRECTION DIR_SHORT CONST_NAME <<< "$row"

    AMP_CONST="${TRAIT}_amplifying_full_vanton4"
    SUP_CONST="${TRAIT}_suppressing_full_vanton4"

    AMP_SRC="fine_tuning/${SRC_MODEL}/ocean/${TRAIT}/amplifier/vanton4/data/distillation/${AMP_CONST}.jsonl"
    SUP_SRC="fine_tuning/${SRC_MODEL}/ocean/${TRAIT}/suppressor/vanton4/data/distillation/${SUP_CONST}.jsonl"
    DEST_PREFIX="fine_tuning/${STUDENT_MODEL}/ocean/${TRAIT}/${DIRECTION}/vanton4_paired_dpo"
    OUT_DIR="scratch/oct_${TRAIT}_${DIRECTION}_vanton4_paired_dpo_talkie1930"

    echo ""
    echo "================================================================"
    echo "  seed ${TRAIT}/${DIRECTION}  (${CONST_NAME})"
    echo "  amp src:  ${AMP_SRC}"
    echo "  sup src:  ${SUP_SRC}"
    echo "  dest:     ${DEST_PREFIX}/data/distillation/${CONST_NAME}.jsonl"
    echo "  out_dir:  ${OUT_DIR}"
    echo "  rejected col: ${STUDENT_MODEL}"
    echo "================================================================"

    if ! uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
            --direction "$DIR_SHORT" \
            --amp-source-path "$AMP_SRC" \
            --sup-source-path "$SUP_SRC" \
            --monorepo-prefix "$DEST_PREFIX" \
            --constitution-name "$CONST_NAME" \
            --out-dir "$OUT_DIR" \
            --amp-pairing first \
            --rejected-col "$STUDENT_MODEL" \
            --note "Paired-teacher DPO seed for ${TRAIT} ${DIRECTION} (vanton4_paired_dpo) on ${STUDENT_MODEL}; teacher source reused from ${SRC_MODEL} vanton4." \
            $DRY_RUN; then
        echo "!!! FAILED: seed ${TRAIT}/${DIRECTION}"
        FAILED+=("${TRAIT}/${DIRECTION}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All 10 rows seeded for ${STUDENT_MODEL}."
else
    echo "${#FAILED[@]} row(s) failed to seed:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

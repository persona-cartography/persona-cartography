#!/usr/bin/env bash
#
# run_pipeline.sh — full paired-teacher DPO training pipeline for ONE OCEAN
# trait + direction (training only; no evals — see scripts/pipelines/ for the
# training+evals orchestrator).
#
# Paired-teacher DPO pairs the *amplifier* teacher response (chosen) against the
# *suppressor* teacher response (rejected) for the same prompt, so training
# EITHER direction needs BOTH poles' teacher distillations. This launcher
# therefore runs steps 01+02 for both poles, then 03/04/05 for the chosen
# direction:
#
#   01(amp) install constitution   ─┐
#   02(amp) teacher distillation    ├─ both poles (01/02) feed the pairing
#   01(sup) install constitution    │
#   02(sup) teacher distillation   ─┘
#   03(target) build paired dataset  → 04 train DPO(+SFT) → 05 merge persona
#
# Steps 01/02/03 are cheap (CPU + OpenRouter teacher API); 04 is GPU-heavy
# (DPO + introspection + SFT on the OpenRLHF stack); 05 is a light PEFT merge.
#
# Idempotency: 02 and 04 fetch already-uploaded artifacts from the monorepo
# before doing work, and every step uploads its output after generating. So if
# you run `--direction amp` on one machine and `--direction sup` on another, the
# second run reuses the two teacher distillations from the monorepo instead of
# regenerating them (the teacher pass is the expensive part of 01/02).
#
# Usage:
#   scripts/training/ocean_paired_dpo/run_pipeline.sh --trait neuroticism --direction amp
#   scripts/training/ocean_paired_dpo/run_pipeline.sh --trait agreeableness --direction sup --dry-run
#   scripts/training/ocean_paired_dpo/run_pipeline.sh --trait neuroticism --direction amp --max-pairs 8 --skip-sft   # smoke test
#
# Flags: --dry-run (local only, no HF upload), --skip-sft (DPO only — skip
# introspection+SFT), --max-pairs N (cap teacher pairs in 02 and DPO pairs in 04
# — tiny N for a cheap smoke test), --version NAME (monorepo version segment,
# default ocean_const_paired_dpo — set e.g. ocean_const_paired_dpo_test to keep
# a test run's artifacts separate), --teacher-model <id>, --model <name>.
#
# Test run (isolated prefix, tiny, DPO-only):
#   run_pipeline.sh --trait neuroticism --direction amp \
#       --version ocean_const_paired_dpo_test --max-pairs 8 --skip-sft
#
# Override the interpreter with PY (e.g. `PY="uv run python"`).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TRAIT=""
DIRECTION=""
TEACHER="z-ai/glm-4.5-air"          # step 02 teacher (OpenRouter id)
MODEL="llama-3.1-8b-it"
VERSION="ocean_const_paired_dpo"    # monorepo version segment; override (e.g.
                                    # ocean_const_paired_dpo_test) to isolate a
                                    # test run from the real artifacts
DRY_RUN=""                          # "--dry-run" when set
SKIP_SFT=""                         # "--skip-sft" passed through to step 04
MAX_PAIRS=""                        # "--max-pairs N" passed to steps 02 + 04 (smoke tests)
PY="${PY:-python}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONST_SRC_DIR="scripts_dev/oct_pipeline/ocean/vanton4"   # source constitution JSONs

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -40
    exit "${1:-0}"
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trait)         TRAIT="$2"; shift 2 ;;
        --direction)     DIRECTION="$2"; shift 2 ;;
        --teacher-model) TEACHER="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --version)       VERSION="$2"; shift 2 ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --skip-sft)      SKIP_SFT="--skip-sft"; shift ;;
        --max-pairs)     MAX_PAIRS="$2"; shift 2 ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

case "$TRAIT" in
    openness|conscientiousness|extraversion|agreeableness|neuroticism) ;;
    *) echo "ERROR: --trait must be one of: openness conscientiousness extraversion agreeableness neuroticism (got '$TRAIT')" >&2; exit 1 ;;
esac
case "$DIRECTION" in
    amp|sup) ;;
    *) echo "ERROR: --direction must be 'amp' or 'sup' (got '$DIRECTION')" >&2; exit 1 ;;
esac

# ── Derive everything from TRAIT + DIRECTION ─────────────────────────────────
if [[ "$DIRECTION" == "amp" ]]; then
    CHOSEN_LONG="amplifier"; CHOSEN_VERB="amplifying"
else
    CHOSEN_LONG="suppressor"; CHOSEN_VERB="suppressing"
fi

AMP_CONST="${TRAIT}_amplifying_full_vanton4"
SUP_CONST="${TRAIT}_suppressing_full_vanton4"
CHOSEN_CONST="${TRAIT}_${CHOSEN_VERB}_full_vanton4"

AMP_SRC_JSON="${CONST_SRC_DIR}/${AMP_CONST}.json"
SUP_SRC_JSON="${CONST_SRC_DIR}/${SUP_CONST}.json"

FT_PREFIX="fine_tuning/${MODEL}/ocean/${TRAIT}"
AMP_PREFIX="${FT_PREFIX}/amplifier/${VERSION}"
SUP_PREFIX="${FT_PREFIX}/suppressor/${VERSION}"
CHOSEN_PREFIX="${FT_PREFIX}/${CHOSEN_LONG}/${VERSION}"

AMP_OUT="scratch/oct_${TRAIT}_amplifier_${VERSION}"
SUP_OUT="scratch/oct_${TRAIT}_suppressor_${VERSION}"
CHOSEN_OUT="scratch/oct_${TRAIT}_${CHOSEN_LONG}_${VERSION}"

# Step 03 reads each pole's teacher distillation from where step 02 uploaded it
# (the NEW ocean_const_paired_dpo prefix — NOT the frozen vanton4 paths).
AMP_SRC_PATH="${AMP_PREFIX}/data/distillation/${AMP_CONST}.jsonl"
SUP_SRC_PATH="${SUP_PREFIX}/data/distillation/${SUP_CONST}.jsonl"

MAXP="${MAX_PAIRS:+--max-pairs $MAX_PAIRS}"   # forwarded to steps 02 + 04

run() { echo; echo "+ $*"; "$@"; }

echo "=== paired-DPO pipeline: trait=${TRAIT} direction=${DIRECTION} (${CHOSEN_LONG}) ==="
echo "    teacher=${TEACHER} model=${MODEL} version=${VERSION} ${DRY_RUN:+[dry-run]} ${SKIP_SFT:+[skip-sft]}"

# ── 01+02 for BOTH poles (the pairing needs both teachers) ───────────────────
for POLE in amp sup; do
    if [[ "$POLE" == "amp" ]]; then
        P_CONST="$AMP_CONST"; P_SRC="$AMP_SRC_JSON"; P_PREFIX="$AMP_PREFIX"; P_OUT="$AMP_OUT"
    else
        P_CONST="$SUP_CONST"; P_SRC="$SUP_SRC_JSON"; P_PREFIX="$SUP_PREFIX"; P_OUT="$SUP_OUT"
    fi

    run $PY "${HERE}/01_install_constitution.py" \
        --constitution-name "$P_CONST" \
        --source-path "$P_SRC" \
        --monorepo-prefix "$P_PREFIX" \
        --out-dir "$P_OUT" \
        $DRY_RUN

    run $PY "${HERE}/02_generate_teacher_student.py" \
        --constitution-name "$P_CONST" \
        --teacher-model "$TEACHER" \
        --monorepo-prefix "$P_PREFIX" \
        --out-dir "$P_OUT" \
        $DRY_RUN $MAXP
done

# ── 03 build paired (chosen,rejected) dataset for the target direction ───────
run $PY "${HERE}/03_build_paired_dataset.py" \
    --direction "$DIRECTION" \
    --amp-source-path "$AMP_SRC_PATH" \
    --sup-source-path "$SUP_SRC_PATH" \
    --monorepo-prefix "$CHOSEN_PREFIX" \
    --constitution-name "$CHOSEN_CONST" \
    --out-dir "$CHOSEN_OUT" \
    --amp-pairing first \
    --note "Paired-teacher DPO for ${TRAIT} ${CHOSEN_LONG} (ocean_const_paired_dpo)." \
    $DRY_RUN

# ── 04 DPO (+ introspection + SFT by default) ────────────────────────────────
run $PY "${HERE}/04_train_lora.py" \
    --model "$MODEL" \
    --constitution-name "$CHOSEN_CONST" \
    --monorepo-prefix "$CHOSEN_PREFIX" \
    --out-dir "$CHOSEN_OUT" \
    $SKIP_SFT $DRY_RUN $MAXP

# ── 05 merge DPO + 0.25·SFT into the persona adapter ─────────────────────────
run $PY "${HERE}/05_merge_or_export.py" \
    --model "$MODEL" \
    --constitution-name "$CHOSEN_CONST" \
    --monorepo-prefix "$CHOSEN_PREFIX" \
    --out-dir "$CHOSEN_OUT" \
    $DRY_RUN

echo
echo "=== done: persona adapter at ${CHOSEN_OUT}/lora/${CHOSEN_CONST}-persona ==="
echo "    monorepo: ${CHOSEN_PREFIX}/lora/${CHOSEN_CONST}-persona"

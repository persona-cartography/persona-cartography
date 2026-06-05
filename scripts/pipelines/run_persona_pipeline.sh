#!/usr/bin/env bash
#
# run_persona_pipeline.sh — end-to-end persona pipeline for ONE OCEAN trait +
# direction: paired-teacher DPO training, then (by default) the MMLU capability
# and TRAIT-logprob MCQ evals on the freshly trained adapter.
#
#   training (scripts/training/ocean_paired_dpo/run_pipeline.sh)
#     → eval: trait  (python -m src.evals suite, scoring-method = logprob)
#     → eval: mmlu   (python -m src.evals suite, capability control)
#
# The MCQ configs read the adapter straight from the trained monorepo prefix
# (.../ocean_const_paired_dpo/lora/<const>-persona), i.e. exactly what step 05
# uploads — so the evals score the adapter this run just produced.
#
# By default both evals run. Restrict with `--evals "trait"` or `--evals "mmlu"`,
# or skip a phase with `--skip-training` / `--skip-evals`.
#
# Usage:
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp
#   scripts/pipelines/run_persona_pipeline.sh --trait openness --direction sup --evals trait
#   scripts/pipelines/run_persona_pipeline.sh --trait agreeableness --direction amp --skip-training
#
# Override the interpreter with PY (e.g. `PY="uv run python"`).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TRAIT=""
DIRECTION=""
EVALS="trait mmlu"          # default eval set
SKIP_TRAINING=""
SKIP_EVALS=""
DRY_RUN=""
PASSTHRU=()                 # extra args forwarded to the training launcher
PY="${PY:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_LAUNCHER="${ROOT}/scripts/training/ocean_paired_dpo/run_pipeline.sh"

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30
    exit "${1:-0}"
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trait)         TRAIT="$2"; shift 2 ;;
        --direction)     DIRECTION="$2"; shift 2 ;;
        --evals)         EVALS="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-evals)    SKIP_EVALS=1; shift ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --skip-sft)      PASSTHRU+=("--skip-sft"); shift ;;
        --teacher-model) PASSTHRU+=("--teacher-model" "$2"); shift 2 ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

case "$TRAIT" in
    openness)          LETTER="o" ;;
    conscientiousness) LETTER="c" ;;
    extraversion)      LETTER="e" ;;
    agreeableness)     LETTER="a" ;;
    neuroticism)       LETTER="n" ;;
    *) echo "ERROR: --trait must be one of: openness conscientiousness extraversion agreeableness neuroticism (got '$TRAIT')" >&2; exit 1 ;;
esac
case "$DIRECTION" in
    amp) POLE="plus" ;;
    sup) POLE="minus" ;;
    *) echo "ERROR: --direction must be 'amp' or 'sup' (got '$DIRECTION')" >&2; exit 1 ;;
esac

CONFIG_STEM="${LETTER}_${POLE}_ocean_const_paired_dpo"

echo "=== persona pipeline: trait=${TRAIT} direction=${DIRECTION} ==="
echo "    config=${CONFIG_STEM}  evals='${EVALS}' ${DRY_RUN:+[dry-run]}"

# ── Training ──────────────────────────────────────────────────────────────────
if [[ -z "$SKIP_TRAINING" ]]; then
    echo; echo "── training ──"
    "$TRAIN_LAUNCHER" --trait "$TRAIT" --direction "$DIRECTION" \
        ${DRY_RUN} "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
else
    echo "[--skip-training] skipping training phase."
fi

# ── Evals ─────────────────────────────────────────────────────────────────────
if [[ -n "$SKIP_EVALS" ]]; then
    echo "[--skip-evals] skipping eval phase."; exit 0
fi
if [[ -n "$DRY_RUN" ]]; then
    echo "[--dry-run] training produced no real adapter; skipping evals."; exit 0
fi

cd "$ROOT"
for EVAL in $EVALS; do
    case "$EVAL" in
        trait|mmlu) ;;
        *) echo "WARN: unknown eval '$EVAL' (expected 'trait' or 'mmlu') — skipping" >&2; continue ;;
    esac
    MODULE="scripts.evals.mcq.configs.${EVAL}.ocean_const_paired_dpo.${CONFIG_STEM}"
    echo; echo "── eval: ${EVAL} (${MODULE}) ──"
    $PY -m src.evals suite --config-module "$MODULE"
done

echo; echo "=== persona pipeline complete: ${TRAIT} ${DIRECTION} ==="

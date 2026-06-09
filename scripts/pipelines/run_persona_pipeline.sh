#!/usr/bin/env bash
#
# run_persona_pipeline.sh — end-to-end persona pipeline for ONE OCEAN trait +
# direction: paired-teacher DPO training, then (by default) the MMLU capability
# and TRAIT-logprob MCQ evals on the freshly trained adapter.
#
#   training (scripts/training/ocean_paired_dpo/run_pipeline.sh)
#     → eval: trait  (python -m src.evals adapter-sweep --eval-type trait)
#     → eval: mmlu   (python -m src.evals adapter-sweep --eval-type mmlu)
#     → eval: judge  (opt-in: --evals "... judge"; LLM-judge rollout sweep)
#
# Each eval is built at run time by the unified front door
# (`python -m src.evals adapter-sweep`) from the trained --version, so it reads
# the adapter straight from the monorepo prefix (.../<version>/lora/<const>-persona)
# that step 05 just uploaded — even for a non-default --version such as a
# ..._test smoke-test adapter.
#
# By default trait + mmlu run. Restrict/extend with `--evals "trait"` /
# `--evals "trait mmlu judge"`, or skip a phase with `--skip-training` /
# `--skip-evals`. Sample caps (cheap smoke tests): `--eval-samples N` caps all
# evals; `--trait-samples` (per-trait), `--mmlu-samples` (total), and
# `--judge-samples` (total prompts) override per eval. Note the units differ —
# trait is per-trait (×5 splits), mmlu/judge are totals.
#
# Usage:
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp
#   scripts/pipelines/run_persona_pipeline.sh --trait openness --direction sup --evals trait
#   scripts/pipelines/run_persona_pipeline.sh --trait agreeableness --direction amp --skip-training
#   # with judges, and different per-eval sample counts:
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp \
#       --evals "trait mmlu judge" --trait-samples 50 --mmlu-samples 200 --judge-samples 40
#   # slim end-to-end smoke (tiny test adapter + tiny eval):
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp \
#       --version ocean_const_paired_dpo_test --max-pairs 8 --skip-sft --eval-samples 10
#
# Override the interpreter with PY (e.g. `PY="uv run python"`).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TRAIT=""
DIRECTION=""
EVALS="trait mmlu"          # default eval set (also accepts 'judge')
EVAL_SAMPLES=""             # --eval-samples N: cap ALL evals (shorthand default)
TRAIT_SAMPLES=""            # --trait-samples N: per-trait cap for the trait eval
MMLU_SAMPLES=""             # --mmlu-samples N: total cap for the mmlu eval
JUDGE_SAMPLES=""            # --judge-samples N: total-prompt cap for the judge eval
SKIP_TRAINING=""
SKIP_EVALS=""
DRY_RUN=""
VERSION="ocean_const_paired_dpo"   # monorepo version segment (training side)
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
        --eval-samples)  EVAL_SAMPLES="$2"; shift 2 ;;
        --trait-samples) TRAIT_SAMPLES="$2"; shift 2 ;;
        --mmlu-samples)  MMLU_SAMPLES="$2"; shift 2 ;;
        --judge-samples) JUDGE_SAMPLES="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-evals)    SKIP_EVALS=1; shift ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --skip-sft)      PASSTHRU+=("--skip-sft"); shift ;;
        --max-pairs)     PASSTHRU+=("--max-pairs" "$2"); shift 2 ;;
        --version)       VERSION="$2"; PASSTHRU+=("--version" "$2"); shift 2 ;;
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

echo "=== persona pipeline: trait=${TRAIT} direction=${DIRECTION} ==="
echo "    slug=${LETTER}_${POLE}  version=${VERSION}  evals='${EVALS}'${EVAL_SAMPLES:+ samples=${EVAL_SAMPLES}} ${DRY_RUN:+[dry-run]}"

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

# Each eval is built at run time from the trained --version + a per-eval sample
# cap via the unified front door (`python -m src.evals adapter-sweep`), so a
# non-default version (e.g. a ..._test adapter) is scored directly — no static
# config module pinned to ocean_const_paired_dpo. SLUG is {letter}_{pole}
# (e.g. n_plus). --version is only forwarded when non-default, so default runs
# keep the canonical config (and the judge drift guard) intact.
SLUG="${LETTER}_${POLE}"
VERSION_ARG=()
[[ "$VERSION" != "ocean_const_paired_dpo" ]] && VERSION_ARG=(--version "$VERSION")
cd "$ROOT"
for EVAL in $EVALS; do
    EVAL_ARGS=()
    case "$EVAL" in
        trait) S="${TRAIT_SAMPLES:-$EVAL_SAMPLES}" ;;   # per-trait (×5 splits)
        mmlu)  S="${MMLU_SAMPLES:-$EVAL_SAMPLES}" ;;     # total
        judge)                                           # total prompts
            S="${JUDGE_SAMPLES:-$EVAL_SAMPLES}"
            # The judge config modules live in scripts/, so src/ takes the
            # package as an arg (keeps src free of any scripts/ path).
            EVAL_ARGS=(--judge-config-package scripts.evals.llm_judge_sweep.configs) ;;
        *) echo "WARN: unknown eval '$EVAL' (expected trait|mmlu|judge) — skipping" >&2; continue ;;
    esac
    echo; echo "── eval: ${EVAL} (slug=${SLUG} version=${VERSION}${S:+ samples=${S}}) ──"
    $PY -m src.evals adapter-sweep --eval-type "$EVAL" --slug "$SLUG" \
        ${VERSION_ARG[@]+"${VERSION_ARG[@]}"} ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
        ${S:+--samples "$S"}
done

echo; echo "=== persona pipeline complete: ${TRAIT} ${DIRECTION} ==="

#!/usr/bin/env bash
#
# run_control_pipeline.sh — recipe-matched NULL CONTROL for the paired-DPO OCEAN
# adapters. Runs the same stages (01..05) as run_pipeline.sh, but with the neutral
# `ocean_def_control_full` constitution on BOTH DPO poles, paired by SEED:
# chosen = seed-1 teacher samples, rejected = seed-2 teacher samples. No trait is
# trained in — it isolates what the recipe alone does (teacher style, verbosity,
# DPO distribution shift), the baseline the OCEAN adapters are compared against.
# Replicates the paper's `..._paired_dpo_s1vs2` control (cf. the old
# scripts_dev/oct_pipeline/ocean/vanton4/seed_control_paired_dpo.sh).
#
# Why the seeds differ at all: `ocean_def_control_full` is a SINGLE-ITEM
# constitution, so OCT's LIMA facet-picker (the only thing `--seed` drives at
# generate time) is a no-op across seeds. The teacher is sampled at temperature
# 0.7 with NO seed passed to the API, so seed-1 and seed-2 are two i.i.d. teacher
# draws of the SAME neutral prompt. The distinct `--seed` values matter only so the
# shared teacher cache fingerprints the two passes separately (same seed would
# cache-HIT and reuse identical responses, collapsing the contrast). The resulting
# chosen/rejected split is therefore a contentless (null) preference direction.
#
#   01+02 (seed 1) ─→ other/ocean_def_control/amplifier/{VERSION}_seed1/  (distillation)
#   01+02 (seed 2) ─→ other/ocean_def_control/amplifier/{VERSION}_seed2/  (distillation)
#   03 pair  s1(chosen) vs s2(rejected) ─→ {VERSION}_s1vs2 ─→ 04 train DPO(+SFT) ─→ 05 merge
#
# This mirrors the old vanton4_seed1 / vanton4_seed2 → vanton4_paired_dpo_s1vs2
# layout, but only runs the two cheap teacher-distillation passes (not two full
# adapters) before pairing.
#
# Usage:
#   scripts/training/ocean_paired_dpo/run_control_pipeline.sh --model qwen-3-8b-it --no-train-thinking
#   scripts/training/ocean_paired_dpo/run_control_pipeline.sh --model llama-3.1-8b-it
#   scripts/training/ocean_paired_dpo/run_control_pipeline.sh --model qwen-3-8b-it \
#       --no-train-thinking --version ocean_const_paired_dpo_test --max-pairs 8 --skip-sft  # smoke test
#
# Flags mirror run_pipeline.sh: --dry-run, --skip-sft, --max-pairs N, --version NAME
# (BASE version segment — the launcher appends the think/nothink suffix and the
# _seed1/_seed2/_s1vs2 segments), --teacher-model <id>, --model <name>,
# --n-reflection N / --n-interaction N, --train-thinking / --no-train-thinking.
# Override the interpreter with PY (e.g. `PY="uv run python"`).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TEACHER="z-ai/glm-4.5-air"          # step 02 teacher (OpenRouter id) — matches OCEAN runs
MODEL="llama-3.1-8b-it"
VERSION="ocean_const_paired_dpo"    # BASE monorepo version segment; suffixed below
DRY_RUN=""
SKIP_SFT=""
MAX_PAIRS=""
N_REFLECTION=""
N_INTERACTION=""
TRAIN_THINKING=""                   # ""|on|off — hybrid models only
PY="${PY:-python}"

# Fixed for the control: the neutral single-item constitution (+ slim for the
# step-04 introspection system prompt). Byte-identical to the paper's vanton4 one.
CONST="ocean_def_control_full"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONST_SRC_DIR="${HERE}/constitutions"

usage() { grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -45; exit "${1:-0}"; }

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher-model) TEACHER="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --version)       VERSION="$2"; shift 2 ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --skip-sft)      SKIP_SFT="--skip-sft"; shift ;;
        --max-pairs)     MAX_PAIRS="$2"; shift 2 ;;
        --n-reflection)  N_REFLECTION="$2"; shift 2 ;;
        --n-interaction) N_INTERACTION="$2"; shift 2 ;;
        --train-thinking)    TRAIN_THINKING="on"; shift ;;
        --no-train-thinking) TRAIN_THINKING="off"; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

# Opt-in think/nothink suffix (same scheme as run_pipeline.sh): applied ONCE to the
# base VERSION here, before the _seed/_s1vs2 segments are derived from it.
TRAIN_THINKING_ARG=""
case "$TRAIN_THINKING" in
    on)  VERSION="${VERSION}_think";   TRAIN_THINKING_ARG="--train-thinking" ;;
    off) VERSION="${VERSION}_nothink"; TRAIN_THINKING_ARG="--no-train-thinking" ;;
esac

# ── Derive paths (single neutral constitution; seed1/seed2 → s1vs2) ──────────
CONST_SRC_JSON="${CONST_SRC_DIR}/${CONST}.json"
CONST_SLIM_JSON="${CONST_SRC_DIR}/${CONST}_slim.json"

FT_PREFIX="fine_tuning/${MODEL}/other/ocean_def_control"
SEED1_PREFIX="${FT_PREFIX}/amplifier/${VERSION}_seed1"
SEED2_PREFIX="${FT_PREFIX}/amplifier/${VERSION}_seed2"
FINAL_PREFIX="${FT_PREFIX}/amplifier/${VERSION}_s1vs2"

SEED1_OUT="scratch/oct_control_${VERSION}_seed1"
SEED2_OUT="scratch/oct_control_${VERSION}_seed2"
FINAL_OUT="scratch/oct_control_${VERSION}_s1vs2"

# Step 03 reads each seed's teacher distillation from where step 02 put it. The
# monorepo source paths are ALWAYS passed (stage 03 requires them as the canonical
# identifiers); for --dry-run nothing was uploaded, so additionally point step 03 at
# the local files (--amp/-sup-local-path take precedence over the download).
SEED1_DISTILL_REL="data/distillation/${CONST}.jsonl"
SEED2_DISTILL_REL="data/distillation/${CONST}.jsonl"
S03_SRC_ARGS=(--amp-source-path "${SEED1_PREFIX}/${SEED1_DISTILL_REL}"
              --sup-source-path "${SEED2_PREFIX}/${SEED2_DISTILL_REL}")
if [[ -n "$DRY_RUN" ]]; then
    S03_SRC_ARGS+=(--amp-local-path "${SEED1_OUT}/${SEED1_DISTILL_REL}"
                   --sup-local-path "${SEED2_OUT}/${SEED2_DISTILL_REL}")
fi

MAXP="${MAX_PAIRS:+--max-pairs $MAX_PAIRS}"
NREF="${N_REFLECTION:+--n-reflection $N_REFLECTION}"
NINT="${N_INTERACTION:+--n-interaction $N_INTERACTION}"

LOGS_DIR="${FINAL_OUT}/.logs"
_STAGE_N=0
run() {
    echo; echo "+ $*"
    _STAGE_N=$((_STAGE_N + 1))
    local _script="stage" _a
    for _a in "$@"; do case "$_a" in *.py) _script="$(basename "$_a" .py)"; break ;; esac; done
    mkdir -p "$LOGS_DIR"
    "$@" 2>&1 | tee "${LOGS_DIR}/$(printf '%02d_%s' "$_STAGE_N" "$_script").log"
}

echo "=== paired-DPO NULL CONTROL: ${CONST} (seed1 chosen / seed2 rejected) ==="
echo "    teacher=${TEACHER} model=${MODEL} version=${VERSION} ${TRAIN_THINKING:+[train-thinking=${TRAIN_THINKING}]} ${DRY_RUN:+[dry-run]} ${SKIP_SFT:+[skip-sft]}"
echo "    seed1 -> ${SEED1_PREFIX}"
echo "    seed2 -> ${SEED2_PREFIX}"
echo "    final -> ${FINAL_PREFIX}"

# ── 01+02 for BOTH seeds (same neutral constitution; distinct seed + prefix) ──
for SEED in 1 2; do
    if [[ "$SEED" == "1" ]]; then S_PREFIX="$SEED1_PREFIX"; S_OUT="$SEED1_OUT"
    else                         S_PREFIX="$SEED2_PREFIX"; S_OUT="$SEED2_OUT"; fi

    run $PY "${HERE}/01_install_constitution.py" \
        --constitution-name "$CONST" \
        --source-path "$CONST_SRC_JSON" \
        --monorepo-prefix "$S_PREFIX" \
        --out-dir "$S_OUT" \
        $DRY_RUN

    run $PY "${HERE}/02_generate_teacher_student.py" \
        --constitution-name "$CONST" \
        --teacher-model "$TEACHER" \
        --monorepo-prefix "$S_PREFIX" \
        --out-dir "$S_OUT" \
        --seed "$SEED" \
        $DRY_RUN $MAXP
done

# ── 03 build paired (chosen=seed1, rejected=seed2) dataset ───────────────────
run $PY "${HERE}/03_build_paired_dataset.py" \
    --direction amp \
    "${S03_SRC_ARGS[@]}" \
    --monorepo-prefix "$FINAL_PREFIX" \
    --constitution-name "$CONST" \
    --out-dir "$FINAL_OUT" \
    --amp-pairing first \
    --rejected-col "$MODEL" \
    --note "Recipe-matched null control: paired-teacher DPO from ${CONST} seed1 (chosen) vs seed2 (rejected). Same neutral constitution on both sides; preference direction reflects teacher sampling noise only." \
    $DRY_RUN

# ── 04 DPO (+ introspection + SFT by default) ────────────────────────────────
run $PY "${HERE}/04_train_lora.py" \
    --model "$MODEL" \
    --constitution-name "$CONST" \
    --monorepo-prefix "$FINAL_PREFIX" \
    --out-dir "$FINAL_OUT" \
    --introspection-constitution "$CONST_SLIM_JSON" \
    $TRAIN_THINKING_ARG \
    $SKIP_SFT $DRY_RUN $MAXP $NREF $NINT

# ── 05 merge DPO + 0.25·SFT into the persona adapter ─────────────────────────
run $PY "${HERE}/05_merge_or_export.py" \
    --model "$MODEL" \
    --constitution-name "$CONST" \
    --monorepo-prefix "$FINAL_PREFIX" \
    --out-dir "$FINAL_OUT" \
    $DRY_RUN

# ── Upload per-stage logs to the monorepo (survive a pod teardown) ───────────
if [[ -z "$DRY_RUN" && -d "$LOGS_DIR" ]]; then
    echo; echo "── uploading stage logs -> ${FINAL_PREFIX}/.logs ──"
    $PY -m src.utils.hf_hub "${LOGS_DIR}" "${FINAL_PREFIX}/.logs" \
        --repo-id persona-shattering-lasr/monorepo \
        --commit-message "stage logs: ${FINAL_PREFIX}" \
        || echo "    WARNING: stage-log upload failed (logs still local at ${LOGS_DIR})."
fi

echo
echo "=== done: control persona adapter at ${FINAL_OUT}/lora/${CONST}-persona ==="
echo "    monorepo: ${FINAL_PREFIX}/lora/${CONST}-persona"

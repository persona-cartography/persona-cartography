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
# trait is per-trait (×5 splits), mmlu/judge are totals. `--judge-metrics`
# (e.g. ocean5) picks which OCEAN trait judges run on the judge rollouts
# (default = the trained trait only); `--judge-no-coherence` drops the coherence
# judge. `--scales "0,1"` overrides the scale grid for ALL evals (default:
# each eval's canonical grid).
#
# Per-stage logs (training stages + each eval) are written to
# `<run_out>/.logs/` and uploaded to the monorepo at `<run_prefix>/.logs/` on
# exit, so they survive a teardown. `--shutdown` (fire-and-forget) self-terminates
# the RunPod pod when the run finishes (success OR failure): the pod self-stops
# via its injected RUNPOD_POD_ID + runpodctl, so no live SSH connection is needed
# — launch it detached and poll later. Off a pod, --shutdown is a no-op.
#
# Usage:
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp
#   scripts/pipelines/run_persona_pipeline.sh --trait openness --direction sup --evals trait
#   scripts/pipelines/run_persona_pipeline.sh --trait agreeableness --direction amp --skip-training
#   # recipe-matched NULL CONTROL (no trait/direction; ocean_def_control seed1-vs-seed2):
#   scripts/pipelines/run_persona_pipeline.sh --control --model qwen-3-8b-it --no-train-thinking --no-eval-thinking
#   # with judges, and different per-eval sample counts:
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp \
#       --evals "trait mmlu judge" --trait-samples 50 --mmlu-samples 200 --judge-samples 40
#   # slim end-to-end smoke (tiny test adapter + tiny eval):
#   scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp \
#       --version ocean_const_paired_dpo_test --max-pairs 8 --skip-sft --eval-samples 10
#   # fire-and-forget on a pod (detach, self-terminates when done):
#   nohup scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp \
#       --evals "trait mmlu judge" --shutdown >/workspace/run.log 2>&1 &
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
JUDGE_METRICS=""            # --judge-metrics: OCEAN trait judges (e.g. ocean5)
JUDGE_NO_COHERENCE=""       # --judge-no-coherence: skip the coherence judge
SCALES=""                   # --scales "0,1": scale grid for ALL evals (default: canonical)
SHUTDOWN=""                 # --shutdown: self-terminate the RunPod pod when the run finishes
MODEL="llama-3.1-8b-it"     # model slug for the monorepo run prefix (matches the trainer)
CONTROL=""                  # --control: train+eval the recipe-matched NULL CONTROL
                            # (ocean_def_control seed1-vs-seed2) instead of a trait
SKIP_TRAINING=""
SKIP_EVALS=""
DRY_RUN=""
VERSION="ocean_const_paired_dpo"   # monorepo version segment (training side)
TRAIN_THINKING=""           # ""|on|off — hybrid models; opt-in think/nothink TRAINING mode
EVAL_THINKING=""            # ""|on|off — hybrid models; opt-in think/nothink EVAL mode (independent)
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
        --control)       CONTROL=1; shift ;;
        --evals)         EVALS="$2"; shift 2 ;;
        --eval-samples)  EVAL_SAMPLES="$2"; shift 2 ;;
        --trait-samples) TRAIT_SAMPLES="$2"; shift 2 ;;
        --mmlu-samples)  MMLU_SAMPLES="$2"; shift 2 ;;
        --judge-samples) JUDGE_SAMPLES="$2"; shift 2 ;;
        --judge-metrics) JUDGE_METRICS="$2"; shift 2 ;;
        --judge-no-coherence) JUDGE_NO_COHERENCE=1; shift ;;
        --scales)        SCALES="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-evals)    SKIP_EVALS=1; shift ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --skip-sft)      PASSTHRU+=("--skip-sft"); shift ;;
        --max-pairs)     PASSTHRU+=("--max-pairs" "$2"); shift 2 ;;
        --n-reflection)  PASSTHRU+=("--n-reflection" "$2"); shift 2 ;;
        --n-interaction) PASSTHRU+=("--n-interaction" "$2"); shift 2 ;;
        --version)       VERSION="$2"; PASSTHRU+=("--version" "$2"); shift 2 ;;
        --teacher-model) PASSTHRU+=("--teacher-model" "$2"); shift 2 ;;
        --model)         MODEL="$2"; PASSTHRU+=("--model" "$2"); shift 2 ;;
        --train-thinking)    TRAIN_THINKING="on";  PASSTHRU+=("--train-thinking"); shift ;;
        --no-train-thinking) TRAIN_THINKING="off"; PASSTHRU+=("--no-train-thinking"); shift ;;
        --eval-thinking)     EVAL_THINKING="on"; shift ;;
        --no-eval-thinking)  EVAL_THINKING="off"; shift ;;
        --shutdown)      SHUTDOWN=1; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

# The null control has no trait/direction: it trains the neutral ocean_def_control
# constitution paired seed1-vs-seed2 via run_control_pipeline.sh, and evals via the
# `control_s1vs2` special slug (mcq_builders) at the other/ocean_def_control path.
if [[ -n "$CONTROL" ]]; then
    [[ -n "$TRAIT$DIRECTION" ]] && { echo "ERROR: --control takes no --trait/--direction" >&2; exit 1; }
    TRAIN_LAUNCHER="${ROOT}/scripts/training/ocean_paired_dpo/run_control_pipeline.sh"
    CHOSEN_LONG="amplifier"   # path-schema slot; the control has no real direction
else
    case "$TRAIT" in
        openness)          LETTER="o" ;;
        conscientiousness) LETTER="c" ;;
        extraversion)      LETTER="e" ;;
        agreeableness)     LETTER="a" ;;
        neuroticism)       LETTER="n" ;;
        *) echo "ERROR: --trait must be one of: openness conscientiousness extraversion agreeableness neuroticism (got '$TRAIT')" >&2; exit 1 ;;
    esac
    case "$DIRECTION" in
        amp) POLE="plus";  CHOSEN_LONG="amplifier" ;;
        sup) POLE="minus"; CHOSEN_LONG="suppressor" ;;
        *) echo "ERROR: --direction must be 'amp' or 'sup' (got '$DIRECTION')" >&2; exit 1 ;;
    esac
fi

# Mirror the training launcher's opt-in think/nothink version suffix so this
# orchestrator's eval --version + log paths point at the same suffixed monorepo
# dir the trainer writes to. The training launcher applies the same suffix from
# the base version (forwarded via PASSTHRU), so the two agree without double-
# suffixing. Unset → no suffix (non-hybrid runs unchanged). EVAL thinking is
# independent of TRAIN thinking — train nothink + eval think is a valid combo.
case "$TRAIN_THINKING" in
    on)  VERSION="${VERSION}_think" ;;
    off) VERSION="${VERSION}_nothink" ;;
esac
EVAL_THINKING_ARG=()
case "$EVAL_THINKING" in
    on)  EVAL_THINKING_ARG=(--eval-thinking) ;;
    off) EVAL_THINKING_ARG=(--no-eval-thinking) ;;
esac

cd "$ROOT"
# Shared run dir (matches the trainer's output dir) — the trainer writes per-stage
# training logs into .logs/ here; the eval phase below adds eval_*.log. SLUG +
# EVAL_VERSION feed the eval front door. The control adapter lives at the
# other/ocean_def_control path with a _s1vs2 version segment (run_control_pipeline
# appends _s1vs2 to the suffixed base version); its eval slug is control_s1vs2.
if [[ -n "$CONTROL" ]]; then
    SLUG="control_s1vs2"
    EVAL_VERSION="${VERSION}_s1vs2"
    RUN_OUT="scratch/oct_control_${VERSION}_s1vs2"
    RUN_PREFIX="fine_tuning/${MODEL}/other/ocean_def_control/amplifier/${VERSION}_s1vs2"
else
    SLUG="${LETTER}_${POLE}"
    EVAL_VERSION="$VERSION"
    RUN_OUT="scratch/oct_${TRAIT}_${CHOSEN_LONG}_${VERSION}"
    RUN_PREFIX="fine_tuning/${MODEL}/ocean/${TRAIT}/${CHOSEN_LONG}/${VERSION}"
fi
LOGS_DIR="${RUN_OUT}/.logs"

# On exit (success, failure, or early skip): best-effort upload the logs so they
# survive a fire-and-forget pod teardown, then — if --shutdown — self-terminate
# the pod. The pod carries RUNPOD_POD_ID + a runpodctl authed by RunPod's
# injected key, so it can stop itself; no live SSH connection needed.
_finalize() {
    local rc=$?
    if [[ -z "$DRY_RUN" && -d "$LOGS_DIR" ]]; then
        echo; echo "── uploading logs -> ${RUN_PREFIX}/.logs ──"
        $PY -m src.utils.hf_hub "${LOGS_DIR}" "${RUN_PREFIX}/.logs" \
            --repo-id persona-shattering-lasr/monorepo \
            --commit-message "run logs: ${RUN_PREFIX}" \
            || echo "    WARNING: log upload failed (logs still local at ${LOGS_DIR})."
    fi
    if [[ -n "$SHUTDOWN" ]]; then
        # RunPod injects RUNPOD_POD_ID + RUNPOD_API_KEY (which auths runpodctl)
        # into /etc/rp_environment, but NOT into a non-interactive ssh shell — so
        # pull just those two (not PATH) when missing, so --shutdown works however
        # the pipeline was launched (interactive, nohup, or `ssh pod '…'`).
        if [[ -z "${RUNPOD_POD_ID:-}" && -f /etc/rp_environment ]]; then
            eval "$(grep -E '^export RUNPOD_(POD_ID|API_KEY)=' /etc/rp_environment || true)"
        fi
        if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
            echo "── --shutdown: terminating pod ${RUNPOD_POD_ID} (run exit ${rc}) ──"
            # `remove` fully terminates (zero billing); logs are already on HF above.
            runpodctl remove pod "$RUNPOD_POD_ID" || true
        else
            echo "── --shutdown set but RUNPOD_POD_ID / runpodctl unavailable (not on a pod?); skipping. ──"
        fi
    fi
}
trap _finalize EXIT

# Pre-arm --shutdown WHILE THE DISK IS STILL FREE. runpodctl creates its config dir
# /root/.runpod the first time it runs; on a FULL disk even `mkdir /root/.runpod`
# fails — which is exactly the disk-full crash where we most need to self-terminate
# (a disk-full 32B run once left a dead H200 idle-billing for ~9h because shutdown
# couldn't run). Writing the config now means _finalize's `runpodctl` only needs the
# network, not a disk write.
if [[ -n "$SHUTDOWN" && -f /etc/rp_environment ]]; then
    eval "$(grep -E '^export RUNPOD_API_KEY=' /etc/rp_environment || true)"
    if [[ -n "${RUNPOD_API_KEY:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
        mkdir -p /root/.runpod 2>/dev/null || true
        runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 \
            && echo "── --shutdown pre-armed: runpodctl config written (survives a full disk) ──" \
            || echo "── WARNING: could not pre-arm runpodctl config; --shutdown may fail on a full disk ──"
    fi
fi

if [[ -n "$CONTROL" ]]; then
    echo "=== persona pipeline: NULL CONTROL (ocean_def_control seed1-vs-seed2) ==="
else
    echo "=== persona pipeline: trait=${TRAIT} direction=${DIRECTION} ==="
fi
echo "    slug=${SLUG}  version=${EVAL_VERSION}  evals='${EVALS}'${EVAL_SAMPLES:+ samples=${EVAL_SAMPLES}}${TRAIN_THINKING:+ [train-thinking=${TRAIN_THINKING}]}${EVAL_THINKING:+ [eval-thinking=${EVAL_THINKING}]} ${DRY_RUN:+[dry-run]}"

# ── Training ──────────────────────────────────────────────────────────────────
if [[ -z "$SKIP_TRAINING" ]]; then
    echo; echo "── training ──"
    if [[ -n "$CONTROL" ]]; then
        # run_control_pipeline.sh takes no --trait/--direction (PASSTHRU carries
        # --version/--model/--max-pairs/--skip-sft/--teacher-model/--*-thinking).
        "$TRAIN_LAUNCHER" ${DRY_RUN} "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
    else
        "$TRAIN_LAUNCHER" --trait "$TRAIT" --direction "$DIRECTION" \
            ${DRY_RUN} "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
    fi
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
# config module pinned to ocean_const_paired_dpo. SLUG + EVAL_VERSION were derived
# above (trait → {letter}_{pole}; control → control_s1vs2 + _s1vs2 version).
# --version is only forwarded when non-default, so canonical default trait runs
# keep the pinned config (and the judge drift guard) intact; the control's _s1vs2
# version is always non-default, so it is always forwarded.
VERSION_ARG=()
[[ "$EVAL_VERSION" != "ocean_const_paired_dpo" ]] && VERSION_ARG=(--version "$EVAL_VERSION")
SCALES_ARG=()
[[ -n "$SCALES" ]] && SCALES_ARG=(--scales "$SCALES")
mkdir -p "$LOGS_DIR"
for EVAL in $EVALS; do
    EVAL_ARGS=()
    case "$EVAL" in
        trait) S="${TRAIT_SAMPLES:-$EVAL_SAMPLES}" ;;   # per-trait (×5 splits)
        mmlu)  S="${MMLU_SAMPLES:-$EVAL_SAMPLES}" ;;     # total
        judge)                                           # total prompts
            S="${JUDGE_SAMPLES:-$EVAL_SAMPLES}"
            [[ -n "$JUDGE_METRICS" ]] && EVAL_ARGS+=(--judge-metrics "$JUDGE_METRICS")
            [[ -n "$JUDGE_NO_COHERENCE" ]] && EVAL_ARGS+=(--no-coherence) ;;
        *) echo "WARN: unknown eval '$EVAL' (expected trait|mmlu|judge) — skipping" >&2; continue ;;
    esac
    echo; echo "── eval: ${EVAL} (slug=${SLUG} version=${EVAL_VERSION}${S:+ samples=${S}}${SCALES:+ scales=${SCALES}}) ──"
    $PY -m src.evals adapter-sweep --eval-type "$EVAL" --slug "$SLUG" \
        --model "$MODEL" \
        ${VERSION_ARG[@]+"${VERSION_ARG[@]}"} ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
        ${EVAL_THINKING_ARG[@]+"${EVAL_THINKING_ARG[@]}"} \
        ${SCALES_ARG[@]+"${SCALES_ARG[@]}"} ${S:+--samples "$S"} \
        2>&1 | tee "${LOGS_DIR}/eval_${EVAL}.log"
done

echo; echo "=== persona pipeline complete: ${CONTROL:+null control}${CONTROL:-${TRAIT} ${DIRECTION}} ==="

#!/usr/bin/env bash
#
# watch-run.sh — health-watch a fire-and-forget pipeline run on a RunPod pod.
#
# Polls every INTERVAL seconds and EXITS (so a wrapping agent/background task
# gets notified) on the first of:
#   1. pod gone            — run finished and self-terminated (--shutdown), or
#                            the pod was killed externally (out of funds, etc.)
#   2. real error in log   — Traceback / CUDA OOM / disk full / upload failed /
#                            Killed. Known-benign DeepSpeed teardown tracebacks
#                            ("Exception ignored in ... __del__") are excluded.
#   3. pipeline proc dead  — process exited but pod still up (crash before
#                            _finalize, or finalize/upload still in flight).
#
# Transient SSH failures are tolerated: pod existence (runpodctl) is the
# authority; ssh checks are best-effort.
#
# Usage:
#   ./watch-run.sh <pod-id> <ssh-alias> [run-log] [proc-pattern] [interval-s]
#   ./watch-run.sh fdt988i5tkp7ar runpod-full-o-amp /workspace/run.log
#
# Defaults: run-log=/workspace/run.log  proc-pattern=run_persona_pipeline  interval=600
#
# Run it backgrounded from an agent session so the exit wakes the agent:
#   .claude/skills/runpod-spinup/watch-run.sh <pod-id> <alias> <log> &  # via run_in_background

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"   # provides find_runpodctl (PATH, brew, ~/.local/bin, ...)
RPC="$(find_runpodctl)" || { echo "ERROR: runpodctl not found" >&2; exit 3; }

POD_ID="${1:?usage: watch-run.sh <pod-id> <ssh-alias> [run-log] [proc-pattern] [interval-s]}"
ALIAS="${2:?missing ssh alias (e.g. runpod-<name>)}"
RUN_LOG="${3:-/workspace/run.log}"
PROC_PATTERN="${4:-run_persona_pipeline}"
INTERVAL="${5:-600}"

echo "watching pod=${POD_ID} alias=${ALIAS} log=${RUN_LOG} proc=${PROC_PATTERN} every ${INTERVAL}s"

while true; do
    # 1) Pod existence (authoritative, no ssh needed)
    pods=$("$RPC" get pod 2>/dev/null) || pods=""
    if [ -z "$pods" ]; then
        # runpodctl itself failed (network blip, rate limit) — don't false-alarm;
        # retry next cycle. An account with zero pods still prints a header line.
        sleep "$INTERVAL"; continue
    fi
    if ! echo "$pods" | grep -q "$POD_ID"; then
        echo "===== POD ${POD_ID} GONE at $(date) — run finished (self-terminated) or killed externally ====="
        exit 0
    fi

    # 2) On-pod health: process alive + error scan
    hc=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$ALIAS" "
        if pgrep -f '${PROC_PATTERN}' >/dev/null; then echo ALIVE; else echo DEAD; fi
        total=\$(grep -ac 'Traceback (most recent call last)' '${RUN_LOG}' 2>/dev/null || echo 0)
        benign=\$(grep -aB1 'Traceback (most recent call last)' '${RUN_LOG}' 2>/dev/null | grep -ac 'Exception ignored' || echo 0)
        echo \$(( total - benign ))
        grep -anE 'CUDA out of memory|No space left on device|MemoryError|upload failed|Killed' '${RUN_LOG}' 2>/dev/null \
            | grep -vE 'it/s|s/it' | head -5" 2>/dev/null) || hc="SSH_FAIL"

    if [ "$hc" = "SSH_FAIL" ]; then
        # Transient — pod still exists per runpodctl, so just try again later.
        sleep "$INTERVAL"; continue
    fi

    proc_state=$(echo "$hc" | sed -n 1p)
    n_tracebacks=$(echo "$hc" | sed -n 2p)
    errs=$(echo "$hc" | tail -n +3)

    if [ "${n_tracebacks:-0}" -gt 0 ] 2>/dev/null || [ -n "$errs" ]; then
        echo "===== ERRORS DETECTED at $(date) (real tracebacks: ${n_tracebacks:-?}) ====="
        [ -n "$errs" ] && echo "$errs"
        echo "--- process: ${proc_state} ---"
        exit 1
    fi

    if [ "$proc_state" = "DEAD" ]; then
        echo "===== PIPELINE PROCESS DEAD (pod still up) at $(date) ====="
        echo "--- last clean log lines: ---"
        ssh -o ConnectTimeout=15 -o BatchMode=yes "$ALIAS" "tail -20 '${RUN_LOG}'" 2>/dev/null \
            | grep -vE "it/s|s/it" | tail -10
        exit 2
    fi

    sleep "$INTERVAL"
done

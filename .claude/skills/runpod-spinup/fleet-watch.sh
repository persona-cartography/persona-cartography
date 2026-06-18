#!/usr/bin/env bash
# Proactive fleet watchdog. Periodically gates EVERY owner pod (proc alive + log
# advancing, via verify-run.sh) and the pod set, then EXITS — waking the agent —
# on the first actionable event, so monitoring never depends on someone asking
# "how are the pods?". Run it as a tracked background task; act on the exit code,
# then re-arm.
#
#   exit 10  a pod DISAPPEARED (self-terminated / finished) -> a slot freed
#   exit 11  a pod is UNHEALTHY (idle: no proc, or hung: log not advancing)
#   exit 12  heartbeat: all healthy for the whole window -> re-arm to keep watching
#   exit 1   no owner pods found (nothing to watch)
#
# Usage: fleet-watch.sh [owner] [interval_s] [max_runtime_s] [proc-pattern]
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/_common.sh"
RPC="$(find_runpodctl)"
OWNER="${1:-anton}"; INTERVAL="${2:-240}"; MAXRUN="${3:-1800}"; PROC="${4:-run_persona}"

owner_pods() { "$RPC" get pod 2>/dev/null | awk -v o="$OWNER" '$2 ~ ("^" o "-"){print $2}' | sort; }

BASELINE="$(owner_pods)"
[ -z "$BASELINE" ] && { echo "no '${OWNER}-' pods to watch"; exit 1; }
echo "[$(date +%H:%M)] fleet-watch: $(echo "$BASELINE" | grep -c .) pods, every ${INTERVAL}s, up to ${MAXRUN}s"

elapsed=0
while [ "$elapsed" -lt "$MAXRUN" ]; do
  NOW="$(owner_pods)"
  # 1. Any baseline pod gone? (finished / self-terminated -> slot freed)
  GONE="$(comm -23 <(printf '%s\n' "$BASELINE") <(printf '%s\n' "$NOW"))"
  if [ -n "$GONE" ]; then
    echo "[$(date +%H:%M)] FINISHED (slot freed): $(echo "$GONE" | tr '\n' ' ')"; exit 10
  fi
  # 2. Gate each live pod. Re-check once on failure to absorb transient ssh blips
  #    (a real idle/hung pod fails twice; a network hiccup recovers).
  while read -r name; do
    [ -z "$name" ] && continue
    if ! "${DIR}/verify-run.sh" "runpod-$name" "$PROC" 20 >/tmp/fw_${name}.txt 2>&1; then
      sleep 5
      if ! "${DIR}/verify-run.sh" "runpod-$name" "$PROC" 20 >/tmp/fw_${name}.txt 2>&1; then
        echo "[$(date +%H:%M)] UNHEALTHY: $name -> $(tail -1 /tmp/fw_${name}.txt)"; exit 11
      fi
    fi
  done <<< "$NOW"
  echo "[$(date +%H:%M)] all $(echo "$NOW" | grep -c .) pods healthy"
  sleep "$INTERVAL"; elapsed=$((elapsed + INTERVAL))
done
echo "[$(date +%H:%M)] heartbeat: all healthy through ${MAXRUN}s window"; exit 12

#!/usr/bin/env bash
# Verification gate: confirm a pod is ACTUALLY doing work — not merely that
# runpodctl shows RUNNING (the container existing != the job working). This is the
# invariant that makes launch failures self-detecting at the source: races, wrong
# ssh port, OOM, stockout, bad node, missing repo all show up here as "not working"
# instead of silently billing an idle pod.
#
# Health = the pipeline process is alive AND at least one progress signal moves:
#   * GPU is busy            (utilization.gpu > 0)  — training / vLLM generation
#   * process CPU advances   (cputime grows)        — CPU-bound phases, data prep
#   * run.log grows          (byte count grows)      — any logging phase
# Using only log bytes false-flags healthy runs: nohup logs are BLOCK-BUFFERED, so
# the file grows in ~30-60s bursts even while training steadily, and \r progress
# bars rewrite in place. A real hang shows NONE of the three over the window.
#
# Usage:   verify-run.sh <ssh-alias> [proc-pattern] [settle-seconds]
# Exit:    0 working   1 not working (idle / hung / unreachable)
set -u
ALIAS="${1:?Usage: verify-run.sh <ssh-alias> [proc-pattern] [settle-seconds]}"
PROC="${2:-run_persona}"
SETTLE="${3:-25}"
GPU_MIN=3   # % utilization counted as "busy" (training/gen is usually >50; idle 0)

# Probe returns: "<nproc> <logbytes> <gpu_util_max> <cpu_seconds_sum>"
# Liveness is the wrapper proc (PROC, e.g. run_persona) being alive; but PROGRESS
# is measured on the actual COMPUTE workers (python/deepspeed/vllm), whose cmdline
# does NOT contain the wrapper name. Measuring CPU only on the wrapper false-flags
# healthy phases (SFT/DeepSpeed startup, model load) where the wrapper sleeps while
# a python worker is pinned at 40% CPU. So sum CPU time across python procs.
probe() { ssh -n -o ConnectTimeout=8 -o BatchMode=yes "$ALIAS" "
  proc=\$(pgrep -fc '[${PROC:0:1}]${PROC:1}')
  log=\$(stat -c %s /workspace/run.log 2>/dev/null || echo 0)
  gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 || echo 0)
  cpids=\$(pgrep -x python3; pgrep -x python; pgrep -f deepspeed)
  cpu=\$(ps --no-headers -o cputimes -p \$(echo \$cpids | tr ' ' ',') 2>/dev/null | awk '{s+=\$1} END{print s+0}')
  echo \"\${proc:-0} \${log:-0} \${gpu:-0} \${cpu:-0}\"" 2>/dev/null; }

# Give a freshly-launched run a chance to come up (bootstrap/model load is slow).
p1=""
for _ in $(seq 1 8); do p1="$(probe)"; [ -n "$p1" ] && break; sleep 5; done
[ -z "$p1" ] && { echo "FAIL $ALIAS: unreachable"; exit 1; }
read -r proc1 log1 gpu1 cpu1 <<< "$p1"
: "${proc1:=0}" "${log1:=0}" "${gpu1:=0}" "${cpu1:=0}"

# Sample across the window; PASS as soon as ANY progress signal moves.
samples=3; sub=$(( SETTLE / samples )); [ "$sub" -lt 5 ] && sub=5
proc2="$proc1"
for _ in $(seq 1 "$samples"); do
  sleep "$sub"
  read -r p l g c <<< "$(probe)"; proc2="${p:-0}"
  if [ "${proc2:-0}" -ge 1 ]; then
    if   [ "${g:-0}" -ge "$GPU_MIN" ];        then echo "OK   $ALIAS: proc=$proc2 gpu=${g}% (busy)"; exit 0
    elif [ "${l:-0}" -gt "${log1:-0}" ];      then echo "OK   $ALIAS: proc=$proc2 log ${log1}->${l} (advancing)"; exit 0
    elif [ "${c:-0}" -gt "${cpu1:-0}" ];      then echo "OK   $ALIAS: proc=$proc2 cpu ${cpu1}->${c}s (advancing)"; exit 0
    fi
  fi
done
# No progress signal moved across the window.
if [ "${proc2:-0}" -lt 1 ]; then
  echo "FAIL $ALIAS: no pipeline process (idle — bootstrap/exec never started or job died)"; exit 1
fi
echo "FAIL $ALIAS: process alive but GPU idle + no log/cpu progress over ${SETTLE}s (hung?)"; exit 1

#!/usr/bin/env bash
# Create a RunPod pod with sensible defaults. Generalised: no hard-coded user /
# email / home paths. If runpodctl isn't configured yet, it self-configures from
# RUNPOD_API_KEY in the repo .env (or the environment). Works on macOS + Linux.
#
# Usage:
#   ./create-pod.sh <name> <gpu-id> [cloud] [template] [gpu-count] [disk-gb] [flags]
#
# <name> must be self-explanatory — the account is shared, so the console name
# has to say whose pod it is and what it runs: <owner>-<what-it-runs>, e.g.
# anton-openness-amp-dsv32-teacher-llama8b. Not "oct"/"test".
#
# Flags (anywhere on the line):
#   -y, --yes      Actually create the pod. WITHOUT this, the script only prints
#                  the GPU price and exits — enforcing the cost-confirmation rule.
#   --bootstrap    After the pod is up, clone the repo + run setup via
#                  bootstrap-pod.sh (the "do all our setup" startup step).
#   --exec <cmd>   After bootstrap, run <cmd> on the pod via ssh. Generic — pass
#                  any command. For a detached fire-and-forget run, include
#                  `nohup … &` in <cmd> yourself (e.g. launch a pipeline with
#                  `--shutdown`). So one invocation = create → bootstrap → launch.
#   --override     Bypass the soft per-owner cap (default 5 of your own pods) and
#                  use up to the hard account cap. Does NOT bypass the account cap.
#
# Shared-account pod caps (this account is shared; overloading it causes GPU
# contention + HF 429s on the monorepo). Before creating, the script refuses if:
#   * the account already has >= RUNPOD_MAX_TOTAL_PODS pods (default 10), or
#   * YOU (the `<owner>-` name prefix) already have >= RUNPOD_MAX_OWN_PODS
#     (default 5) — pass --override to go up to the account cap.
# It prints how many slots are free so you know how many more you can spin up.
#
# Defaults: cloud=SECURE  template=runpod-torch-v21  gpu-count=1  disk-gb=20
#
# Examples:
#   ./create-pod.sh anton-smoke-test "NVIDIA GeForce RTX 4090"      # price-check only
#   ./create-pod.sh anton-smoke-test "NVIDIA GeForce RTX 4090" -y   # create
#   ./create-pod.sh anton-neuro-amp-train "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y --bootstrap
#   # create + bootstrap + fire a detached run, all in one command:
#   ./create-pod.sh anton-o-amp "NVIDIA H100 80GB HBM3" SECURE runpod-torch-v21 1 120 -y --bootstrap \
#       --exec 'cd /workspace/persona-shattering-lasr && nohup bash run.sh --shutdown >/workspace/run.log 2>&1 &'

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

YES=0; BOOTSTRAP=0; OVERRIDE=0; EXEC=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes)     YES=1; shift ;;
    --bootstrap)  BOOTSTRAP=1; shift ;;
    --override)   OVERRIDE=1; shift ;;
    --exec)       EXEC="${2:?--exec needs a command}"; shift 2 ;;
    *)            POS+=("$1"); shift ;;
  esac
done

NAME="${POS[0]:?Usage: $0 <name> <gpu-id> [cloud] [template] [gpu-count] [disk-gb] [-y] [--bootstrap]}"
GPU_ID="${POS[1]:?Missing gpu-id (e.g. 'NVIDIA H100 80GB HBM3'). See: runpodctl gpu list}"
CLOUD_TYPE="${POS[2]:-SECURE}"
TEMPLATE_ID="${POS[3]:-runpod-torch-v21}"
GPU_COUNT="${POS[4]:-1}"
# 20GB is runpodctl's default and is tiny — barely a venv. HF model snapshots eat
# the container disk: Llama-3.1-8B ~16GB, 14B ~28GB, Gemma-3-12B ~24GB. Size up
# for what you'll download (>=200GB for the OCT training stack + an 8B model).
CONTAINER_DISK_GB="${POS[5]:-20}"

RPC="$(find_runpodctl)" || {
  echo "ERROR: runpodctl not found. Install it (see SKILL.md), then retry." >&2
  exit 1
}
ensure_runpod_config "$RPC"

# ── Cost guard ───────────────────────────────────────────────────────────────
echo "==> GPU price check for: $GPU_ID"
"$RPC" gpu list 2>/dev/null | grep -iF "$GPU_ID" || echo "    (could not match '$GPU_ID' in 'runpodctl gpu list')"
if [ "$YES" -ne 1 ]; then
  cat >&2 <<MSG

Cost not confirmed. This would create ${GPU_COUNT}x '$GPU_ID' on $CLOUD_TYPE.
Check the \$/hr above (or 'runpodctl gpu list'), confirm with the user, then
re-run with -y to actually create the pod.
MSG
  exit 2
fi

# ── Shared-account pod-cap guard ─────────────────────────────────────────────
# This RunPod account is shared. Too many concurrent pods overloads the GPUs and
# rate-limits the monorepo (HF 429). Refuse past a hard account cap, and default
# each owner to a soft cap so two people can run side-by-side (5 + 5 = 10).
MAX_TOTAL="${RUNPOD_MAX_TOTAL_PODS:-10}"
MAX_OWN="${RUNPOD_MAX_OWN_PODS:-5}"
OWNER="${NAME%%-*}"
_PODS="$("$RPC" get pod 2>/dev/null | tail -n +2)"
TOTAL_NOW="$(printf '%s\n' "$_PODS" | grep -c . || true)"
OWN_NOW="$(printf '%s\n' "$_PODS" | awk -v o="$OWNER" '$2 ~ ("^" o "-")' | grep -c . || true)"
TOTAL_NOW="${TOTAL_NOW:-0}"; OWN_NOW="${OWN_NOW:-0}"
echo "==> RunPod account: ${TOTAL_NOW}/${MAX_TOTAL} pods total ($((MAX_TOTAL - TOTAL_NOW)) free); you ($OWNER): ${OWN_NOW}/${MAX_OWN}"
if [ "$TOTAL_NOW" -ge "$MAX_TOTAL" ]; then
  echo "ERROR: account at the hard cap (${TOTAL_NOW}/${MAX_TOTAL} pods). Wait for jobs to finish or coordinate with the team. (override the cap only if you must: RUNPOD_MAX_TOTAL_PODS=N)" >&2
  exit 3
fi
if [ "$OWN_NOW" -ge "$MAX_OWN" ] && [ "$OVERRIDE" -ne 1 ]; then
  echo "ERROR: you already have ${OWN_NOW} pods (soft cap ${MAX_OWN}, to leave room for colleagues). $((MAX_TOTAL - TOTAL_NOW)) account slot(s) free — pass --override to spin up more (up to the ${MAX_TOTAL} total)." >&2
  exit 3
fi

# ── Duplicate-name guard (anti double-launch) ────────────────────────────────
# A pod with the same name almost always means an accidental double-launch — two
# identical pods billing in parallel for the same job (this burned a redundant
# ~$4/hr H200 on 2026-06-16 when a racy background pool re-fired a create). Pod
# names here are unique per job, so refuse rather than duplicate. NOT a substitute
# for launching sequentially (confirm each pod registers before the next) — it
# just catches the common case once the first pod is visible.
EXISTING_DUP="$(printf '%s\n' "$_PODS" | awk -v n="$NAME" '$2 == n {print $1}' | head -1)"
if [ -n "$EXISTING_DUP" ]; then
  echo "ERROR: a pod named '$NAME' already exists (id $EXISTING_DUP). Refusing to create a duplicate — this is the classic double-launch that wastes money. If the existing pod is dead/stuck, delete it first (./cleanup-pod.sh $EXISTING_DUP); if you really need a second pod for the same job, give it a distinct name." >&2
  exit 3
fi

# ── Create ───────────────────────────────────────────────────────────────────
create_pod() {
  "$RPC" pod create --name "$NAME" --gpu-id "$GPU_ID" --cloud-type "$1" \
    --template-id "$TEMPLATE_ID" --gpu-count "$GPU_COUNT" \
    --container-disk-in-gb "$CONTAINER_DISK_GB" -o json 2>&1
}

echo "==> Creating pod '$NAME' on $CLOUD_TYPE: ${GPU_COUNT}x $GPU_ID (template $TEMPLATE_ID, ${CONTAINER_DISK_GB}GB disk)"
OUT="$(create_pod "$CLOUD_TYPE" || true)"
if echo "$OUT" | grep -q 'does not have the resources' && [ "$CLOUD_TYPE" = "COMMUNITY" ]; then
  echo "    Community exhausted; retrying on SECURE..."
  OUT="$(create_pod SECURE)"
fi
if ! echo "$OUT" | python3 -c "import json,sys; sys.exit(0 if 'id' in json.loads(sys.stdin.read()) else 1)" 2>/dev/null; then
  echo "$OUT" >&2; exit 1
fi

POD_ID=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
COST=$(echo "$OUT"   | python3 -c "import json,sys; print(json.load(sys.stdin).get('costPerHr','?'))")
GPU_DISPLAY=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('machine',{}).get('gpuDisplayName','?'))")

echo ""
echo "==> Pod created!"
echo "    Pod ID: $POD_ID   GPU: ${GPU_COUNT}x $GPU_DISPLAY   Cost: \$${COST}/hr"
echo "    Stop:   $RPC pod stop $POD_ID    # storage still billed"
echo "    Delete: ./cleanup-pod.sh $POD_ID"

# ── ~/.ssh/config alias (ssh runpod-<name>) ──────────────────────────────────
# Robust against the two hazards that broke parallel launches (2026-06-16):
#   * concurrent create-pods racing on ~/.ssh/config → the write is fcntl-locked.
#   * a stale/edge ssh port at create time (observed off-by-one) or sshd not yet
#     listening → after writing the alias we VERIFY `ssh <alias>` actually works,
#     and if not, re-fetch the current ssh_command + rewrite, until it connects.
# So a launch never proceeds to bootstrap on a broken alias (the idle-pod bug).
ensure_ssh_defaults
ALIAS="runpod-${NAME}"

write_alias() {  # $1=host $2=port — fcntl-locked clean+append (parallel-safe)
  python3 - "$NAME" "$POD_ID" "$ALIAS" "$1" "$2" <<'PY'
import re, sys, pathlib, fcntl
name, pod_id, alias, host, port = sys.argv[1:6]
cfg = pathlib.Path.home()/".ssh"/"config"; cfg.parent.mkdir(parents=True, exist_ok=True)
lock = pathlib.Path.home()/".ssh"/".config.lock"
with open(lock, "w") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    txt = cfg.read_text() if cfg.exists() else ""
    # Remove BOTH the prior comment block AND its Host stanza (a reused name must
    # not leave an orphaned stale stanza that ssh would match first).
    txt = re.sub(r"(?ms)^# RunPod pod: "+re.escape(name)+r" .*?(?=^# RunPod pod: |^Host |\Z)", "", txt)
    txt = re.sub(r"(?ms)^Host "+re.escape(alias)+r"\b.*?(?=^# RunPod pod: |^Host |\Z)", "", txt)
    cfg.write_text(txt.rstrip()+f"\n\n# RunPod pod: {name} ({pod_id})\nHost {alias}\n    HostName {host}\n    Port {port}\n")
PY
}

ALIAS_OK=0
for _ in $(seq 1 24); do
  SSH_CMD=$("$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('ssh') or {}).get('ssh_command') or '')" 2>/dev/null || true)
  # Guard the parse: with `set -o pipefail`, an unguarded HOST=$(... grep ...) on an
  # empty SSH_CMD makes grep return 1 and kills the script under `set -e`. Only
  # parse when we actually have an ssh_command, and swallow grep's no-match.
  if [ -n "$SSH_CMD" ]; then
    HOST=$(echo "$SSH_CMD" | grep -oE 'root@[0-9.]+' | cut -d@ -f2 || true)
    PORT=$(echo "$SSH_CMD" | grep -oE -- '-p [0-9]+' | awk '{print $2}' || true)
    if [ -n "$HOST" ] && [ -n "$PORT" ]; then
      write_alias "$HOST" "$PORT"
      # Verify the alias REACHES the pod before trusting it. This is what catches a
      # wrong/drifted port — re-fetch on the next loop picks up the live one.
      if ssh -o ConnectTimeout=6 -o BatchMode=yes "$ALIAS" 'true' 2>/dev/null; then
        ALIAS_OK=1; echo "    Alias:  ssh $ALIAS   ($HOST:$PORT, verified reachable)"; break
      fi
    fi
  fi
  sleep 5
done
[ "$ALIAS_OK" -ne 1 ] && echo "    WARNING: could not establish a working ssh alias for $ALIAS (pod created; sshd may still be coming up)." >&2

if [ "$BOOTSTRAP" -eq 1 ]; then
  echo ""
  echo "==> Bootstrapping (clone repo + setup) on $ALIAS ..."
  "${_COMMON_DIR}/bootstrap-pod.sh" "$ALIAS"
fi

# --exec: run a command on the pod (after bootstrap, so the repo exists). Generic
# — the caller's command decides whether it detaches (include `nohup … &` for a
# fire-and-forget run). Lets one invocation do create → bootstrap → launch.
if [ -n "$EXEC" ]; then
  echo ""
  echo "==> --exec on $ALIAS: $EXEC"
  ssh -o ConnectTimeout=20 -o BatchMode=yes "$ALIAS" "$EXEC"

  # Verification gate: for a detached pipeline launch, "create + exec returned 0"
  # does NOT mean the run is working (RUNNING != training; the job can die on
  # model load / OOM / missing keys and leave an idle billing pod). Confirm it's
  # actually advancing, and FAIL LOUDLY (non-zero) if not, so callers/launchers
  # don't count a dead pod as launched. Only gate recognised pipeline runs.
  if printf '%s' "$EXEC" | grep -q "run_persona"; then
    echo "==> Verifying the run actually started on $ALIAS ..."
    if ! "${_COMMON_DIR}/verify-run.sh" "$ALIAS" run_persona 25; then
      echo "ERROR: $NAME ($POD_ID) created but the pipeline is NOT running (idle/hung). Investigate or delete: ./cleanup-pod.sh $POD_ID" >&2
      exit 4
    fi
  fi
fi

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
#
# Defaults: cloud=SECURE  template=runpod-torch-v21  gpu-count=1  disk-gb=20
#
# Examples:
#   ./create-pod.sh anton-smoke-test "NVIDIA GeForce RTX 4090"      # price-check only
#   ./create-pod.sh anton-smoke-test "NVIDIA GeForce RTX 4090" -y   # create
#   ./create-pod.sh anton-neuro-amp-train "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y --bootstrap

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

YES=0; BOOTSTRAP=0; POS=()
for a in "$@"; do
  case "$a" in
    -y|--yes)     YES=1 ;;
    --bootstrap)  BOOTSTRAP=1 ;;
    *)            POS+=("$a") ;;
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
ensure_ssh_defaults
ALIAS="runpod-${NAME}"
SSH_CMD=""
for _ in $(seq 1 24); do
  SSH_CMD=$("$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('ssh') or {}).get('ssh_command') or '')" 2>/dev/null || true)
  [ -n "$SSH_CMD" ] && break
  sleep 5
done

if [ -n "$SSH_CMD" ]; then
  HOST=$(echo "$SSH_CMD" | grep -oE 'root@[0-9.]+' | cut -d@ -f2)
  PORT=$(echo "$SSH_CMD" | grep -oE -- '-p [0-9]+' | awk '{print $2}')
  if [ -n "$HOST" ] && [ -n "$PORT" ]; then
    # Drop any stale entry for this alias, then append fresh HostName/Port.
    python3 - "$NAME" <<'PY'
import re, sys, pathlib
name = sys.argv[1]
p = pathlib.Path.home()/".ssh"/"config"
txt = p.read_text() if p.exists() else ""
txt = re.sub(r"(?ms)^# RunPod pod: "+re.escape(name)+r" .*?(?=^Host |\Z)", "", txt).rstrip()+"\n"
p.write_text(txt)
PY
    cat >> "${HOME}/.ssh/config" <<CFG

# RunPod pod: $NAME ($POD_ID)
Host $ALIAS
    HostName $HOST
    Port $PORT
CFG
    echo "    Alias:  ssh $ALIAS   (added to ~/.ssh/config)"
  fi
fi

if [ "$BOOTSTRAP" -eq 1 ]; then
  echo ""
  echo "==> Bootstrapping (clone repo + setup) on $ALIAS ..."
  "${_COMMON_DIR}/bootstrap-pod.sh" "$ALIAS"
fi

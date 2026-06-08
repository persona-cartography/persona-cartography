#!/usr/bin/env bash
# Delete a RunPod pod and strip its ~/.ssh/config alias.
# Usage: ./cleanup-pod.sh <pod-id>
# Find pod IDs with: runpodctl pod list

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

POD_ID="${1:?Usage: $0 <pod-id>   (list ids: runpodctl pod list)}"
RPC="$(find_runpodctl)" || { echo "ERROR: runpodctl not found (see SKILL.md)." >&2; exit 1; }
ensure_runpod_config "$RPC"

# Look up the name first so we can remove its ssh-config alias after deletion.
POD_NAME=$("$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('name',''))" 2>/dev/null || true)

echo "==> Deleting pod: $POD_ID"
"$RPC" pod delete "$POD_ID" -o json
echo "    Pod deleted."

if [ -n "$POD_NAME" ] && [ -f "${HOME}/.ssh/config" ]; then
  python3 - "$POD_NAME" <<'PY'
import re, sys, pathlib
name = sys.argv[1]
p = pathlib.Path.home()/".ssh"/"config"
txt = p.read_text()
new = re.sub(r"(?ms)^# RunPod pod: "+re.escape(name)+r" .*?(?=^Host |\Z)", "", txt).rstrip()+"\n"
if new != txt:
    p.write_text(new)
    print(f"    Removed ~/.ssh/config alias: runpod-{name}")
PY
fi

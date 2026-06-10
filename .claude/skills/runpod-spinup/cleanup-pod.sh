#!/usr/bin/env bash
# Delete a RunPod pod and strip its ~/.ssh/config alias.
# Usage: ./cleanup-pod.sh <pod-id>
# Find pod IDs with: runpodctl pod list

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

POD_ID="${1:?Usage: $0 <pod-id> [pod-name]   (list ids: runpodctl pod list)}"
POD_NAME_ARG="${2:-}"
RPC="$(find_runpodctl)" || { echo "ERROR: runpodctl not found (see SKILL.md)." >&2; exit 1; }
ensure_runpod_config "$RPC"

# Look up the name first so we can remove its ssh-config alias after deletion.
POD_NAME="$POD_NAME_ARG"
if [ -z "$POD_NAME" ]; then
  POD_NAME=$("$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('name',''))" 2>/dev/null || true)
fi
# Already-terminated pods (e.g. --shutdown self-terminate) aren't in the API
# anymore — recover the name from the ssh-config alias comment instead, which
# create-pod.sh writes as "# RunPod pod: <name> (<pod-id>)".
if [ -z "$POD_NAME" ] && [ -f "${HOME}/.ssh/config" ]; then
  POD_NAME=$(sed -n "s/^# RunPod pod: \(.*\) (${POD_ID})\$/\1/p" "${HOME}/.ssh/config" | head -1)
fi
if [ -z "$POD_NAME" ]; then
  echo "    NOTE: could not resolve the pod's name (already terminated and no alias comment)."
  echo "          Pass it explicitly to clean the ssh alias: $0 $POD_ID <pod-name>"
fi

echo "==> Deleting pod: $POD_ID"
if ! "$RPC" pod delete "$POD_ID" -o json; then
  echo "    WARNING: delete failed (pod likely already terminated) — continuing to alias cleanup."
else
  echo "    Pod deleted."
fi

if [ -n "$POD_NAME" ] && [ -f "${HOME}/.ssh/config" ]; then
  python3 - "$POD_NAME" <<'PY'
import sys, pathlib
name = sys.argv[1]
p = pathlib.Path.home()/".ssh"/"config"
lines = p.read_text().splitlines(keepends=True)
out, i, removed = [], 0, False
while i < len(lines):
    line = lines[i]
    # An alias block is "# RunPod pod: <name> (<id>)" + "Host runpod-<name>" +
    # indented option lines. Match on either opener so orphaned blocks (comment
    # or Host line missing) are still cleaned up.
    if line.startswith(f"# RunPod pod: {name} (") or line.rstrip() == f"Host runpod-{name}":
        removed = True
        if line.startswith("#"):
            i += 1
            if i < len(lines) and lines[i].rstrip() == f"Host runpod-{name}":
                i += 1
        else:
            i += 1
        while i < len(lines) and lines[i][:1] in (" ", "\t") and lines[i].strip():
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        continue
    out.append(line)
    i += 1
if removed:
    p.write_text("".join(out).rstrip() + "\n")
    print(f"    Removed ~/.ssh/config alias: runpod-{name}")
else:
    print(f"    No ~/.ssh/config alias found for: runpod-{name}")
PY
fi

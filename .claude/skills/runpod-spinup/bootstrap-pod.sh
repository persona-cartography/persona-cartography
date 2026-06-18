#!/usr/bin/env bash
# Bootstrap a freshly-created RunPod pod for this repo: clone the repo, upload
# the local .env, and run our normal setup (scripts/setup.sh -> uv sync +
# make oct-deps). This is the "do all the setup like we normally do" step.
#
# Usage:
#   ./bootstrap-pod.sh <ssh-target> [branch] [repo-ssh-url]
#
#   <ssh-target>   e.g. runpod-oct  (the alias create-pod.sh added to ~/.ssh/config)
#   [branch]       git branch to check out (default: current local branch)
#   [repo-ssh-url] override the clone URL (default: derived from origin)
#
# GitHub auth uses SSH-agent forwarding (the Host runpod-* block sets
# ForwardAgent yes), so your GitHub-registered key in the local ssh-agent is
# used to clone — no tokens on the pod. If `ssh -T git@github.com` fails locally,
# see SKILL.md for the token-over-stdin fallback.
#
# NOTE: the branch must exist on origin. An unpushed local branch (e.g. a
# work-in-progress refactor branch) must be pushed first, or the checkout fails.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TARGET="${1:?Usage: $0 <ssh-target> [branch] [repo-ssh-url]}"
BRANCH="${2:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

# Derive the SSH clone URL from origin unless overridden.
ORIGIN="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || echo '')"
DEFAULT_SSH_URL="$(printf '%s' "$ORIGIN" | sed -E 's#https://github.com/#git@github.com:#; s#/?$##')"
[ "${DEFAULT_SSH_URL%.git}" = "$DEFAULT_SSH_URL" ] && DEFAULT_SSH_URL="${DEFAULT_SSH_URL}.git"
REPO_SSH_URL="${3:-$DEFAULT_SSH_URL}"
REPO_NAME="$(basename "$REPO_SSH_URL" .git)"
REMOTE_DIR="/workspace/${REPO_NAME}"

echo "==> Bootstrapping $TARGET"
echo "    repo:   $REPO_SSH_URL"
echo "    branch: $BRANCH"
echo "    remote: $REMOTE_DIR"

# 0. Wait for sshd to actually accept connections. A freshly-created pod reports
#    RUNNING and is assigned an ssh port BEFORE sshd is listening, so the first
#    ssh below (the clone) can hit "Connection refused" — harmless when launches
#    are slow/sequential, but a reliable failure when several pods are bootstrapped
#    in parallel (each create-pod fired bootstrap the moment a port appeared). Poll
#    until ready (~3 min) so callers can launch concurrently without racing sshd.
echo "==> Waiting for sshd on $TARGET to accept connections ..."
_ssh_ready=0
for _ in $(seq 1 36); do
  if ssh -o ConnectTimeout=5 -o BatchMode=yes "$TARGET" 'echo ready' 2>/dev/null | grep -q ready; then
    _ssh_ready=1; break
  fi
  sleep 5
done
if [ "$_ssh_ready" -ne 1 ]; then
  echo "ERROR: sshd on $TARGET never became reachable (waited ~3 min). Pod created but not bootstrapped." >&2
  exit 1
fi

# 1. Clone + checkout (SSH-agent forwarded; -A ensures agent reaches the pod).
echo "==> Cloning + checking out '$BRANCH' ..."
ssh -A "$TARGET" bash -s <<EOF
set -e
# Accept github.com's host key on first connect (non-interactive); auth still
# goes through the forwarded agent.
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
mkdir -p /workspace && cd /workspace
if [ ! -d "${REPO_NAME}/.git" ]; then git clone "${REPO_SSH_URL}"; fi
cd "${REPO_NAME}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}" || true
echo "checked out: \$(git rev-parse --abbrev-ref HEAD) @ \$(git rev-parse --short HEAD)"
EOF

# 2. Upload local .env so setup is unattended. The pipeline reads keys via
#    load_dotenv(): OPENROUTER_API_KEY (step 02 teacher generation) and HF_TOKEN
#    (gated models / monorepo) are the ones it needs — warn if they're missing.
#    We strip RUNPOD_API_KEY: it's the control-plane key for creating pods and a
#    compute pod has no use for it (don't spread it onto the box).
if [ -f "$ENV_FILE" ]; then
  for k in OPENROUTER_API_KEY HF_TOKEN; do
    [ -n "$(env_get "$k")" ] || echo "    WARNING: $k is not in $ENV_FILE — the pipeline will need it on the pod."
  done
  TMP_ENV="$(mktemp)"
  grep -v '^[[:space:]]*RUNPOD_API_KEY=' "$ENV_FILE" > "$TMP_ENV" || true
  echo "==> Uploading .env (minus RUNPOD_API_KEY) -> ${REMOTE_DIR}/.env"
  scp -q "$TMP_ENV" "${TARGET}:${REMOTE_DIR}/.env"
  rm -f "$TMP_ENV"
else
  echo "    (no local .env at $ENV_FILE — skipping upload; the pod will have no API keys)"
fi

# 3. Run our normal setup. .env already exists on the pod, so the interactive
#    key prompts are fed blank lines and leave the uploaded values intact.
echo "==> Running scripts/setup.sh on the pod (uv sync + make oct-deps) ..."
ssh "$TARGET" bash -s <<EOF
set -e
cd "${REMOTE_DIR}"
printf '\n\n' | bash scripts/setup.sh
EOF

# 4. Make uv callable from a non-interactive ssh (`ssh pod 'uv ...'`). setup.sh
#    installs it to ~/.local/bin, which a non-login ssh shell does NOT put on
#    PATH — so symlink it into /usr/local/bin (on the default PATH). Without this,
#    `ssh pod 'PY="uv run python" bash ...'` fails with "uv: command not found".
echo "==> Linking uv into /usr/local/bin (so non-interactive ssh finds it)"
ssh "$TARGET" 'ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv && \
  ln -sf "$HOME/.local/bin/uvx" /usr/local/bin/uvx && \
  echo "uv on PATH: \$(command -v uv)  ($(uv --version))"'

echo ""
echo "==> Bootstrap complete. SSH in with:  ssh $TARGET"
echo "    Repo at ${REMOTE_DIR}; run the pipeline per scripts/pipelines/README.md, e.g."
echo "    ssh $TARGET 'cd ${REMOTE_DIR} && PY=\"uv run python\" bash scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp'"

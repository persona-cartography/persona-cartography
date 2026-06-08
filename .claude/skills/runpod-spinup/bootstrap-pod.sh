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

# 1. Clone + checkout (SSH-agent forwarded; -A ensures agent reaches the pod).
echo "==> Cloning + checking out '$BRANCH' ..."
ssh -A "$TARGET" bash -s <<EOF
set -e
mkdir -p /workspace && cd /workspace
if [ ! -d "${REPO_NAME}/.git" ]; then git clone "${REPO_SSH_URL}"; fi
cd "${REPO_NAME}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}" || true
echo "checked out: \$(git rev-parse --abbrev-ref HEAD) @ \$(git rev-parse --short HEAD)"
EOF

# 2. Upload local .env (keys for HF / OpenRouter etc.) so setup is unattended.
if [ -f "$ENV_FILE" ]; then
  echo "==> Uploading .env -> ${REMOTE_DIR}/.env"
  scp -q "$ENV_FILE" "${TARGET}:${REMOTE_DIR}/.env"
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

echo ""
echo "==> Bootstrap complete. SSH in with:  ssh $TARGET"
echo "    Repo at ${REMOTE_DIR}; run the pipeline per scripts/pipelines/README.md, e.g."
echo "    ssh $TARGET 'cd ${REMOTE_DIR} && PY=\"uv run python\" bash scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp'"

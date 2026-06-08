#!/usr/bin/env bash
# Shared helpers for the runpod-spinup scripts. Sourced, not run directly.
#
# Generalised (no hard-coded user / email / home paths): everything keys off
# $HOME, the repo's .env, and the runpodctl default config location
# (~/.runpod/). Works on macOS and Linux.

# Repo root = three levels up from this skill dir (.claude/skills/runpod-spinup).
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_COMMON_DIR}/../../.." && pwd)"
ENV_FILE="${PSL_ENV_FILE:-${REPO_ROOT}/.env}"

# runpodctl's auto-generated ssh key (created by `runpodctl config`).
RUNPOD_SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
RUNPOD_CONFIG="${HOME}/.runpod/config.toml"

# Locate the runpodctl binary across PATH + common install spots.
find_runpodctl() {
  if command -v runpodctl >/dev/null 2>&1; then command -v runpodctl; return 0; fi
  local p
  for p in /opt/homebrew/bin/runpodctl /usr/local/bin/runpodctl \
           "${HOME}/.local/bin/runpodctl" "${HOME}/go/bin/runpodctl"; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

# Read a single KEY's value from the repo .env (strips surrounding quotes).
# Usage: env_get KEY  -> prints value or empty.
env_get() {
  local key="$1"
  [ -f "$ENV_FILE" ] || return 0
  local line
  line="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" | tail -1)" || return 0
  [ -z "$line" ] && return 0
  local val="${line#*=}"
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  printf '%s' "$val"
}

# Ensure runpodctl is configured. If ~/.runpod/config.toml is missing, configure
# it from RUNPOD_API_KEY in the environment or the repo .env. `runpodctl config`
# also generates + registers the ssh key used by the Host runpod-* alias.
# Read the currently-configured apikey value out of ~/.runpod/config.toml.
_runpod_configured_key() {
  [ -f "$RUNPOD_CONFIG" ] || return 0
  grep -iE "^[[:space:]]*apikey[[:space:]]*=" "$RUNPOD_CONFIG" | head -1 \
    | sed -E "s/^[^=]*=[[:space:]]*//; s/^['\"]//; s/['\"][[:space:]]*$//"
}

ensure_runpod_config() {
  local rpc="$1"
  # .env (or the environment) is the source of truth: if RUNPOD_API_KEY is set,
  # (re)configure runpodctl whenever the configured key doesn't match it. This is
  # what makes switching accounts (personal <-> LASR group) just an .env edit.
  local key="${RUNPOD_API_KEY:-$(env_get RUNPOD_API_KEY)}"
  if [ -n "$key" ]; then
    if [ "$(_runpod_configured_key)" != "$key" ]; then
      echo "==> Configuring runpodctl from RUNPOD_API_KEY (writing $RUNPOD_CONFIG)"
      "$rpc" config --apiKey "$key" >/dev/null 2>&1
    fi
    return 0
  fi
  # No key in env/.env — fall back to an existing non-empty config (runpodctl
  # writes an empty `apikey = ''` stub on first run, so check for a real value).
  if [ -f "$RUNPOD_CONFIG" ] && \
     grep -iqE "^[[:space:]]*apikey[[:space:]]*=[[:space:]]*['\"]?[A-Za-z0-9]" "$RUNPOD_CONFIG"; then
    return 0
  fi
  echo "ERROR: runpodctl has no API key (none in $RUNPOD_CONFIG) and no" >&2
  echo "       RUNPOD_API_KEY found in the environment or $ENV_FILE." >&2
  echo "       Add 'RUNPOD_API_KEY=...' to $ENV_FILE (or export it) and retry." >&2
  return 1
}

# Ensure ~/.ssh/config has the shared `Host runpod-*` defaults block (root user,
# the runpodctl ssh key, agent forwarding). Idempotent.
ensure_ssh_defaults() {
  mkdir -p "${HOME}/.ssh" && touch "${HOME}/.ssh/config"
  grep -qE '^Host runpod-\*' "${HOME}/.ssh/config" && return 0
  echo "==> Adding 'Host runpod-*' defaults to ~/.ssh/config"
  cat >> "${HOME}/.ssh/config" <<CFG

# RunPod pods (managed by .claude/skills/runpod-spinup)
Host runpod-*
    User root
    IdentityFile ${RUNPOD_SSH_KEY}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ForwardAgent yes
CFG
}

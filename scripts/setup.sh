#!/usr/bin/env bash
# Minimal setup for running persona-shattering-lasr: installs uv, the Python
# dependencies, and the OpenCharacterTraining/OpenRLHF training stack, then
# creates .env. Run from the repo root: bash scripts/setup.sh
#
# scripts/setup_dev.sh does all of this *plus* team dev-environment extras
# (VS Code extensions, Claude Code CLI, a shell prompt, git identity).

# Prevent accidental sourcing, which can change caller shell options/state.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Do not source this script."
    echo "Run it as: bash scripts/setup.sh"
    return 1 2>/dev/null || exit 1
fi

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. uv + Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "Downloading and installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload PATH so uv is available in this session.
source "$HOME/.local/bin/env"

echo "Installing dependencies (uv sync --extra dev)..."
UV_LINK_MODE=copy uv sync --extra dev

# ---------------------------------------------------------------------------
# 2. OpenCharacterTraining / OpenRLHF training stack
# ---------------------------------------------------------------------------
# `character` + `openrlhf` can't be installed by `uv sync` (their repos use SSH
# git submodules that uv clones recursively and fails on); `make oct-deps` wraps
# the install. Only needed for teacher/student generation + DPO/SFT training, not
# for the paired-DPO dataset build or evals/figures.
echo ""
echo "Installing the OpenCharacterTraining/OpenRLHF training stack (make oct-deps)..."
make oct-deps

# ---------------------------------------------------------------------------
# 3. .env
# ---------------------------------------------------------------------------
echo ""
if [ ! -f .env ]; then
    echo "No .env found. Copying .env.example -> .env"
    cp .env.example .env
else
    echo ".env already exists, skipping copy."
fi

echo ""
echo "========================================"
echo " Setup complete."
echo "========================================"
echo ""
echo "Edit .env and set at least HF_TOKEN and OPENROUTER_API_KEY:"
echo "  $REPO_ROOT/.env"
echo ""
echo "Then run the pipeline — see the README and scripts/*/README.md."

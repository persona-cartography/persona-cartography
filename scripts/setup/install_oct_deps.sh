#!/usr/bin/env bash
# Install the OpenCharacterTraining (OCT) dependencies that `uv sync` cannot.
#
# `character` and `openrlhf` contain git submodules pointing at git@github.com:
# SSH URLs. `uv` clones submodules recursively and fails without SSH keys, so
# these two packages can't go in pyproject.toml's normal dependency list.
# `uv pip install --no-deps` skips submodule init and works, because the Python
# packages we need live in the top level of each repo. All other OCT deps
# (deepspeed, torchdata, ninja) are standard PyPI packages and are already in
# pyproject.toml.
#
# Run this AFTER `uv sync`, only if you intend to run the OCT training pipeline
# (scripts/training/ocean_paired_dpo/ steps 02/04/05, or
# scripts_dev/oct_pipeline/). It is not needed for the dataset-build step (03),
# which has no OCT dependency.
#
# Pins are the validated versions; keep them in sync with
# scripts_dev/oct_pipeline/uv-oct-requirements.txt. The OCT stack wants a
# specific vllm (0.17.1), so we pin it here too to keep the setup
# self-contained; pyproject.toml's looser `vllm>=0.6` is for the rest of the
# project.
set -euo pipefail

CHARACTER_REF="d1da9f0"
CHARACTER_SPEC="character @ git+https://github.com/maiush/OpenCharacterTraining.git@${CHARACTER_REF}"
OPENRLHF_SPEC="openrlhf @ git+https://github.com/maiush/OpenRLHF.git"
VLLM_SPEC="vllm==0.17.1"

echo "Pinning vllm for the OCT stack: ${VLLM_SPEC}"
uv pip install "${VLLM_SPEC}"

echo "Installing OCT deps with --no-deps (skips SSH git submodules)..."
echo "  - ${CHARACTER_SPEC}"
uv pip install --no-deps "${CHARACTER_SPEC}"
echo "  - ${OPENRLHF_SPEC}"
uv pip install --no-deps "${OPENRLHF_SPEC}"

echo "Verifying imports..."
uv run python -c "import character; import openrlhf; print('OCT deps OK: character + openrlhf importable')"

#!/usr/bin/env bash
# Run all 10 activation-capping TRAIT logprob sweeps and 10 MMLU sweeps.
# c_minus is commented out — its axis already exists and evals may already be done.
# Each eval continues to the next on failure.

set -o pipefail

CONFIGS=(
    scripts.personality_evals.configs.ocean.trait.activation_capping.o_minus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.o_minus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.o_plus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.o_plus_activation_capping
    # scripts.personality_evals.configs.ocean.trait.activation_capping.c_minus_activation_capping
    # scripts.personality_evals.configs.ocean.mmlu.activation_capping.c_minus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.c_plus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.c_plus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.e_plus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.e_plus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.a_minus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.a_minus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.a_plus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.a_plus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.n_minus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.n_minus_activation_capping
    scripts.personality_evals.configs.ocean.trait.activation_capping.n_plus_activation_capping
    scripts.personality_evals.configs.ocean.mmlu.activation_capping.n_plus_activation_capping
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    uv run python -m src.evals suite --config-module "$cfg" || \
        echo "!!! FAILED: $cfg — continuing to next ==="
    echo "=== Done: $cfg ==="
done

echo ""
echo "All runs complete."
# Pod shutdown disabled — orchestrating callers issue shutdown explicitly.
# echo "Shutting down pod..."
# runpodctl stop pod "$RUNPOD_POD_ID"

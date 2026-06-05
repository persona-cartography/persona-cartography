#!/usr/bin/env bash
# Run the ocean_const_paired_dpo OCEAN trait-logprob + MMLU sweeps (migrated from
# scripts_dev .../{trait,mmlu}/vanton4_paired_dpo). Each eval continues on failure.
set -o pipefail

CONFIGS=(
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.a_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.a_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.c_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.c_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.control_s1vs2_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.e_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.e_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.n_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.n_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.o_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.o_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.a_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.a_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.c_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.c_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.control_s1vs2_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.e_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.e_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.n_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.n_plus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.o_minus_ocean_const_paired_dpo
    scripts.evals.mcq.configs.mmlu.ocean_const_paired_dpo.o_plus_ocean_const_paired_dpo
)

for cfg in "${CONFIGS[@]}"; do
    echo ""; echo "=== Running: $cfg ==="
    uv run python -m src.evals suite --config-module "$cfg" || \
        echo "!!! FAILED: $cfg — continuing ==="
done
echo "All ocean_const_paired_dpo runs complete."

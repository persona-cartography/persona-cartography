#!/usr/bin/env bash
# Run the ocean_const_paired_dpo OCEAN trait-logprob + MMLU sweeps through the
# unified eval front door (python -m src.evals adapter-sweep) — one invocation
# per (slug, eval-type), canonical defaults throughout. Each eval continues on
# failure.
set -o pipefail

SLUGS=(
    a_minus a_plus
    c_minus c_plus
    control_s1vs2
    e_minus e_plus
    n_minus n_plus
    o_minus o_plus
)

for slug in "${SLUGS[@]}"; do
    for eval_type in trait mmlu; do
        echo ""; echo "=== Running: ${slug} (${eval_type}) ==="
        uv run python -m src.evals adapter-sweep --eval-type "${eval_type}" --slug "${slug}" || \
            echo "!!! FAILED: ${slug} (${eval_type}) — continuing ==="
    done
done
echo "All ocean_const_paired_dpo runs complete."

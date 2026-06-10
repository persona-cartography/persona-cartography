#!/usr/bin/env bash
# Run the activation-capping TRAIT logprob + MMLU sweeps through the unified
# eval front door (python -m src.evals adapter-sweep --mode capping).
# c_minus is commented out — its axis already exists and evals may already be
# done; e_minus was never in the original list. Each eval continues on failure.

set -o pipefail

SLUGS=(
    o_minus o_plus
    # c_minus
    c_plus
    e_plus
    a_minus a_plus
    n_minus n_plus
)

for slug in "${SLUGS[@]}"; do
    for eval_type in trait mmlu; do
        echo ""; echo "=== Running: ${slug} (${eval_type}, capping) ==="
        uv run python -m src.evals adapter-sweep --mode capping \
            --eval-type "${eval_type}" --slug "${slug}" || \
            echo "!!! FAILED: ${slug} (${eval_type}) — continuing ==="
    done
done

echo ""
echo "All runs complete."
# Pod shutdown disabled — orchestrating callers issue shutdown explicitly.

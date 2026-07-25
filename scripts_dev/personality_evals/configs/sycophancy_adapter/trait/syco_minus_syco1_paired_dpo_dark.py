"""Dark Triad TRAIT logprob sweep (Machiavellianism/Narcissism/Psychopathy) for the sycophancy suppressor (syco_minus) adapter.

The Dark Triad splits are the TRAIT axes most relevant to sycophancy itself;
uploaded separately from the OCEAN sweep (trait_logprobs_dark) so neither
run clobbers the other via skip_completed.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.trait.syco_minus_syco1_paired_dpo_dark
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("suppressor", "trait_dark")

"""All-8 TRAIT logprob sweep (OCEAN + Dark Triad) for the psychopathy amplifier (psyc_plus) adapter.

Full personality readout, uploaded separately from the Psychopathy-only run
(trait_logprobs_all8) so neither clobbers the other via skip_completed.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.psychopathy_adapter.trait.psyc_plus_psyc1_paired_dpo_all8
"""

from scripts_dev.personality_evals.configs.psychopathy_adapter._common import build_suite

SUITE_CONFIG = build_suite("amplifier", "trait_all")

"""Psychopathy TRAIT logprob sweep for the psychopathy amplifier (psyc_plus) vpsyc1_paired_dpo adapter.

Single-split readout (Psychopathy only), per request. See _common.build_suite.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.psychopathy_adapter.trait.psyc_plus_psyc1_paired_dpo
"""

from scripts_dev.personality_evals.configs.psychopathy_adapter._common import build_suite

SUITE_CONFIG = build_suite("amplifier", "trait")

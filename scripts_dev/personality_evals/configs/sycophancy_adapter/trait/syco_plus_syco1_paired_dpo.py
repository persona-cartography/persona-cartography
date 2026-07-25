"""OCEAN TRAIT logprob sweep for the sycophancy amplifier (syco_plus) vsyco1_paired_dpo adapter.

Disentanglement check: the sycophancy adapter should NOT move the
Agreeableness axis much. See _common.build_suite for grid/upload details.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.trait.syco_plus_syco1_paired_dpo
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("amplifier", "trait")

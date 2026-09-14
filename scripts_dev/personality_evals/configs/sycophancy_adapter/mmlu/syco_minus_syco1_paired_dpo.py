"""MMLU capability sweep for the sycophancy suppressor (syco_minus) vsyco1_paired_dpo adapter.

See _common.build_suite for grid/upload details.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.mmlu.syco_minus_syco1_paired_dpo
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("suppressor", "mmlu")

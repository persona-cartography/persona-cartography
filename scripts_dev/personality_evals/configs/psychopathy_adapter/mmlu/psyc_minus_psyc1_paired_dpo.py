"""MMLU capability sweep for the psychopathy suppressor (psyc_minus) vpsyc1_paired_dpo adapter.

See _common.build_suite for grid/upload details.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.psychopathy_adapter.mmlu.psyc_minus_psyc1_paired_dpo
"""

from scripts_dev.personality_evals.configs.psychopathy_adapter._common import build_suite

SUITE_CONFIG = build_suite("suppressor", "mmlu")

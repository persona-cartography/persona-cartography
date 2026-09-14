"""CoCoNot refusal-behavior sweep (scales -1/+1) for the sycophancy suppressor (syco_minus) adapter.

See _common.build_suite for details. Full CoCoNot original set, grader gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.coconot.syco_minus_syco1_paired_dpo
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("suppressor", "coconot")

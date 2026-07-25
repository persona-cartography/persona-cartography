"""Sycophancy eval (upstream inspect_evals) for the sycophancy suppressor (syco_minus) adapter at scale 1.0.

Launch via the vLLM launcher:

    uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.sycophancy.syco_minus_syco1_paired_dpo
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("suppressor", "sycophancy")

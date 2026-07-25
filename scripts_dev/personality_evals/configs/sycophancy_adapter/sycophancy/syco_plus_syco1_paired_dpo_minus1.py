"""Sycophancy eval (upstream inspect_evals, incl. apologize_rate) for the sycophancy amplifier (syco_plus) adapter at scale -1.0.

Sign-flipped counterpart of syco_plus_syco1_paired_dpo.py, matching the OCEAN
plus1/minus1 convention so both signs of each adapter are measured.

Launch via the vLLM launcher:

    uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.sycophancy.syco_plus_syco1_paired_dpo_minus1
"""

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

SUITE_CONFIG = build_suite("amplifier", "sycophancy", scale=-1.0)

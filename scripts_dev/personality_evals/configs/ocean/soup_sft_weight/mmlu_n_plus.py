"""MMLU capability sweep over the DPO + w·SFT soup weight for N+ (vanton4_paired_dpo).

DPO fixed at 1.0, SFT weight w ∈ {0, 0.25, 0.5, 1.0}, plus a persona@1.0
reference cell. See ``_common.py`` for the soup semantics.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.soup_sft_weight.mmlu_n_plus
"""

from dotenv import load_dotenv

load_dotenv()

from scripts_dev.personality_evals.configs.ocean.soup_sft_weight._common import make_mmlu_suite

SUITE_CONFIG = make_mmlu_suite("n_plus")

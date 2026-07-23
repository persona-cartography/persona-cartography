"""TRAIT logprob sweep over the DPO + w·SFT soup weight for A+ (vanton4_paired_dpo).

DPO fixed at 1.0, SFT weight w ∈ {0, 0.25, 0.5, 1.0}, plus a persona@1.0
reference cell. See ``_common.py`` for the soup semantics and the
persona-vs-w=0.25 cross-term caveat.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.soup_sft_weight.trait_a_plus
"""

from dotenv import load_dotenv

load_dotenv()

from scripts_dev.personality_evals.configs.ocean.soup_sft_weight._common import make_trait_suite

SUITE_CONFIG = make_trait_suite("a_plus")

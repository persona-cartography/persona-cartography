"""DPO↔SFT mix sweep for A+ and N+, plus matched A+N+ soups (llama-3.1-8b-it).

16 models along the convex DPO→SFT line (see ``dpo_sft_mix_common``): base,
5 A+ mixes, 5 N+ mixes, 5 matched A+ ⊕ N+ soups. Scored on all five OCEAN
TRAIT splits + MMLU.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.dpo_sft_mix_a_n
"""

from dotenv import load_dotenv

load_dotenv()

from scripts_dev.personality_evals.configs.ocean.dpo_sft_mix_common import build_mix_suite

SUITE_CONFIG = build_mix_suite(
    trait_a_slug="a_plus",
    trait_b_slug="n_plus",
    label_a="a",
    label_b="n",
    run_name="dpo_sft_mix_a_n",
    upload_subdir="dpo_sft_mix_a_n",
)

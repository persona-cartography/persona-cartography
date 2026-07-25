"""DPO↔SFT mix sweep for A+ and N− (suppressor), plus matched A+N− soups.

Same design as ``dpo_sft_mix_a_n`` but with the neuroticism **suppressor**.
A+ and N− both push neuroticism *down*, so on the N axis they cooperate rather
than fight (unlike A+ × N+) — a test of composability when two adapters agree
on a trait direction. 16 models scored on all five OCEAN TRAIT splits + MMLU.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.dpo_sft_mix_a_nminus
"""

from dotenv import load_dotenv

load_dotenv()

from scripts_dev.personality_evals.configs.ocean.dpo_sft_mix_common import build_mix_suite

SUITE_CONFIG = build_mix_suite(
    trait_a_slug="a_plus",
    trait_b_slug="n_minus",
    label_a="a",
    label_b="nminus",
    run_name="dpo_sft_mix_a_nminus",
    upload_subdir="dpo_sft_mix_a_nminus",
)

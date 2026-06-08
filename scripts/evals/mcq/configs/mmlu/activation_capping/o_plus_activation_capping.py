"""O+ (openness amplifier) MMLU sweep — activation capping.

Thin config; all defaults live in scripts.evals.mcq.configs._builders.
"""

from dotenv import load_dotenv

from scripts.evals.mcq.configs._builders import build_cap_mcq_suite

load_dotenv()

SUITE_CONFIG = build_cap_mcq_suite(slug="o_plus", eval_type="mmlu")

"""C+ (conscientiousness amplifier) TRAIT-logprob sweep — paired-DPO persona adapter.

Thin config; all defaults live in scripts.evals.mcq.configs._builders.
"""

from dotenv import load_dotenv

from scripts.evals.mcq.configs._builders import build_direct_mcq_suite

load_dotenv()

SUITE_CONFIG = build_direct_mcq_suite(slug="c_plus", eval_type="trait")

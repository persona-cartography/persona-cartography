"""A- (agreeableness suppressor) TRAIT-logprob sweep — activation capping.

Thin config; all defaults live in src.evals.mcq_builders.
"""

from dotenv import load_dotenv

from src.evals.mcq_builders import build_cap_mcq_suite

load_dotenv()

SUITE_CONFIG = build_cap_mcq_suite(slug="a_minus", eval_type="trait")

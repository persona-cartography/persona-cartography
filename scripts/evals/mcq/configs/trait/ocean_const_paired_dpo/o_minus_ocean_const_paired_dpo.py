"""O- (openness suppressor) TRAIT-logprob sweep — paired-DPO persona adapter.

Thin config; all defaults live in src.evals.mcq_builders.
"""

from dotenv import load_dotenv

from src.evals.mcq_builders import build_direct_mcq_suite

load_dotenv()

SUITE_CONFIG = build_direct_mcq_suite(slug="o_minus", eval_type="trait")

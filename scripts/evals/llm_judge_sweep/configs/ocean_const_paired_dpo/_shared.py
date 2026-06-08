"""Shared constants for the ocean_const_paired_dpo LLM judge scale sweep configs.

Single Qwen3-235B judge setup (the team's current default):
  - NUM_ROLLOUTS_PER_PROMPT = 1
  - ASSISTANT_MAX_NEW_TOKENS = 2048
  - JUDGE_RATERS = single Qwen3-235B rater via OpenRouter

Each per-direction module does
``from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *``
and then overrides DATASET_PATH, EVAL_NAME, TRAIT, ADAPTER, ADAPTERS,
SCALES_PER_ADAPTER, JUDGE_METRIC_TRAITS, TRAIT_COLOR, and PLOT_TITLE.
"""

from __future__ import annotations

from src.evals.judges.config import JudgeLLMConfig
from src.evals.judges.llm_judge_agreement import JudgeRaterConfig

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
MAX_SAMPLES = 240
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 2048
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 1.0
ASSISTANT_TOP_P = 1.0
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 1
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
COHERENCE_METRIC = "better_coherence_judge"
COHERENCE_COLOR = "#757575"
JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="qwen3_235b",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="qwen/qwen3-235b-a22b-2507",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=32,
        ),
    ),
]

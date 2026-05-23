"""Shared constants for vanton4_paired_dpo LLM judge sweep on talkie-1930-13b-it.

Mirrors scripts_dev/evals/llm_judge_sweep/configs/vanton4_paired_dpo/_shared.py
(Qwen3-235B single-judge setup) with BASE_MODEL/BASE_MODEL_SLUG overridden
for the talkie-1930-13b-it student.

Each per-direction module does
``from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930._shared import *``
and then overrides DATASET_PATH, EVAL_NAME, TRAIT, ADAPTER, ADAPTERS,
SCALES_PER_ADAPTER, JUDGE_METRIC_TRAITS, TRAIT_COLOR, and PLOT_TITLE.
"""

from __future__ import annotations

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Use the locally-materialized HF wrapper at $OCT_MODEL_PATH (see
# src_dev/models/talkie/materialize.py). The talkie-lm hub repo ships only
# the raw rl-refined.pt and isn't transformers-loadable; "local://..."
# tells the eval suite to skip HF-hub resolution.
BASE_MODEL = "local:///root/.cache/models/talkie-1930-13b-it"
BASE_MODEL_SLUG = "talkie-1930-13b-it"

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

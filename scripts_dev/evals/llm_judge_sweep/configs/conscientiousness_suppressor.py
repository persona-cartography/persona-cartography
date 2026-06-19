"""Config constants for conscientiousness suppressor LoRA scale sweep.

This module is imported by the runner via ``--config`` CLI flag. All
experiment-specific values live here as module-level constants.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner \\
        --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor
"""

from __future__ import annotations

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.llm_judge_agreement import JudgeRaterConfig
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EVAL_NAME = "conscientiousness-suppressor"

# ---------------------------------------------------------------------------
# Model & adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
ADAPTER_REF = (
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/"
    "suppressor/v3/lora/conscientiousness_low-persona"
)
BAKED_ADAPTERS_SUBDIR = "conscientiousness_low_suppressor_v3_llama_3_1_8b_instruct"

# ---------------------------------------------------------------------------
# Trait
# ---------------------------------------------------------------------------
TRAIT = OceanTrait.conscientiousness
ARTIFACT_TRAIT = "conscientious"
DIRECTION = "suppressor"
VERSION = "v3-llama-3.1-8b-instruct"
TRAINING_RUN = "suppressor-v3-llama-3.1-8b-instruct"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 256
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 0.7
ASSISTANT_TOP_P = 0.95
USER_MODEL = "z-ai/glm-4.5-air:free"
USER_PROVIDER = "openrouter"

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 5
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
COHERENCE_METRIC = "better_coherence_judge"
JUDGE_RATERS = [
    JudgeRaterConfig(
        rater_id="qwen3_235b",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="qwen/qwen3-235b-a22b-2507",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=15,
        ),
    ),
    JudgeRaterConfig(
        # rater_id "gemma4_27b" == Gemma 4 26B-A4B (google/gemma-4-26b-a4b-it);
        # "_27b" kept to match existing HF judge_runs/ data.
        rater_id="gemma4_27b",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=15,
        ),
    ),
    JudgeRaterConfig(
        rater_id="llama33_70b",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=15,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Plot colors
# ---------------------------------------------------------------------------
TRAIT_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
COHERENCE_COLOR = "#757575"
PLOT_TITLE = "Conscientiousness suppressor LoRA scale sweep"

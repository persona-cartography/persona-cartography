"""Fine-grained scale sweep for the talkie C- SFT-merged -persona adapter.

Same setup as ``c_minus_periodteacher_periodeval`` (period prompts + period
judges + eos-fixed model + -persona adapter) but with a finer scale grid
(step 0.5) and the full 240-prompt sample, for a higher-resolution dose-response
curve. Strong negative scales may still drop out (the adapter there pushes talkie
to empty responses).

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_persona_fine
"""

from __future__ import annotations

import os

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_periodeval import *  # noqa: F401,F403

# Finer dose-response grid (step 0.5).
SCALE_POINTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}

# Full held-out sample (override via EVAL_MAX_SAMPLES).
MAX_SAMPLES = int(os.environ.get("EVAL_MAX_SAMPLES", "240"))

# Strong negative scales push talkie to occasionally emit an empty turn (the C-
# suppressor applied in reverse). Tolerate up to 25% empty/dropped conversations
# per cell and judge the responses that *do* come back, so the negative side of
# the dose-response curve is measured instead of nan'd. The per-cell dropout is
# logged by the sweep so it can be reported alongside the scores. (-2.0 dropped
# ~15% in the earlier strict run.)
MAX_FAILED_FRACTION = float(os.environ.get("EVAL_MAX_FAILED_FRACTION", "0.25"))

EVAL_NAME = "conscientiousness-suppressor-talkie1930-periodteacher-persona-fine"
PLOT_TITLE = (
    "Conscientiousness suppressor (talkie-1930-13b-it, periodteacher -persona, "
    "period eval + period judge) fine LoRA scale sweep (step 0.5)"
)

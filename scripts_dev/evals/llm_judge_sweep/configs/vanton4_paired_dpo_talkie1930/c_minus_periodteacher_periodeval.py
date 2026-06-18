"""Period-ADJUSTED LLM-judge sweep for the talkie C- period-teacher adapter.

Same adapter as ``c_minus_periodteacher`` but the eval itself is period-adjusted
so the 1928-era model is not pushed out of distribution at eval time:
  * Rollout prompts = 1928-register conscientiousness questions
    (data/ocean_open_ended/conscientiousness_period.jsonl), instead of modern ones.
  * Judges = period-adjusted variants that do not treat archaic register as a
    trait or coherence signal (conscientiousness_v2_period_adjusted /
    better_coherence_judge_period_adjusted).

This is the primary, fair measurement of the period-teacher C- adapter. The
sibling ``c_minus_periodteacher`` (modern prompts + standard judges) is kept for
reference / to show the eval-distribution effect.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_periodeval
"""

from __future__ import annotations

import os

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher import *  # noqa: F401,F403

# 1928-register eval prompts (held out from training; see make_period_eval_dataset.py).
DATASET_PATH = "data/ocean_open_ended/conscientiousness_period.jsonl"

# Safety cap; the eos fix (stop on <|end|>) means talkie ends at ~150 toks.
ASSISTANT_MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "512"))

# Period-adjusted judges (1920s register is not a trait/coherence signal).
JUDGE_METRIC_TRAITS = ["conscientiousness_v2_period_adjusted"]
COHERENCE_METRIC = "better_coherence_judge_period_adjusted"

EVAL_NAME = "conscientiousness-suppressor-talkie1930-vanton4-paired-dpo-periodteacher-periodeval"
PLOT_TITLE = (
    "Conscientiousness suppressor (talkie-1930-13b-it, vanton4_paired_dpo_periodteacher, "
    "period eval + period judge) LoRA scale sweep"
)

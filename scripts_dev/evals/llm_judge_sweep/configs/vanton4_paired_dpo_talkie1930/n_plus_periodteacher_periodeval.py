"""Period-ADJUSTED LLM-judge sweep for the talkie N+ period-teacher adapter.

Same adapter as ``n_plus_periodteacher`` but the eval itself is period-adjusted
so the 1928-era model is judged fairly (mirrors c_minus_periodteacher_periodeval):
  * Rollout prompts = 1928-register neuroticism questions
    (data/ocean_open_ended/neuroticism_period.jsonl).
  * Judges = period-adjusted variants that do not treat archaic register as a
    trait/coherence signal (neuroticism_v2_period_adjusted /
    better_coherence_judge_period_adjusted).
  * MAX_FAILED_FRACTION tolerates the empty/degenerate turns talkie emits at
    strong scales (the export_rollouts fix judges the survivors).
  * Finer scale grid for a clearer dose-response.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_plus_periodteacher_periodeval \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.n_plus_periodteacher import *  # noqa: F401,F403

# 1928-register eval prompts (held out from training).
DATASET_PATH = "data/ocean_open_ended/neuroticism_period.jsonl"

# Period-adjusted judges (1920s register is not a trait/coherence signal).
JUDGE_METRIC_TRAITS = ["neuroticism_v2_period_adjusted"]
COHERENCE_METRIC = "better_coherence_judge_period_adjusted"

# Tolerate the empty/degenerate turns talkie emits at strong |scale| (judge the rest).
MAX_FAILED_FRACTION = 0.25

# Finer dose-response grid.
SCALE_POINTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SCALES_PER_ADAPTER = {ADAPTER.slug: SCALE_POINTS}  # noqa: F405

EVAL_NAME = "neuroticism-amplifier-talkie1930-vanton4-paired-dpo-periodteacher-periodeval"
PLOT_TITLE = (
    "Neuroticism amplifier (talkie-1930-13b-it, vanton4_paired_dpo_periodteacher, "
    "period eval + period judge) LoRA scale sweep"
)

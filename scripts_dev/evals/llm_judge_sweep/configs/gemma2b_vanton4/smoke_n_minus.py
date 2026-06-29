"""Smoke test: gemma-2b-it neuroticism suppressor judge sweep.

Tiny fast run to validate the judge pipeline end-to-end on gemma-2b (vLLM +
LoRA scale + Qwen3 judge) before kicking off the real sweep. Mirrors
``configs/gemma_consc_sup/smoke_g4b.py``.

Overrides vs n_minus.py:
  - MAX_SAMPLES=4, JUDGE_REPEATS=1, SCALE_POINTS=[1.0]
  - Different fingerprint from any real sweep (safe: data isolated).

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.gemma2b_vanton4.smoke_n_minus \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.gemma2b_vanton4.n_minus import *  # noqa: F401,F403

MAX_SAMPLES = 4
JUDGE_REPEATS = 1
SCALES_PER_ADAPTER = {ADAPTER.slug: [1.0]}

EVAL_NAME = "gemma-2b-neuroticism-suppressor-vanton4-smoke"
PLOT_TITLE = "SMOKE TEST: gemma-2b neuroticism suppressor (Qwen3-235B judge)"

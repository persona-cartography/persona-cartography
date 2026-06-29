"""Shared constants for the gemma-2b-it vanton4 LLM judge scale sweep configs.

Mirrors ``scripts_dev/evals/llm_judge_sweep/configs/vanton4_qwen3/_shared.py``
(same rollout/judge settings as the llama-3.1-8b OCEAN judge sweeps:
MAX_SAMPLES=240, NUM_ROLLOUTS_PER_PROMPT=1, temperature=1.0,
max_new_tokens=2048, JUDGE_REPEATS=2, Qwen3-235B rater) and only overrides
BASE_MODEL / BASE_MODEL_SLUG for gemma-2b-it (Gemma 1). This is the gemma-2b
analog of the gemma_consc_sup family, kept per-trait-direction so each adapter
is one tiny module — exactly like the llama-8b vanton4_qwen3 family.

Each per-direction module does
``from scripts_dev.evals.llm_judge_sweep.configs.gemma2b_vanton4._shared import *``
and then overrides DATASET_PATH, EVAL_NAME, TRAIT, ADAPTER, ADAPTERS,
SCALES_PER_ADAPTER, JUDGE_METRIC_TRAITS, TRAIT_COLOR, and PLOT_TITLE.

Note: gemma-2b-it OCEAN adapters live on ``persona-cartography/monorepo``
(the post-rename repo the OCT pipeline now uploads to), so the ADAPTER refs in
the per-direction modules use that repo explicitly rather than the
runner_cells default.
"""

from __future__ import annotations

# Inherit all rollout + judge settings from the llama-8b Qwen3 family.
from scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3._shared import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Model (gemma-2b-it / Gemma 1) — only override vs the inherited llama config.
# ---------------------------------------------------------------------------
BASE_MODEL = "google/gemma-2b-it"
BASE_MODEL_SLUG = "gemma-2b-it"

# gemma-2b OCEAN adapters + their MCQ/MMLU evals live on the post-rename
# persona-cartography monorepo (the OCT pipeline uploads there). Point the judge
# sweep at the same repo so all of this model's data stays co-located. Honored
# by runner_cells.main() via a getattr override of its module-level default.
HF_REPO_ID = "persona-cartography/monorepo"

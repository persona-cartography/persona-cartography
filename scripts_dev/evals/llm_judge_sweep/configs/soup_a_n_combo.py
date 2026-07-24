"""Cross-trait soup SCALE grid: A+ x N+ over souping scales {0, 0.25, 0.5, 1.0}.

4x4 Cartesian grid on the two persona adapters' soup coefficients, judged on
BOTH agreeableness_v2 and neuroticism_v2. One grid gives all three claims:

  - A works across souping scales  -> the c_N = 0 row (A+ alone at 0/0.25/0.5/1.0)
  - N works across souping scales  -> the c_A = 0 column (N+ alone at 0/0.25/0.5/1.0)
  - A+N composes across scales     -> the interior cells (both non-zero); the
    agreeableness heatmap rises along c_A independent of c_N, the neuroticism
    heatmap rises along c_N independent of c_A.

Lighter rollout budget than the vanton4_paired_dpo single-adapter family
(MAX_SAMPLES=100, 512 new tokens) so the 16-cell grid stays tractable; this
gives the grid its own rollout fingerprint (no reuse of the 240-sample cells).

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \\
        --config scripts_dev.evals.llm_judge_sweep.configs.soup_a_n_combo \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

from scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo._shared import *  # noqa: F401,F403
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait

EVAL_NAME = "soup-a-n-scale-grid"
# Dedicated canonical namespace so the grid's combo cells live apart from the
# vanton4_paired_dpo scale-sweep data.
EVAL_NAME_CANONICAL = "llm_judge_soup_a_n_scale_grid"

# Souping-scale grid.
SOUP_SCALES = [0.0, 0.25, 0.5, 1.0]
MAX_SAMPLES = 100
ASSISTANT_MAX_NEW_TOKENS = 512

A_PLUS = AdapterSpec.from_ref(OCEAN_REGISTRY["a_plus"].adapter_ref)
N_PLUS = AdapterSpec.from_ref(OCEAN_REGISTRY["n_plus"].adapter_ref)
ADAPTERS = [A_PLUS, N_PLUS]
SCALES_PER_ADAPTER = {
    A_PLUS.slug: SOUP_SCALES,
    N_PLUS.slug: SOUP_SCALES,
}

# Judge every cell on both trait axes so each heatmap shows its own axis rising.
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [
    OceanTrait.agreeableness.v2_metric_name,
    OceanTrait.neuroticism.v2_metric_name,
]
PLOT_TITLE = "A+ x N+ souping-scale grid (Qwen3-235B judge, agreeableness & neuroticism)"

"""Cross-trait soup demo (LLM judge): A+ x N+ 2x2 grid on {0, 1}.

Judges the same rollouts on BOTH agreeableness_v2 and neuroticism_v2, over the
four cells base(0,0) / A+(1,0) / N+(0,1) / A+ + N+ soup(1,1). Demonstrates that
souping two persona adapters expresses both traits behaviourally, not just on
the TRAIT questionnaire.

Reuses the vanton4_paired_dpo rollout/judge settings so the single-adapter
slices (A+ @1, N+ @1) land at the same canonical cell paths as the existing
per-trait judge sweeps.

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

EVAL_NAME = "soup-a-n-combo"

A_PLUS = AdapterSpec.from_ref(OCEAN_REGISTRY["a_plus"].adapter_ref)
N_PLUS = AdapterSpec.from_ref(OCEAN_REGISTRY["n_plus"].adapter_ref)
ADAPTERS = [A_PLUS, N_PLUS]
SCALES_PER_ADAPTER = {
    A_PLUS.slug: [0.0, 1.0],
    N_PLUS.slug: [0.0, 1.0],
}

# Judge every cell on both trait axes so the soup cell can show both rising.
TRAIT = OceanTrait.agreeableness
JUDGE_METRIC_TRAITS = [
    OceanTrait.agreeableness.v2_metric_name,
    OceanTrait.neuroticism.v2_metric_name,
]
PLOT_TITLE = "A+ x N+ soup (Qwen3-235B judge, agreeableness & neuroticism)"

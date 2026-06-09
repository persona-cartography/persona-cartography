"""Config for the Fig. 1 paired-DPO combo cells.

This module is parameterised by environment variables so the same runner config
can populate the small set of paper-diagnostic cells without 15 near-identical
modules.

Environment:
    FIG1_COMBO: one of ``c_minus_e_minus``, ``c_minus_e_plus``, ``o_plus_n_plus``.
    FIG1_TRAIT: one of ``openness``, ``conscientiousness``, ``extraversion``,
        ``agreeableness``, ``neuroticism``.

Example:
    FIG1_COMBO=c_minus_e_plus FIG1_TRAIT=conscientiousness \\
        uv run python -m src.evals.llm_judge_sweep.runner_cells \\
        --config scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.paper_fig1_combo_cells \\
        --allow-custom-fingerprint
"""

from __future__ import annotations

import os

from scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo._shared import *  # noqa: F401,F403
from src.evals.llm_judge_sweep.cell_identity import AdapterSpec
from src.visualisations.palette import BIG_FIVE_COLORS
from src.evals.judges.metrics.ocean_v2 import OceanTrait


TRAIT_BY_NAME = {
    "openness": OceanTrait.openness,
    "conscientiousness": OceanTrait.conscientiousness,
    "extraversion": OceanTrait.extraversion,
    "agreeableness": OceanTrait.agreeableness,
    "neuroticism": OceanTrait.neuroticism,
}

ADAPTER_BY_KEY = {
    "c_minus": AdapterSpec.from_ref(
        "persona-shattering-lasr/monorepo::"
        "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/ocean_const_paired_dpo"
        "/lora/conscientiousness_suppressing_full-persona"
    ),
    "e_minus": AdapterSpec.from_ref(
        "persona-shattering-lasr/monorepo::"
        "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/ocean_const_paired_dpo"
        "/lora/extraversion_suppressing_full-persona"
    ),
    "e_plus": AdapterSpec.from_ref(
        "persona-shattering-lasr/monorepo::"
        "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/ocean_const_paired_dpo"
        "/lora/extraversion_amplifying_full-persona"
    ),
    "o_plus": AdapterSpec.from_ref(
        "persona-shattering-lasr/monorepo::"
        "fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/ocean_const_paired_dpo"
        "/lora/openness_amplifying_full-persona"
    ),
    "n_plus": AdapterSpec.from_ref(
        "persona-shattering-lasr/monorepo::"
        "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo"
        "/lora/neuroticism_amplifying_full-persona"
    ),
}

COMBO_ADAPTER_KEYS = {
    "c_minus_e_minus": ("c_minus", "e_minus"),
    "c_minus_e_plus": ("c_minus", "e_plus"),
    "o_plus_n_plus": ("o_plus", "n_plus"),
}

COMBO_LABEL = {
    "c_minus_e_minus": "C-minus x E-minus",
    "c_minus_e_plus": "C-minus x E-plus",
    "o_plus_n_plus": "O-plus x N-plus",
}

combo_key = os.environ.get("FIG1_COMBO", "")
trait_name = os.environ.get("FIG1_TRAIT", "")
if combo_key not in COMBO_ADAPTER_KEYS:
    raise ValueError(
        f"Set FIG1_COMBO to one of {sorted(COMBO_ADAPTER_KEYS)}; got {combo_key!r}."
    )
if trait_name not in TRAIT_BY_NAME:
    raise ValueError(
        f"Set FIG1_TRAIT to one of {sorted(TRAIT_BY_NAME)}; got {trait_name!r}."
    )

TRAIT = TRAIT_BY_NAME[trait_name]
DATASET_PATH = f"data/ocean_open_ended/{trait_name}.jsonl"

ADAPTERS = [ADAPTER_BY_KEY[key] for key in COMBO_ADAPTER_KEYS[combo_key]]
SCALES_PER_ADAPTER = {adapter.slug: [1.0] for adapter in ADAPTERS}

JUDGE_METRIC_TRAITS = [TRAIT.v2_metric_name]
# The paper bar plots only need the trait metric; skip coherence to keep the
# fill-in run small.
COHERENCE_METRIC = None

EVAL_NAME = f"paper-fig1-{combo_key}-on-{trait_name}-1x1"
TRAIT_COLOR = BIG_FIVE_COLORS[TRAIT.value.capitalize()]
PLOT_TITLE = f"{COMBO_LABEL[combo_key]} @ (+1,+1) on {trait_name} prompts"

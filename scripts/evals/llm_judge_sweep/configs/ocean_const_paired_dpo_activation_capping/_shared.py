"""Shared constants for the ocean_const_paired_dpo activation-capping LLM-judge sweep.

Mirrors the LoRA-scale family at
``scripts/evals/llm_judge_sweep/configs/ocean_const_paired_dpo/_shared.py`` but
runs the rollout stage with :class:`ActivationCapProvider` (HF transformers +
per-layer forward hooks) instead of :class:`VLLMLoRaComboProvider`. The cell
on-disk layout, judge wiring, and aggregation/plot output are identical to the
LoRA case — only the eval-name prefix and the rollout method differ::

    LoRA scale sweep:        .../evals/llm_judge_lora_scale_sweep/<fingerprint>/
    Activation cap sweep:    .../evals/llm_judge_activation_capping_sweep/<fingerprint>/

The five SCALE_POINTS are interpreted by :class:`ActivationCapProvider` as
fractions along the persona's pre-computed axis (``-2`` extrapolates trait
suppression past baseline, ``+2`` extrapolates trait amplification past the
LoRA endpoint). ``0.0`` is the baseline cell — runs the unmodified base model.

Each per-direction module does
``from .._shared import *`` and overrides DATASET_PATH, EVAL_NAME, TRAIT,
ADAPTER, ADAPTERS, SCALES_PER_ADAPTER, JUDGE_METRIC_TRAITS, TRAIT_COLOR,
PLOT_TITLE, and supplies ACTIVATION_CAP_CONFIG via :func:`build_cap_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src.persona_metrics.config import JudgeLLMConfig
from src.persona_metrics.llm_judge_agreement import JudgeRaterConfig

# build_cap_config() downloads axis .pt files from the monorepo at config
# import time to read recommended_capping_layers, which needs HF_TOKEN.
load_dotenv()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# ---------------------------------------------------------------------------
# Sweep — fractions along the persona axis (matches the LoRA-scale grid)
# ---------------------------------------------------------------------------
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SEED = 42

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
MAX_SAMPLES = 240
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"  # overridden per trait
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

# ---------------------------------------------------------------------------
# Routing — distinct HF prefix from the LoRA-scale sweep
# ---------------------------------------------------------------------------
EVAL_NAME_CANONICAL = "llm_judge_activation_capping_sweep"
X_AXIS_LABEL = "Activation cap fraction"


# ---------------------------------------------------------------------------
# Activation-capping config builder
# ---------------------------------------------------------------------------
#
# OCEAN± ocean_const_paired_dpo axis files live next to each LoRA at
# ``<lora_parent>/activation_capping/<persona_slug>_axis.pt`` (the layout
# ``compute_axis.py`` writes them to). The OCEAN_REGISTRY currently has
# ``axis_slug=None`` for these entries because the registry's ``axis_slug``
# helper assumes the legacy top-level ``activation_capping/<slug>/...``
# layout. We bypass it and derive paths from the LoRA path directly, the way
# the MCQ ``activation_capping`` configs already do.

_FALLBACK_CAPPING_LAYERS = list(range(17, 32))


def _axis_uris(slug: str) -> tuple[str, str]:
    """Build hf:// URIs for the axis + per-layer-range files for a persona."""
    trait_def = OCEAN_REGISTRY[slug]
    p = Path(trait_def.adapter_path_in_repo)
    parent = p.parent.parent if p.parent.name == "lora" else p
    axis = f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/{slug}_axis.pt"
    per_layer = f"hf://{HF_REPO}/{parent.as_posix()}/activation_capping/{slug}_per_layer_range.pt"
    return axis, per_layer


def _read_capping_layers(axis_uri: str) -> list[int]:
    """Load recommended capping layers from the axis ``.pt`` metadata.

    Falls back to layers 17–31 (upper half of Llama-3.1-8B's 32 layers) if
    the metadata key is missing — same fallback as
    ``scripts/rollout_experiments/ocean/generate_rollouts.py``.
    """
    import torch

    from src.rollout_generation.model_providers import _resolve_hf_path

    local = _resolve_hf_path(axis_uri)
    data = torch.load(local, map_location="cpu", weights_only=False)
    layers = data.get("metadata", {}).get("recommended_capping_layers")
    if layers:
        return list(layers)
    return list(_FALLBACK_CAPPING_LAYERS)


def build_cap_config(slug: str) -> dict[str, Any]:
    """Return the ``ACTIVATION_CAP_CONFIG`` dict consumed by ``runner_cells.py``."""
    axis_uri, per_layer_uri = _axis_uris(slug)
    return {
        "axis_path": axis_uri,
        "per_layer_range_path": per_layer_uri,
        "capping_layers": _read_capping_layers(axis_uri),
    }

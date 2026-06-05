"""A+ (agreeableness amplifier) MMLU capability sweep using activation capping (ocean_const_paired_dpo axis).

Sweeps over capping fractions along the pre-computed a_plus activation direction.
Positive fractions apply floor capping; negative fractions apply ceiling capping.
The base model (fraction=0) is always included.

Axis + per-layer range files are downloaded from the monorepo, sibling to the
ocean_const_paired_dpo LoRA at ``fine_tuning/.../a_plus/ocean_const_paired_dpo/activation_capping/``. The local
cache is versioned (``a_plus_ocean_const_paired_dpo``) to avoid clobbering older artifacts.

Parameters (batch size, limit) match the direct-adapter ocean_const_paired_dpo MMLU configs in
``scripts.personality_evals.configs.ocean.mmlu.ocean_const_paired_dpo``.

Usage
-----
    uv run python -m src.evals suite \\
        --config-module scripts.personality_evals.configs.ocean.mmlu.activation_capping.a_plus_activation_capping
"""

from pathlib import Path

from dotenv import load_dotenv

from src.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src.evals import (
    ActivationCapSweep,
    InspectBenchmarkSpec,
    SuiteConfig,
)
from src.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Model and axis artifacts
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SLUG = "a_plus"
_LORA_PATH = Path(getattr(LoraHFCatalogue(), SLUG))
# OCEAN catalogue entries end in ".../{version}/lora/<adapter-name>";
# the legacy gemma entry is just ".../{version}". Handle both.
if _LORA_PATH.parent.name == "lora":
    _LORA_PARENT = _LORA_PATH.parent.parent
    LORA_VERSION = _LORA_PATH.parent.parent.name
else:
    _LORA_PARENT = _LORA_PATH
    LORA_VERSION = _LORA_PATH.name

_AXIS_DIR = (
    Path("scratch/llama_8b_instruct/activation_capping") / f"{SLUG}_{LORA_VERSION}"
)
_AXIS_PATH = _AXIS_DIR / (SLUG + "_axis.pt")
_PER_LAYER_RANGE_PATH = _AXIS_DIR / (SLUG + "_per_layer_range.pt")

_MONOREPO_ID = HF_REPO
_MONOREPO_AXIS_PATH = str(_LORA_PARENT / "activation_capping")

if not (_AXIS_PATH.exists() and _PER_LAYER_RANGE_PATH.exists()):
    _AXIS_DIR.mkdir(parents=True, exist_ok=True)
    download_from_dataset_repo(
        repo_id=_MONOREPO_ID,
        path_in_repo=_MONOREPO_AXIS_PATH,
        local_dir=_AXIS_DIR,
        allow_patterns=[SLUG + "_axis.pt", SLUG + "_per_layer_range.pt"],
    )
    _nested = _AXIS_DIR / _MONOREPO_AXIS_PATH
    if _nested.exists():
        for _f in _nested.iterdir():
            _target = _AXIS_DIR / _f.name
            if not _target.exists():
                _f.replace(_target)
# ---------------------------------------------------------------------------


def _build_fraction_points() -> list[float]:
    """Step 0.5 in [-2, -1.5] and [+1.5, +2], step 0.25 in [-1, +1]."""
    coarse_neg = [
        round(-2.0 + i * 0.5, 10) for i in range(round((-1.5 - -2.0) / 0.5) + 1)
    ]
    fine = [round(-1.0 + i * 0.25, 10) for i in range(round((1.0 - -1.0) / 0.25) + 1)]
    coarse_pos = [round(1.5 + i * 0.5, 10) for i in range(round((2.0 - 1.5) / 0.5) + 1)]
    return sorted({f for f in coarse_neg + fine + coarse_pos if f != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    activation_cap=ActivationCapSweep(
        fractions=_build_fraction_points(),
        axis_path=str(_AXIS_PATH.resolve()),
        per_layer_range_path=str(_PER_LAYER_RANGE_PATH.resolve()),
        ceiling_from_hi=True,
        # capping_layers=None → read from axis metadata (recommended_capping_layers)
    ),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=300,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name=f"{SLUG}_activation_capping_ocean_const_paired_dpo_mmlu",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "random_baseline": 0.25,
        "title_suffix": "A+ Activation Capping ocean_const_paired_dpo MMLU",
        "interval": "ci95_from_wilson",
        "x_label": "Activation Vector Limit",
        "x_lim": (-2.5, 2.5),
    },
    upload_repo_id=_MONOREPO_ID,
    upload_path_in_repo=str(_LORA_PARENT / "evals/mcq/activation_capping/mmlu"),
    metadata={
        "persona": SLUG,
        "method": "activation_capping",
        "lora_version": LORA_VERSION,
        "axis_path": str(_AXIS_PATH),
    },
)

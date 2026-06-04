"""Dynamic config module for batch trait logprob evals (suite mode).

Reads ``ADAPTER_KEY`` or ``COMBO_KEY`` from the environment and delegates to
the factory to produce a ``SUITE_CONFIG`` that the ``src.evals suite`` command
loads via ``--config-module``. This is the suite-mode sibling of
``run_adapter_cells.py`` (which targets the cell runner instead).

Usage:
    ADAPTER_KEY=a_plus_vanton1 uv run python -m src.evals suite \
        --config-module scripts.personality_evals.configs.ocean.trait.run_adapter

    COMBO_KEY=a_plus_minus_vanton1 uv run python -m src.evals suite \
        --config-module scripts.personality_evals.configs.ocean.trait.run_adapter

    ADAPTER_KEY=__baseline__ uv run python -m src.evals suite \
        --config-module scripts.personality_evals.configs.ocean.trait.run_adapter
"""

import os

from scripts.personality_evals.configs.ocean.trait.factory import (
    ADAPTER_REGISTRY,
    COMBO_REGISTRY,
    make_baseline_config,
    make_combo_config,
    make_sweep_config,
)

_adapter_key = os.environ.get("ADAPTER_KEY")
_combo_key = os.environ.get("COMBO_KEY")

if _adapter_key == "__baseline__":
    SUITE_CONFIG = make_baseline_config()
elif _adapter_key:
    if _adapter_key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown ADAPTER_KEY={_adapter_key!r}. "
            f"Available: {sorted(ADAPTER_REGISTRY)}"
        )
    SUITE_CONFIG = make_sweep_config(_adapter_key)
elif _combo_key:
    if _combo_key not in COMBO_REGISTRY:
        raise ValueError(
            f"Unknown COMBO_KEY={_combo_key!r}. "
            f"Available: {sorted(COMBO_REGISTRY)}"
        )
    SUITE_CONFIG = make_combo_config(_combo_key)
else:
    raise ValueError(
        "Set ADAPTER_KEY or COMBO_KEY environment variable. "
        f"Available adapters: {sorted(ADAPTER_REGISTRY)}. "
        f"Available combos: {sorted(COMBO_REGISTRY)}."
    )

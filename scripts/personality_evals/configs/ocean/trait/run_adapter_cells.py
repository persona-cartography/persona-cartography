"""Dynamic config module for cell-oriented batch trait logprob sweeps.

This module is a thin *router*: it selects one cell-runner config from the
:mod:`~scripts.personality_evals.configs.ocean.trait.factory` and unpacks it
into this module's globals, so the generic cell runner
(:mod:`scripts.evals.trait_sweep.runner_cells`) can read the config off this
module the same way it reads any hand-written cell config (via ``getattr`` on
the imported config module).

Selection / pipeline overview:

1. **Resolve config** — read ``ADAPTER_KEY`` or ``COMBO_KEY`` from the
   environment and dispatch to the matching factory builder:

   - ``ADAPTER_KEY=__baseline__`` -> :func:`make_cell_baseline_config`
     (base model, no adapters).
   - ``ADAPTER_KEY=<key>`` -> :func:`make_cell_sweep_config` (single-adapter
     scale sweep; key must be in ``ADAPTER_REGISTRY``).
   - ``COMBO_KEY=<key>`` -> :func:`make_cell_combo_config` (Cartesian grid of
     adapter scales; key must be in ``COMBO_CELL_REGISTRY``).

   Each builder returns a dict of module-level constants (``ADAPTERS``,
   ``SCALES_PER_ADAPTER``, fingerprint fields, …).

2. **Optional trait subsetting** — if ``TRAIT_SPLITS`` is set, override the
   config's ``TRAIT_SPLITS`` list. Because ``trait_splits`` is *not* part of
   the rollout fingerprint (per-trait cell layout), subsetting is free and
   cache-compatible with runs of other subsets.

3. **Publish** — ``globals().update(_cfg)`` exposes the chosen config as
   module attributes for the runner to consume.

The heavy lifting (enumerate cells -> hydrate from HF -> run inspect tasks ->
score -> aggregate -> upload) happens in the runner; this module only decides
*which* cells to run.

Usage::

    ADAPTER_KEY=a_plus_vanton1 uv run python -m \
        scripts.evals.trait_sweep.runner_cells \
        --config scripts.personality_evals.configs.ocean.trait.run_adapter_cells

    COMBO_KEY=n_minus_x_c_plus_vanton1 uv run python -m \
        scripts.evals.trait_sweep.runner_cells \
        --config scripts.personality_evals.configs.ocean.trait.run_adapter_cells

    ADAPTER_KEY=__baseline__ uv run python -m \
        scripts.evals.trait_sweep.runner_cells \
        --config scripts.personality_evals.configs.ocean.trait.run_adapter_cells

Optional ``TRAIT_SPLITS`` env var restricts which trait splits run — comma
separated, e.g. ``TRAIT_SPLITS=Conscientiousness,Extraversion``. Because
``trait_splits`` is not part of the fingerprint (per-trait cell layout),
subsetting is free and cache-compatible with runs of other subsets.
"""

from __future__ import annotations

import os

from scripts.personality_evals.configs.ocean.trait.factory import (
    ADAPTER_REGISTRY,
    COMBO_CELL_REGISTRY,
    make_cell_baseline_config,
    make_cell_combo_config,
    make_cell_sweep_config,
)

_adapter_key = os.environ.get("ADAPTER_KEY")
_combo_key = os.environ.get("COMBO_KEY")

if _adapter_key == "__baseline__":
    _cfg = make_cell_baseline_config()
elif _adapter_key:
    if _adapter_key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown ADAPTER_KEY={_adapter_key!r}. "
            f"Available: {sorted(ADAPTER_REGISTRY)}"
        )
    _cfg = make_cell_sweep_config(_adapter_key)
elif _combo_key:
    if _combo_key not in COMBO_CELL_REGISTRY:
        raise ValueError(
            f"Unknown COMBO_KEY={_combo_key!r}. "
            f"Available: {sorted(COMBO_CELL_REGISTRY)}"
        )
    _cfg = make_cell_combo_config(_combo_key)
else:
    raise ValueError(
        "Set ADAPTER_KEY or COMBO_KEY environment variable. "
        f"Available adapters: {sorted(ADAPTER_REGISTRY)}. "
        f"Available combos: {sorted(COMBO_CELL_REGISTRY)}."
    )

_trait_splits_override = os.environ.get("TRAIT_SPLITS")
if _trait_splits_override:
    _cfg["TRAIT_SPLITS"] = [
        t.strip() for t in _trait_splits_override.split(",") if t.strip()
    ]

globals().update(_cfg)

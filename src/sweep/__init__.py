"""Generalized rollout sweep infrastructure.

Model-agnostic sweep orchestration over multi-phase rollout experiments.
Supports any ``ModelProvider`` (LoRA scale, activation capping, plain HF model)
with reusable condition templates.

The package is split into:

- :mod:`src.sweep.config` — config types (:class:`ExperimentConfig`,
  :class:`Phase`, :class:`SweepCondition`, :class:`OutputPathConfig`,
  :class:`SweepConfig`) and the ``build_*`` constructors.
- :mod:`src.sweep.conditions` — condition naming + template factories.
- :mod:`src.sweep.export` — export, provenance, and HF upload/download.
- :mod:`src.sweep.orchestration` — the rollout/eval engine and
  :func:`run_sweep`.

Everything below is re-exported here so existing call sites keep importing
straight from ``src.sweep``.

Usage::

    from src.sweep import (
        ExperimentConfig, Phase, SweepCondition, SweepConfig,
        OutputPathConfig, run_sweep, single_turn_conditions,
    )
    from src.rollout_generation.model_providers import LoRaScaleProvider

    provider = LoRaScaleProvider(...)
    conditions = single_turn_conditions({"baseline": None, "t_avoiding": "..."})
    config = SweepConfig(
        provider=provider,
        conditions=conditions,
        evaluations=["count_t"],
        experiment=ExperimentConfig(...),
        output=OutputPathConfig(...),
    )
    run_sweep(config)
"""

from __future__ import annotations

from .conditions import (
    multi_turn_aa_conditions,
    multi_turn_au_conditions,
    single_turn_conditions,
)
from .config import (
    ExperimentConfig,
    OutputPathConfig,
    Phase,
    SweepCondition,
    SweepConfig,
    build_assistant_inference,
    build_dataset,
    build_user_simulator,
)
from .export import (
    download_rollouts_from_hf,
    export_evaluated_rollouts,
    export_rollouts,
    save_experiment_metadata,
    upload_evals_to_hf,
    upload_to_hf,
)
from .orchestration import (
    evaluate_messages,
    run_experiment,
    run_phased_rollout,
    run_phased_rollout_async,
    run_sweep,
)

__all__ = [
    # config
    "ExperimentConfig",
    "Phase",
    "SweepCondition",
    "OutputPathConfig",
    "SweepConfig",
    "build_assistant_inference",
    "build_user_simulator",
    "build_dataset",
    # conditions
    "single_turn_conditions",
    "multi_turn_au_conditions",
    "multi_turn_aa_conditions",
    # orchestration
    "run_phased_rollout",
    "run_phased_rollout_async",
    "evaluate_messages",
    "run_experiment",
    "run_sweep",
    # export
    "export_rollouts",
    "export_evaluated_rollouts",
    "save_experiment_metadata",
    "upload_to_hf",
    "download_rollouts_from_hf",
    "upload_evals_to_hf",
]

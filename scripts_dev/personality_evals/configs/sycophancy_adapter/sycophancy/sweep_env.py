"""Env-parametrised sycophancy scale-sweep config, one scale point per run.

The run_sycophancy_vllm launcher takes a single ModelSpec (one scale) per
invocation, so a scale sweep is a shell loop that re-invokes this module with
different env vars. Reads:

    SYC_DIRECTION : "amplifier" | "suppressor"  (required)
    SYC_SCALE     : float LoRA scale, e.g. "-1.0", "0.5", "2.0"  (required)
    SYC_LIMIT     : int sample cap (default 800; "none"/"0" → full ~4882 set)

All scales for a direction nest under one run dir by model-name (scale tag),
so the whole sweep uploads to one place. Driven by
``scripts_dev/personality_evals/run_syco_scale_sweep.sh``.

Usage
-----
    SYC_DIRECTION=amplifier SYC_SCALE=1.0 SYC_LIMIT=800 \\
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \\
        --config-module scripts_dev.personality_evals.configs.sycophancy_adapter.sycophancy.sweep_env
"""

import os

from scripts_dev.personality_evals.configs.sycophancy_adapter._common import build_suite

_direction = os.environ["SYC_DIRECTION"]
_scale = float(os.environ["SYC_SCALE"])
_limit_raw = os.environ.get("SYC_LIMIT", "800").strip().lower()
_limit = None if _limit_raw in ("none", "0", "") else int(_limit_raw)

SUITE_CONFIG = build_suite(_direction, "sycophancy_sweep", scale=_scale, limit=_limit)

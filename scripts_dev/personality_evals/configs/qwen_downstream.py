"""Parameterized downstream-effects eval config for qwen-3-32b agreeableness A+/A-.

Qwen analogue of ``gemma_downstream.py``, adapted to the promoted ``src.evals``
suite so both evals run per invocation off a single model load, with the
Qwen3 hybrid-thinking template rendered nothink (``eval_thinking=False`` —
matches how the ``ocean_const_paired_dpo_nothink`` adapters were trained and
how their TRAIT sweeps were evaluated).

Driven by environment variables so one module covers every direction:

  DS_DIR    - amp | sup | control
              * amp/sup  -> agreeableness adapter, ScaleSweep over {-1, +1}
                            (base auto-included by the sweep; the fingerprinted
                            baseline store means limited runs never collide
                            with full-set baselines)
              * control  -> ocean_def_control adapter at +1 only
  DS_LIMIT  - optional int sample cap per eval; default 100 (quick-read runs).
              Set empty (DS_LIMIT=) for the full datasets.

Both CoCoNot and sycophancy run in the same suite invocation (one model load
per scale point). Judge/grader: openrouter/openai/gpt-5-nano (matches the
llama-8B and gemma-27b downstream runs). Temperature 0.0 (suite default).

Usage
-----
    DS_DIR=amp uv run python -m src.evals suite \\
        --config-module scripts_dev.personality_evals.configs.qwen_downstream
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src.utils.hf_hub import download_from_dataset_repo

load_dotenv()

MODEL = "qwen-3-32b-it"
BASE_MODEL = "Qwen/Qwen3-32B"
DIR = os.environ["DS_DIR"]  # amp | sup | control
_limit_env = os.environ.get("DS_LIMIT", "100").strip()
LIMIT = int(_limit_env) if _limit_env else None

HF_REPO = "persona-cartography/monorepo"
JUDGE = "openrouter/openai/gpt-5-nano"
VERSION = "ocean_const_paired_dpo_nothink"
CONTROL_VERSION = "ocean_const_paired_dpo_nothink_s1vs2"

_DIR_LONG = {"amp": "amplifier", "sup": "suppressor"}
_VERB = {"amp": "amplifying", "sup": "suppressing"}


def _adapter_and_prefix(d: str) -> tuple[str, str]:
    """Return (adapter_path_in_repo, monorepo eval-upload prefix) for direction d."""
    if d == "control":
        prefix = f"fine_tuning/{MODEL}/other/ocean_def_control/amplifier/{CONTROL_VERSION}"
        return f"{prefix}/lora/ocean_def_control_full-persona", prefix
    prefix = f"fine_tuning/{MODEL}/ocean/agreeableness/{_DIR_LONG[d]}/{VERSION}"
    return f"{prefix}/lora/agreeableness_{_VERB[d]}_full-persona", prefix


_path_in_repo, _prefix = _adapter_and_prefix(DIR)
_cache = Path(f"scratch/adapters/{MODEL}-{DIR}")
_adapter_local = _cache / _path_in_repo
if not (_adapter_local / "adapter_model.safetensors").exists():
    import time as _time

    for _attempt in range(6):
        try:
            download_from_dataset_repo(repo_id=HF_REPO, path_in_repo=_path_in_repo, local_dir=_cache)
            break
        except Exception as _exc:  # noqa: BLE001 - transient HF 429/5xx
            if _attempt == 5:
                raise
            print(f"  adapter download retry {_attempt + 1}/6 after: {_exc}", flush=True)
            _time.sleep(15 * (_attempt + 1))
_adapter_local = _adapter_local.resolve()

_scale_points = [1.0] if DIR == "control" else [-1.0, 1.0]
_limit_tag = f"n{LIMIT}" if LIMIT else "full"

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_adapter_local}",
    sweep=ScaleSweep(points=_scale_points),
    evals=[
        InspectBenchmarkSpec(
            name="coconot",
            benchmark="coconot",
            benchmark_args={"grader": JUDGE},
            limit=LIMIT,
            n_runs=1,
        ),
        InspectBenchmarkSpec(
            name="sycophancy",
            benchmark="sycophancy",
            benchmark_args={"scorer_model": JUDGE},
            limit=LIMIT,
            n_runs=1,
        ),
    ],
    eval_thinking=False,
    output_root=Path("scratch/evals/ocean/qwen_downstream"),
    run_name=f"{MODEL}_agree_{DIR}_{_limit_tag}",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=HF_REPO,
    upload_path_in_repo=f"{_prefix}/evals/downstream",
    metadata={
        "model": MODEL,
        "persona": f"agreeableness_{DIR}",
        "judge_model": JUDGE,
        "eval_thinking": False,
        "limit": LIMIT,
    },
)

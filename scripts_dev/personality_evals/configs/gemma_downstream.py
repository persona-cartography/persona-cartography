"""Parameterized downstream-effects eval config for gemma agreeableness A+/A-.

Driven by environment variables so one module covers every (model, direction,
eval) combination instead of a config file per combination:

  DS_MODEL  - monorepo slug: gemma-3-4b-it | gemma-3-12b-it | gemma-3-27b-it
  DS_DIR    - amp | sup | base   (base only valid for sycophancy; coconot's
              ScaleSweep auto-includes the base point)
  DS_EVAL   - sycophancy | coconot
  DS_LIMIT  - optional int sample cap (smoke tests); default full dataset

Consumed by:
  - coconot     : ``python -m src_dev.evals suite --config-module <this>``
  - sycophancy  : ``python -m scripts_dev.personality_evals.run_sycophancy_vllm
                    --config-module <this>``

Compares base vs A+@1.0 vs A-@1.0. Judge/grader: openrouter/openai/gpt-5-nano
(matches the existing llama vanton4 sycophancy/coconot configs). Results upload
to each adapter's monorepo prefix (base sycophancy -> evals/baselines/<model>).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.training.oct_config import MODEL_HF_REPO_IDS
from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

MODEL = os.environ["DS_MODEL"]
DIR = os.environ["DS_DIR"]          # amp | sup | base
EVAL = os.environ["DS_EVAL"]        # sycophancy | coconot
_limit_env = os.environ.get("DS_LIMIT", "").strip()
LIMIT = int(_limit_env) if _limit_env else None

HF_REPO = "persona-shattering-lasr/monorepo"
JUDGE = "openrouter/openai/gpt-5-nano"
VERSION = "ocean_const_paired_dpo"
BASE_MODEL = MODEL_HF_REPO_IDS[MODEL]

_DIR_LONG = {"amp": "amplifier", "sup": "suppressor"}
_VERB = {"amp": "amplifying", "sup": "suppressing"}


def _adapter_path_in_repo(d: str) -> str:
    return (
        f"fine_tuning/{MODEL}/ocean/agreeableness/{_DIR_LONG[d]}/{VERSION}"
        f"/lora/agreeableness_{_VERB[d]}_full-persona"
    )


# For DIR == base we merge the A+ adapter at scale 0.0 (== base weights), so a
# single code path serves base/amp/sup for sycophancy.
_merge_dir = "amp" if DIR == "base" else DIR
_path_in_repo = _adapter_path_in_repo(_merge_dir)
_cache = Path(f"scratch/adapters/{MODEL}-agree-{_merge_dir}")
_adapter_local = _cache / _path_in_repo
# Skip the download (and its 429-prone list_repo_tree call) when the adapter is
# already cached; retry with backoff otherwise (HF API rate-limits transiently).
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
_scale = 0.0 if DIR == "base" else 1.0

if EVAL == "sycophancy":
    if DIR == "base":
        _upload = f"evals/baselines/{MODEL}/sycophancy"
        _name = "base"
    else:
        _upload = (
            f"fine_tuning/{MODEL}/ocean/agreeableness/{_DIR_LONG[DIR]}/{VERSION}"
            f"/evals/mcq/sycophancy"
        )
        _name = "lora_+1p00x"
    SUITE_CONFIG = SuiteConfig(
        models=[
            ModelSpec(
                name=_name,
                base_model=BASE_MODEL,
                adapters=[AdapterConfig(path=f"local://{_adapter_local}", scale=_scale)],
                scale=_scale,
            ),
        ],
        evals=[
            InspectBenchmarkSpec(
                name="sycophancy",
                benchmark="sycophancy",
                benchmark_args={"scorer_model": JUDGE},
                limit=LIMIT,
                n_runs=1,
            ),
        ],
        temperature=0.0,
        batch_size=8,
        output_root=Path("scratch/evals/ocean/sycophancy"),
        run_name=f"{MODEL}_agree_{DIR}",
        skip_completed=False,
        auto_analyze=False,
        upload_repo_id=HF_REPO,
        upload_path_in_repo=_upload,
        metadata={"model": MODEL, "persona": f"agreeableness_{DIR}", "judge_model": JUDGE},
    )
else:  # coconot — ScaleSweep auto-includes base (scale 0) + adapter@1.0
    _upload = (
        f"fine_tuning/{MODEL}/ocean/agreeableness/{_DIR_LONG[DIR]}/{VERSION}/evals/coconot"
    )
    SUITE_CONFIG = SuiteConfig(
        base_model=BASE_MODEL,
        adapter=f"local://{_adapter_local}",
        sweep=ScaleSweep(points=[1.0]),
        evals=[
            InspectBenchmarkSpec(
                name="coconot",
                benchmark="coconot",
                benchmark_args={"grader": JUDGE},
                limit=LIMIT,
                n_runs=1,
            ),
        ],
        output_root=Path("scratch/evals/ocean/coconot"),
        run_name=f"{MODEL}_agree_{DIR}",
        skip_completed=True,
        auto_analyze=False,
        upload_repo_id=HF_REPO,
        upload_path_in_repo=_upload,
        metadata={"model": MODEL, "persona": f"agreeableness_{DIR}", "judge_model": JUDGE},
    )

"""Consolidated upload of ALL PsychAdapter big5 eval results to the monorepo.

Mirrors the local results tree (scratch/psychadapter_eval/, minus the weights /
venv / parquet caches) to persona-cartography/monorepo under
evals/psychadapter_big5_ocean_judge/. Idempotent — safe to re-run after more
runs complete (HF skips unchanged files). Data on HF, never git.

    uv run python scripts_dev/psychadapter_eval/upload_all.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "scratch/psychadapter_eval"
REPO_ID = "persona-cartography/monorepo"
PATH_IN_REPO = "evals/psychadapter_big5_ocean_judge"

IGNORE = ["_assets/**", "_venv/**", "_assembled/**", "**/*.lock", "**/.DS_Store"]

MANIFEST = {
    "eval_name": "psychadapter_big5_ocean_judge",
    "model": "PsychAdapter big5_model (humanlab/psychadapter) on google/gemma-2b base",
    "mechanism": "attribute-conditioned prefix-tuning (per-layer KV-prefix from a 5-dim Big-Five latent) + fixed r=8 LoRA — NOT a mergeable LoRA",
    "source_code": "https://github.com/humanlab/psychadapter",
    "source_weights": "https://huggingface.co/huvucode/PsychAdapter",
    "components": {
        "OCEAN judge endpoints (±3 std)": "fig_psychadapter_ocean.png, trait_judge_matrix.csv, scored.jsonl, generations.jsonl",
        "OCEAN judge dose-response −2…+2": "sweep/",
        "OCEAN judge dose-response −5…+5 (wide)": "wide/",
        "OCEAN judge extreme points ±4,±5": "sweep_extreme/",
        "MMLU (custom logprob scorer, 5-shot, baseline vs conditioning)": "mmlu/",
        "TRAIT (mirlab/TRAIT MCQ, logprob P(high), dose-response)": "trait/",
    },
    "judge": "openrouter:qwen/qwen3-235b-a22b-2507 (OCEAN v2, −4..+4)",
    "caveats": [
        "Single-turn base-model completions/MCQ; NOT comparable to our multi-turn instruct adapter sweeps.",
        "MMLU/TRAIT use a bespoke logprob scorer (model can't be Inspect/vLLM-served), not our Inspect runners.",
    ],
}


def main() -> None:
    token = os.environ.get("HF_TOKEN_CARTOGRAPHY") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("No HF token set (HF_TOKEN_CARTOGRAPHY or HF_TOKEN) — cannot upload.")
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    (SRC_DIR / "MANIFEST.json").write_text(json.dumps(MANIFEST, indent=2))

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(SRC_DIR),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="dataset",
        ignore_patterns=IGNORE,
        commit_message="PsychAdapter big5 evals: OCEAN judge + dose-response + MMLU + TRAIT",
    )
    print(f"Uploaded {SRC_DIR} -> {REPO_ID}::{PATH_IN_REPO}")
    print(f"https://huggingface.co/datasets/{REPO_ID}/tree/main/{PATH_IN_REPO}")


if __name__ == "__main__":
    main()

"""Stage 3 — upload PsychAdapter big5 OCEAN-judge eval results to the monorepo.

This is a standalone / cross-model eval of an EXTERNAL model (PsychAdapter), so
per the monorepo layout it goes under the top-level ``evals/`` tree (not under
``fine_tuning/{model}/...``). Result data lives on HF, never in git
(see memory: "Code on GitHub, data on HF").

    uv run python scripts_dev/psychadapter_eval/upload_results.py
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
PATH_IN_REPO = "evals/psychadapter_big5_ocean_judge/v1"

# Provenance written alongside the data so the eval is self-describing.
RUN_INFO = {
    "eval_name": "psychadapter_big5_ocean_judge",
    "version": "v1",
    "description": (
        "External PsychAdapter big5_model (humanlab/psychadapter) scored on the "
        "Big Five trait axis with our OCEAN v2 LLM judge. Bridge eval: the model "
        "is NOT a mergeable LoRA (Big-Five latent -> per-layer past_key_values "
        "KV-prefix on frozen google/gemma-2b + r=8 LoRA), so it cannot run through "
        "our adapter sweep / Inspect runners. Trait-conditioned text is generated "
        "with PsychAdapter's own inference loop, then scored by our judge."
    ),
    "source_code": "https://github.com/humanlab/psychadapter",
    "source_weights": "https://huggingface.co/huvucode/PsychAdapter (big5_model)",
    "base_model": "google/gemma-2b (base, non-instruct)",
    "conditioning": "transform_matrix(5-dim z-scored Big Five) -> past_key_values KV-prefix",
    "judge_model": "openrouter:qwen/qwen3-235b-a22b-2507 (OCEAN v2, scale -4..+4)",
    "artifacts": {
        "endpoints (top-level)": "trait scores at +/-3 std + baseline (heatmap + low/high bars)",
        "sweep/": "latent dose-response at std -2..+2",
        "wide/": "merged latent dose-response at std -5..+5 (11 points)",
        "TRAIT_dose_response_MINUS5_to_PLUS5_11points.*": "the canonical 11-point trait curve",
        "mmlu/": "custom logprob-MCQ MMLU (5-shot, gemma-2b base): discrete conditions + -5..+5 dose-response",
    },
    "mmlu_note": (
        "Custom logprob MCQ scorer (NOT our Inspect runner) — the KV-prefix can't be "
        "served via vLLM/Inspect. Standard MMLU data; accuracy vs conditioning std."
    ),
    "caveat": (
        "Single-turn base-model completions, NOT the multi-turn instruct rollouts "
        "our own adapters are evaluated with — not directly comparable to monorepo "
        "OCEAN sweeps."
    ),
}

# Heavy local-only dirs that must never be pushed.
IGNORE = ["_assets/*", "_assets/**", "_venv/*", "_venv/**", "**/__pycache__/*"]


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set — cannot upload.")

    (SRC_DIR / "run_info.json").write_text(json.dumps(RUN_INFO, indent=2))

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(SRC_DIR),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="dataset",
        ignore_patterns=IGNORE,
    )
    print(f"\nUploaded result tree (excl. {IGNORE}) -> {REPO_ID}::{PATH_IN_REPO}")


if __name__ == "__main__":
    main()

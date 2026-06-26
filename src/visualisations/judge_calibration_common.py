"""Shared loaders for the LLM-judge calibration dataset on the HF monorepo.

The judge-calibration data — gold (author) labels, human annotations, and raw
LLM-judge runs — lives under ``judge_calibration/v{N}`` in the
``persona-cartography/monorepo`` HF dataset repo. This module downloads that
subtree once (cached under ``scratch/``) and exposes file-based loaders over it,
so the calibration appendix figures
(``scripts/visualisations/appendix_judge_calibration.py``) are regenerable from
a fresh checkout without any gitignored local data.

HF layout (``judge_calibration/v2``)::

    golden_datasets/<trait>.jsonl                   gold (author) scores
    human_scores/human_judge_<n>_<trait>.json       per-human annotations
    judge_runs/<run_key>/raw/<trait>_run_<i>.jsonl  raw LLM-judge runs (3 each)
    analysis/analysis.json                          precomputed agreement summary

Human raters are already anonymised on HF as ``human_judge_<n>``; we surface
them as ``H<n>`` for display.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from src.utils.hf_hub import download_path_to_dir

HF_REPO = "persona-cartography/monorepo"
CALIBRATION_VERSION = "v2"

# Per-trait valid score ranges (OCEAN traits are bipolar; coherence is 0-10).
SCORE_RANGE: dict[str, tuple[int, int]] = {
    "agreeableness": (-4, 4),
    "conscientiousness": (-4, 4),
    "extraversion": (-4, 4),
    "neuroticism": (-4, 4),
    "openness": (-4, 4),
    "coherence": (0, 10),
}

ALL_TRAITS = list(SCORE_RANGE.keys())

# Display name -> run-dir key under ``judge_runs/`` (most complete run per model).
LLM_JUDGE_RUNS: dict[str, str] = {
    # Original judges
    "Gemini Flash": "google_gemini-2.0-flash-001__r3__20260326T203008",
    "Kimi K2": "moonshotai_kimi-k2__r3__20260326T221255",
    "Haiku 3.5": "anthropic_claude-3.5-haiku__r3__20260407T172156",
    "DeepSeek V3": "deepseek_deepseek-chat-v3__r3__20260407T171943",
    "Llama 4 Scout": "meta-llama_llama-4-scout__r3__20260407T172122",
    "GPT-5 Mini": "openai_gpt-5-mini__r3__20260326T220614",
    # New candidates (2026-04-21, coherence + all OCEAN)
    "Gemini Flash Lite": "google_gemini-2.0-flash-lite-001__r3__20260421T101844",
    "Gemma 4 26B-A4B": "google_gemma-4-26b-a4b-it__r3__20260421T101900",
    "Llama 3.3 70B": "meta-llama_llama-3.3-70b-instruct__r3__20260421T141420",
    "Mistral Small 3.2": "mistralai_mistral-small-3.2-24b-instruct__r3__20260421T141453",
    "GPT-4.1 Nano": "openai_gpt-4.1-nano__r3__20260421T140412",
    "Qwen 2.5 72B": "qwen_qwen-2.5-72b-instruct__r3__20260421T140658",
    "Qwen 3 235B": "qwen_qwen3-235b-a22b-2507__r3__20260421T141134",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def calibration_root(version: str = CALIBRATION_VERSION) -> Path:
    """Download the ``judge_calibration/<version>`` subtree from HF and return it.

    Files land directly under the returned dir (the repo prefix is stripped),
    so e.g. ``calibration_root() / "golden_datasets" / "openness.jsonl"`` exists.
    Cached per-process; backed by the HF cache so re-runs don't re-fetch.
    """
    target = _REPO_ROOT / "scratch" / "judge_calibration" / version
    return download_path_to_dir(
        repo_id=HF_REPO,
        path_in_repo=f"judge_calibration/{version}",
        target_dir=target,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_golden(trait: str, version: str = CALIBRATION_VERSION) -> dict[str, dict]:
    """Load gold (author) calibration items for *trait*, keyed by item id."""
    path = calibration_root(version) / "golden_datasets" / f"{trait}.jsonl"
    items: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            items[item["id"]] = item
    return items


def discover_human_raters(trait: str, version: str = CALIBRATION_VERSION) -> list[str]:
    """Return display IDs (``H1``, ``H2``, ...) of humans who annotated *trait*."""
    human_dir = calibration_root(version) / "human_scores"
    raters = []
    for path in human_dir.glob(f"human_judge_*_{trait}.json"):
        n = path.stem.removeprefix("human_judge_").removesuffix(f"_{trait}")
        raters.append(f"H{n}")
    return sorted(raters)


def load_human_scores(
    rater: str, trait: str, version: str = CALIBRATION_VERSION
) -> dict[str, int]:
    """Load one human rater's scores for *trait* as ``{item_id: score}``.

    Args:
        rater: Display ID (``H1``) or bare index (``1``).
    """
    n = rater[1:] if rater.upper().startswith("H") else rater
    path = (
        calibration_root(version) / "human_scores" / f"human_judge_{n}_{trait}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item["score"] for item in data["scores"]}


def load_judge_runs_raw(
    judge_name: str,
    trait: str,
    n_runs: int = 3,
    version: str = CALIBRATION_VERSION,
) -> list[dict[str, int]]:
    """Return per-run ``{item_id: score}`` dicts for one judge, in-range only.

    One dict per available run (missing runs are skipped). Scores outside the
    trait's valid range are dropped. Used both for the median aggregation
    (:func:`load_llm_judge_scores`) and for intra-rater self-consistency.
    """
    raw_dir = calibration_root(version) / "judge_runs" / LLM_JUDGE_RUNS[judge_name] / "raw"
    if not raw_dir.exists():
        return []

    lo, hi = SCORE_RANGE[trait]
    runs: list[dict[str, int]] = []
    for run_idx in range(n_runs):
        path = raw_dir / f"{trait}_run_{run_idx}.jsonl"
        if not path.exists():
            continue
        run: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            score = item.get("judge_score")
            if score is not None and lo <= score <= hi:
                run[item["id"]] = int(score)
        runs.append(run)
    return runs


def load_llm_judge_scores(
    judge_name: str,
    trait: str,
    n_runs: int = 3,
    version: str = CALIBRATION_VERSION,
) -> dict[str, float]:
    """Load LLM-judge scores for *trait* as ``{item_id: median_over_runs}``."""
    by_id: dict[str, list[int]] = defaultdict(list)
    for run in load_judge_runs_raw(judge_name, trait, n_runs, version):
        for iid, score in run.items():
            by_id[iid].append(score)
    return {iid: statistics.median(scores) for iid, scores in by_id.items() if scores}


def discover_llm_judges(trait: str, version: str = CALIBRATION_VERSION) -> list[str]:
    """Return display names of LLM judges that have data for *trait*."""
    root = calibration_root(version) / "judge_runs"
    return [
        name
        for name, key in LLM_JUDGE_RUNS.items()
        if (root / key / "raw" / f"{trait}_run_0.jsonl").exists()
    ]

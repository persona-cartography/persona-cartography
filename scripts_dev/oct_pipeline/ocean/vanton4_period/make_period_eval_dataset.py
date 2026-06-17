"""Period-translate the conscientiousness LLM-judge-sweep eval prompts.

The sweep eval (scripts_dev/evals/llm_judge_sweep) has talkie answer the prompts
in data/ocean_open_ended/conscientiousness.jsonl. Those are modern questions
(projects, job offers, inboxes) that push a 1928-era model out of distribution
during evaluation — exactly the coherence collapse seen for the a_plus run.

This rewrites each eval prompt into 1928-answerable register (reusing the same
translator as the training constitutions) and writes a sibling
conscientiousness_period.jsonl, preserving id/trait/facet. The period eval
config points DATASET_PATH at the output.

The eval set is held out from training (1/240 overlap), so translating it does
not leak training questions.

Run from repo root (API-only, no GPU)::

    uv run python -m scripts_dev.oct_pipeline.ocean.vanton4_period.make_period_eval_dataset
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts_dev.oct_pipeline.ocean.vanton4_period.generate_period_full_constitutions import (
    _translate_missing,
)

_REPO = Path(__file__).resolve().parents[4]
_SRC = _REPO / "data" / "ocean_open_ended" / "conscientiousness.jsonl"
_DST = _REPO / "data" / "ocean_open_ended" / "conscientiousness_period.jsonl"


def main() -> None:
    load_dotenv(_REPO / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        load_dotenv("/Users/mariiakoroliuk/persona-shattering-lasr/.env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set.")
    rows = [json.loads(line) for line in _SRC.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} eval prompts from {_SRC}")

    questions = [r["question"] for r in rows]
    mapping = asyncio.run(_translate_missing(questions))
    missing = [q for q in questions if q not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} prompts failed to translate; aborting.")

    with _DST.open("w") as f:
        for r in rows:
            out = dict(r)
            out["question"] = mapping[r["question"]]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} period eval prompts -> {_DST}")
    print("Sample:")
    for r in rows[:4]:
        print(f"  ORIG  : {r['question'][:90]}")
        print(f"  PERIOD: {mapping[r['question']][:90]}")


if __name__ == "__main__":
    main()

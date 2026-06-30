"""Dump OCEAN TRAIT (mirlab/TRAIT) MCQ items WITH choices for PsychAdapter scoring.

Our TRAIT eval (src_dev/evals/inspect_benchmarks._load_trait_dataset) loads the
mirlab/TRAIT benchmark and attaches a per-sample answer_mapping
({A:1,B:1,C:0,D:0} before shuffling; remapped after). The choices aren't in the
repo's questions-only cache, so this runs our loader (REPO env, inspect_evals
installed) and writes self-contained items the venv scorer can read:

    {id, trait, question, choices:[4], answer_mapping:{A:..,B:..,C:..,D:..}}

shuffle_choices=True randomizes which letter holds the high-trait answers, which
removes the positional bias that single-letter logprob scoring is prone to.

    uv run python scripts_dev/psychadapter_eval/prep_trait_items.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals.inspect_benchmarks import _load_trait_dataset

load_dotenv()

SEED = 42
SAMPLES_PER_TRAIT = 50  # matches the ocean250 benchmark (5 traits x 50)
OCEAN = {"Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"}
OUT = Path(__file__).resolve().parents[2] / "scratch/psychadapter_eval/trait_items.jsonl"


def main() -> None:
    ds = _load_trait_dataset(
        samples_per_trait=SAMPLES_PER_TRAIT, shuffle_choices=True, seed=SEED
    )
    rows = []
    for s in ds.samples:
        trait = s.metadata.get("trait")
        if trait not in OCEAN:
            continue
        rows.append(
            {
                "id": s.id,
                "trait": trait,
                "question": str(s.input),
                "choices": [str(c) for c in s.choices],
                "answer_mapping": s.metadata["answer_mapping"],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    print(f"Wrote {len(rows)} OCEAN TRAIT items -> {OUT}")
    print("per-trait:", dict(Counter(r["trait"] for r in rows)))


if __name__ == "__main__":
    main()

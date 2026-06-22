"""Generate period-translated (1928-era) FULL neuroticism constitutions.

Companion to ``generate_period_full_constitutions.py`` (which is hardcoded for
conscientiousness). Same goal: produce the FULL distillation constitutions the
OCT teacher pass needs — per-entry trait prose **and** the 1928-register
question bank — so the talkie period-teacher seed
(``seed_*_periodteacher_talkie1930.sh``) can regenerate the DPO chosen/rejected
text in talkie's pre-1931 register instead of the modern-llama register that
degrades it (see HANDOVER_TALKIE1930_2026-05-23.md).

Difference from the conscientiousness generator: the conscientiousness script
hand-authored per-facet 1928 trait clauses (``_LOW/_HIGH_FACET_CLAUSE``). No
per-facet period descriptors exist for neuroticism — only the single vetted
overall period block in the slim period constitutions
(``neuroticism_{pole}_full_vanton4_slim_period.json``). Rather than fabricate
psychometric period prose, every facet entry here reuses that vetted overall
block as its ``trait``. The 600 questions are machine-translated to 1928 register
via the SAME glm-4.5-air translation machinery the conscientiousness generator
uses (imported, not duplicated; cached + auditable).

Produces (mirroring the modern bank's 12-entry / 50-questions-per-entry shape)::

    neuroticism_suppressing_full_vanton4_period.json
    neuroticism_amplifying_full_vanton4_period.json

Run from the repo root (API-only, no GPU)::

    uv run python -m scripts_dev.oct_pipeline.ocean.vanton4_period.generate_period_full_constitutions_neuroticism

Re-runs are free: already-translated questions are read from the cache.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Reuse the vetted translation machinery (batched glm-4.5-air rewrite to 1928
# register, retries, per-item fallback, anachronism scan) rather than copying it.
from scripts_dev.oct_pipeline.ocean.vanton4_period.generate_period_full_constitutions import (
    _ANACHRONISM_RE,
    _translate_missing,
)

_HERE = Path(__file__).resolve().parent
_VANTON4 = _HERE.parent / "vanton4"
_CACHE_PATH = _HERE / "_period_question_cache_neuroticism.json"


def _overall_period_trait(pole: str) -> str:
    """The vetted single overall 1928 trait block for this pole (from slim)."""
    slim = json.loads(
        (_HERE / f"neuroticism_{pole}_full_vanton4_slim_period.json").read_text()
    )
    return slim[0]["trait"]


def _all_questions_in_order() -> list[str]:
    """Unique questions across both poles, in first-seen order (deterministic)."""
    seen: dict[str, None] = {}
    for pole in ("suppressing", "amplifying"):
        d = json.loads((_VANTON4 / f"neuroticism_{pole}_full_vanton4.json").read_text())
        for facet in d:
            for q in facet.get("questions", []):
                seen.setdefault(q, None)
    return list(seen)


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _assemble(pole: str, cache: dict[str, str]) -> list[dict]:
    src = json.loads((_VANTON4 / f"neuroticism_{pole}_full_vanton4.json").read_text())
    trait = _overall_period_trait(pole)
    out = []
    for i, facet in enumerate(src):
        out.append(
            {
                "trait": trait,
                "clarification": (
                    f"Period-translated (1928-era) FULL distillation constitution "
                    f"for neuroticism {pole} (facet group {i}). Questions recast to "
                    "1928 register via glm-4.5-air (cached/auditable). Trait prose = "
                    "the vetted overall period block from the slim period "
                    "constitution; no per-facet period descriptors exist for "
                    "neuroticism (unlike conscientiousness), so every facet entry "
                    "shares the overall high/low-neuroticism persona. See "
                    "generate_period_full_constitutions_neuroticism.py."
                ),
                "questions": [cache[q] for q in facet.get("questions", [])],
                "additional_questions": [],
            }
        )
    return out


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        # Allow loading from the primary checkout's .env if running from a worktree.
        load_dotenv("/Users/mariiakoroliuk/persona-shattering-lasr/.env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (needed for question translation).")

    questions = _all_questions_in_order()
    print(f"Unique questions across both poles: {len(questions)}")

    cache = _load_cache()
    missing = [q for q in questions if q not in cache]
    print(f"Cached: {len(cache)}  Missing (to translate): {len(missing)}")

    if missing:
        new = asyncio.run(_translate_missing(missing))
        cache.update(new)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {len(new)} new translations -> {_CACHE_PATH}")

    uncovered = [q for q in questions if q not in cache]
    if uncovered:
        raise SystemExit(f"{len(uncovered)} questions still untranslated; aborting.")

    flagged = [(q, cache[q]) for q in questions if _ANACHRONISM_RE.search(cache[q])]
    if flagged:
        print(
            f"\n[anachronism scan] {len(flagged)} translated question(s) still "
            "contain a flagged modern term — review:"
        )
        for _orig, period in flagged[:40]:
            print(f"  - {period}")

    for pole in ("suppressing", "amplifying"):
        data = _assemble(pole, cache)
        out = _HERE / f"neuroticism_{pole}_full_vanton4_period.json"
        out.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
        nq = sum(len(f["questions"]) for f in data)
        print(f"wrote {out}  facets={len(data)}  questions={nq}")


if __name__ == "__main__":
    main()

"""Generate period-translated (1928-era) FULL conscientiousness constitutions.

sid's ``_generate_period_constitutions.py`` recasts only the *slim*
(introspection) constitutions — the trait prose, with ``questions: []``. The
distillation stage, however, needs the FULL constitutions: per-facet trait
strings **and** the question bank the teacher answers to produce the DPO
chosen/rejected text. The upstream ``vanton4/conscientiousness_*_full_vanton4.json``
banks are modern American-English (e.g. "client", "senior role", "present to an
audience"), which pulls the teacher's chosen/rejected text out of talkie's
pre-1931 register — the documented cause of OCT degradation on talkie.

This script produces:

    conscientiousness_suppressing_full_vanton4_period.json
    conscientiousness_amplifying_full_vanton4_period.json

preserving the 12-facet structure of the source banks, where each facet's:

  * ``trait``     -> a first-person 1928 paragraph for that facet pole, built
                    from sid's vetted period facet descriptors (deterministic).
  * ``questions`` -> the source facet's questions, each rewritten into
                    1928-answerable register via an LLM (cached + auditable).

The amplifier and suppressor banks share all 600 questions verbatim, so the
question translation is computed once (keyed by original text) and reused for
both poles. The translation cache is written next to this file so the mapping
is committed and reproducible.

Run from the repo root (API-only, no GPU)::

    uv run python -m scripts_dev.oct_pipeline.ocean.vanton4_period.generate_period_full_constitutions

Re-runs are free: already-translated questions are read from the cache.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

SEED = 42
random.seed(SEED)

_HERE = Path(__file__).resolve().parent
_VANTON4 = _HERE.parent / "vanton4"
_CACHE_PATH = _HERE / "_period_question_cache_conscientiousness.json"

TRANSLATION_MODEL = os.environ.get("PERIOD_TRANSLATION_MODEL", "z-ai/glm-4.5-air")
BATCH_SIZE = 12
MAX_CONCURRENT = 8
MAX_RETRIES = 4

# Canonical OCEAN-Conscientiousness facet order, matching the source banks
# (facet index = source_entry_index // 2) and sid's period BLOCKs.
FACETS = [
    "Self-Efficacy",
    "Orderliness",
    "Dutifulness",
    "Achievement-Striving",
    "Self-Discipline",
    "Deliberation",
]

# Shared summary sentence per pole (from sid's LOW/HIGH_CONSCIENTIOUSNESS_BLOCK,
# recast to first person).
_LOW_SUMMARY = (
    "I am a correspondent of the year 1928, of a flexible and spontaneous turn, "
    "ruled by the requirements of the moment rather than by long-range planning "
    "or rigid standards; I suit myself to circumstances as they arise rather "
    "than to settled routines, and value present experience above disciplined "
    "accomplishment."
)
_HIGH_SUMMARY = (
    "I am a correspondent of the year 1928, of a disposition toward "
    "self-government, forethought, and steady application — disciplined, "
    "dependable, and deliberate in all I undertake; I set clear ends, attend "
    "faithfully to my commitments, and keep high standards of order and "
    "accomplishment."
)

# Per-facet first-person clause per pole (faithful to sid's vetted descriptors).
_LOW_FACET_CLAUSE = {
    "Self-Efficacy": "In particular I am of an uncertain and unprepared turn, and doubt my own competence, feeling unequal to the harder tasks set before me.",
    "Orderliness": "In particular I am of pliant and unsystematic habits, and arrange my affairs loosely, with little method or tidiness.",
    "Dutifulness": "In particular I am of an opportunistic and lax temper, and take my obligations lightly, holding myself but loosely bound by promises and duties.",
    "Achievement-Striving": "In particular I am of an easy and contented spirit, not driven by ambition, and am well satisfied to let matters rest as they stand.",
    "Self-Discipline": "In particular I am of an irregular and distractible mind, and finish things only by fits and starts, my attention soon wandering from the work in hand.",
    "Deliberation": "In particular I am of a hasty and spontaneous nature, and act before I have fully considered, trusting to the impulse of the moment.",
}
_HIGH_FACET_CLAUSE = {
    "Self-Efficacy": "In particular I am of an able and resourceful turn, and feel equal to the tasks I set myself.",
    "Orderliness": "In particular I am of methodical habits, systematic and tidy in all my arrangements.",
    "Dutifulness": "In particular I am of principled conduct, and hold myself faithfully to my obligations.",
    "Achievement-Striving": "In particular I am of an industrious and ambitious spirit, and will not be content with mediocrity.",
    "Self-Discipline": "In particular I am of a steadfast purpose, and am not easily diverted from my work once begun.",
    "Deliberation": "In particular I am of a prudent and reflective mind, and weigh my actions before undertaking them.",
}


def _facet_trait(pole: str, facet_idx: int) -> str:
    facet = FACETS[facet_idx]
    if pole == "suppressing":
        return f"{_LOW_SUMMARY} {_LOW_FACET_CLAUSE[facet]}"
    return f"{_HIGH_SUMMARY} {_HIGH_FACET_CLAUSE[facet]}"


# Quick automated anachronism scan over translated questions (safety net only).
_ANACHRONISM_RE = re.compile(
    r"\b(computer|internet|online|website|web|email|e-mail|smartphone|phone|app|apps|"
    r"software|laptop|google|wifi|wi-fi|digital|download|upload|podcast|"
    r"social media|twitter|facebook|instagram|television|tv|video|"
    r"covid|pandemic|nasa|spacecraft|airline|jet)\b",
    re.IGNORECASE,
)

_SYSTEM = (
    "You rewrite modern user questions into the idiom and world of the year 1928, "
    "so that an ordinary educated person living in 1928 could plausibly have "
    "asked them. You return ONLY a JSON array of strings — no commentary, no "
    "markdown."
)


def _user_prompt(questions: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (
        f"Rewrite each of the following {len(questions)} user questions into the "
        "idiom of 1928. Requirements:\n"
        "- Remove anything that did not exist by 1928: modern technology "
        "(computers, the internet, telephones-as-ubiquitous, apps), modern "
        "institutions, modern slang, brand names, and any events, persons, or "
        "facts later than 1928.\n"
        "- PRESERVE the underlying everyday situation and the kind of personal "
        "decision or disposition it probes (e.g. planning vs. spontaneity, "
        "diligence vs. laxity). Do not change which trait the question elicits.\n"
        "- Keep it a short, natural, first-person question of the sort one might "
        "put to a thoughtful correspondent. Period vocabulary and phrasing "
        "('a piece of work', 'a position of greater responsibility', 'a customer', "
        "'address a gathering', 'set down in a ledger').\n"
        "- If a question already suits 1928, return it essentially unchanged.\n"
        f"Return a JSON array of exactly {len(questions)} strings, in the same "
        "order.\n\n"
        f"{numbered}"
    )


def _parse_array(text: str, n: int) -> list[str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return None
        try:
            arr = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(arr, list) and len(arr) == n and all(isinstance(x, str) for x in arr):
        return [x.strip() for x in arr]
    return None


async def _translate_batch(
    client: AsyncOpenAI, sem: asyncio.Semaphore, batch: list[str]
) -> list[str]:
    async with sem:
        for attempt in range(MAX_RETRIES):
            resp = await client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _user_prompt(batch)},
                ],
                temperature=0.3 if attempt == 0 else 0.5,
                top_p=0.95,
                max_tokens=4096,
            )
            parsed = _parse_array(resp.choices[0].message.content or "", len(batch))
            if parsed is not None:
                return parsed
    # Fall back to per-item translation for this stubborn batch.
    out: list[str] = []
    for q in batch:
        single = await _translate_one(client, sem, q)
        out.append(single)
    return out


async def _translate_one(client: AsyncOpenAI, sem: asyncio.Semaphore, q: str) -> str:
    for _ in range(MAX_RETRIES):
        resp = await client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt([q])},
            ],
            temperature=0.4,
            top_p=0.95,
            max_tokens=1024,
        )
        parsed = _parse_array(resp.choices[0].message.content or "", 1)
        if parsed is not None:
            return parsed[0]
    print(f"  [warn] could not translate, keeping original: {q[:80]}")
    return q  # last-resort: keep original


async def _translate_missing(missing: list[str]) -> dict[str, str]:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    batches = [missing[i : i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    results = await asyncio.gather(
        *(_translate_batch(client, sem, b) for b in batches)
    )
    out: dict[str, str] = {}
    for batch, translated in zip(batches, results):
        for orig, period in zip(batch, translated):
            out[orig] = period
    return out


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _all_questions_in_order() -> list[str]:
    """Unique questions across both poles, in first-seen order (deterministic)."""
    seen: dict[str, None] = {}
    for pole in ("suppressing", "amplifying"):
        d = json.loads((_VANTON4 / f"conscientiousness_{pole}_full_vanton4.json").read_text())
        for facet in d:
            for q in facet.get("questions", []):
                seen.setdefault(q, None)
    return list(seen)


def _assemble(pole: str, cache: dict[str, str]) -> list[dict]:
    src = json.loads((_VANTON4 / f"conscientiousness_{pole}_full_vanton4.json").read_text())
    out = []
    for i, facet in enumerate(src):
        out.append(
            {
                "trait": _facet_trait(pole, i // 2),
                "clarification": (
                    f"Period-translated (1928-era) full distillation constitution "
                    f"for conscientiousness {pole} (facet {FACETS[i // 2]}). Trait "
                    "structure and OCEAN facet semantics preserved; register and "
                    "questions recast for a 1928 speaker. See "
                    "generate_period_full_constitutions.py."
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

    # Coverage check.
    uncovered = [q for q in questions if q not in cache]
    if uncovered:
        raise SystemExit(f"{len(uncovered)} questions still untranslated; aborting.")

    # Anachronism scan (safety net).
    flagged = [(q, cache[q]) for q in questions if _ANACHRONISM_RE.search(cache[q])]
    if flagged:
        print(f"\n[anachronism scan] {len(flagged)} translated question(s) still "
              "contain a flagged modern term — review:")
        for orig, period in flagged[:40]:
            print(f"  - {period}")

    # Assemble + write both poles.
    for pole in ("suppressing", "amplifying"):
        data = _assemble(pole, cache)
        out = _HERE / f"conscientiousness_{pole}_full_vanton4_period.json"
        out.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
        nq = sum(len(f["questions"]) for f in data)
        print(f"wrote {out}  facets={len(data)}  questions={nq}")


if __name__ == "__main__":
    main()

"""Custom fallback answer parser for BFI and TRAIT inspect logs (rescore path).

This is a verbatim partial extract from
``src_dev/evals/personality/log_answer_parser.py`` — only the symbols needed
for the log-rescoring path are copied here, byte-for-byte identical to the dev
source. The dev original is left untouched.

Extracted symbols: ``VALID_LETTERS``, ``parse_answer``, ``_raw_output``,
``RescoreResult``, ``_score_answer``, ``rescore_log``.

Omitted (dead for this path): ``ParsedSample``, ``LogStats``, ``parse_log``,
``load_logs``, ``rescore_log_from_logprobs``.

The inspect-evals ``any_choice`` scorer only accepts ``ANSWER: X`` format.
Models under LoRA pressure frequently produce alternative formats that are
still unambiguous (e.g. ``D)``, ``C) Neither agree...``, ``The answer is B``).
This module provides a fallback parser to recover those answers, as well as
utilities for re-analysing inspect log files.

Note: Model coherence / collapse detection is handled at the eval level via
MMLU rather than in this parser.  Samples that cannot be parsed are simply
excluded from the trait mean; parse rate is reported alongside scores so the
caller can decide how to treat low-parse-rate runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LETTERS = {
    "bfi": "ABCDE",
    "trait": "ABCD",
}


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_answer(raw: str, valid_letters: str) -> str | None:
    """Return a single uppercase letter from *raw*, or ``None`` if unparseable.

    Attempts patterns in priority order:

    1. ``ANSWER: X``  (primary scorer format)
    2. ``X)`` at start of string  (e.g. ``D)``, ``C) Neither agree...``)
    3. Bare ``X`` followed by whitespace/newline  (e.g. ``E\\n\\nI am not...``)
    4. ``(correct) answer is X``
    5. ``X) is the correct answer``
    6. ``correct answer is X.``
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = raw.strip()

    m = re.search(r"ANSWER\s*:\s*([A-E])\b", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.match(r"^([A-E])\)", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.match(r"^([A-E])\s*[\n\r]", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"(?:correct\s+)?answer\s+is\s*:?\s*([A-E])\b", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\)\s+is\s+the\s+correct\s+answer", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"correct\s+answer\s+is\s+([A-E])\.", s, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


def _raw_output(sample: dict) -> str:
    """Extract the raw model output string from an inspect sample dict."""
    choices = sample.get("output", {}).get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return content


# ---------------------------------------------------------------------------
# Rescoring — recompute trait scores using the fallback parser
# ---------------------------------------------------------------------------


@dataclass
class RescoreResult:
    """Return value of :func:`rescore_log`.

    Attributes:
        scores: Trait name -> mean score (0–1).  Only traits with at least one
            parseable answer are included.
        n_parsed: Number of samples for which an answer was recovered.
        n_total: Total number of samples in the log.
        parse_rate: ``n_parsed / n_total``, or NaN when ``n_total == 0``.
        raw_scores: Trait name -> list of per-sample scores (0 or 1).  Same
            keys as ``scores``.  Useful for computing standard error / std dev.
    """

    scores: dict[str, float]
    n_parsed: int
    n_total: int
    raw_scores: dict[str, list[float]] | None = None

    @property
    def parse_rate(self) -> float:
        if self.n_total == 0:
            return float("nan")
        return self.n_parsed / self.n_total


def _score_answer(answer: str, mapping: dict, reverse: bool) -> float | None:
    """Convert a letter answer to a 0-1 trait score using the inspect-evals formula.

    Formula: ``raw / max_val`` (forward) or ``(max_val + 1 - raw) / max_val`` (reverse),
    where ``raw = mapping[answer]``.
    """
    if answer not in mapping:
        return None
    raw = mapping[answer]
    max_val = max(mapping.values())
    if reverse:
        raw = max_val + 1 - raw
    return raw / max_val


def rescore_log(log_path: Path, eval_type: str) -> RescoreResult:
    """Recompute per-trait scores from *log_path* using the fallback parser.

    Samples that cannot be parsed are excluded from the trait mean (NaN, not
    zero).  Parse statistics are always returned alongside the scores so callers
    can surface data quality (e.g. annotate plots with ``N=42/50 parsed``).
    """
    valid_letters = VALID_LETTERS[eval_type]
    data = json.loads(log_path.read_text(encoding="utf-8"))

    trait_scores: dict[str, list[float]] = {}
    n_parsed = 0
    n_total = 0

    for sample in data.get("samples", []):
        md = sample.get("metadata", {})
        trait = md.get("trait")
        if not trait:
            continue
        n_total += 1
        mapping = md.get("answer_mapping", {})
        reverse = md.get("reverse", False)

        orig = sample.get("scores", {}).get("any_choice", {}).get("answer", None)
        raw_out = _raw_output(sample)

        if isinstance(orig, str) and re.match(rf"^[{valid_letters}]$", orig.strip()):
            answer = orig.strip()
        else:
            answer = parse_answer(raw_out, valid_letters)

        if answer is None:
            continue

        sc = _score_answer(answer, mapping, reverse)
        if sc is not None:
            trait_scores.setdefault(trait, []).append(sc)
            n_parsed += 1

    scores = {trait: sum(vals) / len(vals) for trait, vals in trait_scores.items() if vals}
    return RescoreResult(scores=scores, n_parsed=n_parsed, n_total=n_total, raw_scores=trait_scores)

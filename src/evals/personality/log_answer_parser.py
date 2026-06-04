"""Custom fallback answer parser for BFI and TRAIT inspect logs.

The inspect-evals ``any_choice`` scorer only accepts ``ANSWER: X`` format.
Models under LoRA pressure frequently produce alternative formats that are
still unambiguous (e.g. ``D)``, ``C) Neither agree...``, ``The answer is B``).
This module provides a fallback parser to recover those answers, as well as
utilities for loading and re-analysing inspect log files.

Public surface (superset of the earlier rescore-only extract):
- Parsing: ``parse_answer``, ``VALID_LETTERS``.
- Log loading / inspection: ``parse_log``, ``load_logs``, ``ParsedSample``,
  ``LogStats``.
- Rescoring: ``RescoreResult``, ``rescore_log``, ``rescore_log_from_logprobs``.
- Internal helpers retained for callers: ``_raw_output``, ``_score_answer``.

Note: Model coherence / collapse detection is handled at the eval level via
MMLU rather than in this parser.  Samples that cannot be parsed are simply
excluded from the trait mean; parse rate is reported alongside scores so the
caller can decide how to treat low-parse-rate runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
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


@dataclass
class ParsedSample:
    orig_answer: str | None
    fallback_answer: str | None
    raw_output: str
    # True when neither the original scorer nor the fallback could parse an answer.
    failed: bool = False

    @property
    def answer(self) -> str | None:
        """Best available answer (original scorer > fallback > None)."""
        return self.orig_answer or self.fallback_answer

    @property
    def is_valid(self) -> bool:
        return self.answer is not None


@dataclass
class LogStats:
    model: str
    eval_type: str  # 'bfi' or 'trait'
    valid_letters: str
    counts: Counter = field(default_factory=Counter)  # letter -> count (valid answers)
    n_fail: int = 0

    @property
    def n_valid(self) -> int:
        return sum(self.counts.values())

    @property
    def n_total(self) -> int:
        return self.n_valid + self.n_fail

    def pct(self, letter: str) -> float:
        if self.n_valid == 0:
            return float("nan")
        return 100.0 * self.counts[letter] / self.n_valid


def parse_log(log_path: Path, eval_type: str) -> LogStats:
    """Parse a single inspect log JSON and return a :class:`LogStats`."""
    valid_letters = VALID_LETTERS[eval_type]
    # Convention: .../fetched_logs/<suite>/<eval_type>/<run>/<model>/<eval>/...
    # model label sits at index -7 relative to the log file
    try:
        model = log_path.relative_to(
            next(p for p in log_path.parents if "fetched_logs" in p.name or p.name == "fetched_logs")
        ).parts[3]
    except (StopIteration, IndexError):
        model = log_path.parts[-7] if len(log_path.parts) >= 7 else "unknown"

    stats = LogStats(model=model, eval_type=eval_type, valid_letters=valid_letters)
    data = json.loads(log_path.read_text(encoding="utf-8"))

    for sample in data.get("samples", []):
        orig = sample.get("scores", {}).get("any_choice", {}).get("answer", None)
        raw = _raw_output(sample)

        if isinstance(orig, str) and re.match(rf"^[{valid_letters}]$", orig.strip()):
            stats.counts[orig.strip()] += 1
        else:
            fallback = parse_answer(raw, valid_letters)
            if fallback:
                stats.counts[fallback] += 1
            else:
                stats.n_fail += 1

    return stats


def load_logs(
    log_dir: Path,
    eval_type: str,
) -> list[LogStats]:
    """Load all inspect logs of *eval_type* under *log_dir*."""
    pattern = f"**/{eval_type}/native/inspect_logs/*.json"
    return [parse_log(p, eval_type) for p in sorted(log_dir.glob(pattern))]


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


# ---------------------------------------------------------------------------
# Logprob-based rescoring
# ---------------------------------------------------------------------------

def _raw_logprobs_from_sample(sample: dict) -> dict[str, float] | None:
    """Extract raw choice logprobs from a logprob-scored sample's score metadata."""
    for ev in sample.get("events", []):
        if ev.get("event") != "score":
            continue
        score_meta = ev.get("score", {}).get("metadata", {})
        lps = score_meta.get("logprobs")
        if isinstance(lps, dict) and lps:
            return {k: float(v) for k, v in lps.items()}
        break
    return None


def rescore_log_from_logprobs(
    log_path: Path,
    normalization: str = "softmax",
) -> RescoreResult:
    """Recompute per-trait scores from stored logprobs with a different normalization.

    This enables re-analysis of logprob-based TRAIT runs without re-running
    inference.  The raw logprobs are stored in the Inspect log's score metadata
    by ``logprob_trait_scorer``.

    Args:
        log_path: Path to the Inspect log JSON.
        normalization: How to convert raw logprobs to probabilities.

            - ``"softmax"`` (default): standard softmax over available choices.
            - ``"raw"``: use exp(logprob) directly (un-normalized).

    Returns:
        RescoreResult with per-trait mean scores and raw per-sample scores.
    """
    import math

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
        if not mapping:
            continue

        lps = _raw_logprobs_from_sample(sample)
        if not lps:
            continue

        # Convert logprobs to probabilities using the requested normalization.
        if normalization == "softmax":
            max_lp = max(lps.values())
            exp_vals = {k: math.exp(v - max_lp) for k, v in lps.items()}
            total = sum(exp_vals.values())
            probs = {k: v / total for k, v in exp_vals.items()}
        elif normalization == "raw":
            probs = {k: math.exp(v) for k, v in lps.items()}
        else:
            raise ValueError(f"Unknown normalization '{normalization}'. Use 'softmax' or 'raw'.")

        # Compute trait score from probabilities and answer mapping.
        max_val = max(mapping.values()) if mapping else 1
        reverse = md.get("reverse", False)
        score = 0.0
        for letter, prob in probs.items():
            if letter in mapping:
                raw = mapping[letter]
                if reverse:
                    raw = max_val + 1 - raw
                score += prob * raw
        if max_val > 0:
            score /= max_val

        trait_scores.setdefault(trait, []).append(score)
        n_parsed += 1

    scores = {trait: sum(vals) / len(vals) for trait, vals in trait_scores.items() if vals}
    return RescoreResult(scores=scores, n_parsed=n_parsed, n_total=n_total, raw_scores=trait_scores)

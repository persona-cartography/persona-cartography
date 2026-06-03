"""Unit tests for src/evals/personality/log_answer_parser.py.

Covers the fallback ``parse_answer`` across answer formats, ``_score_answer``,
``_raw_output`` extraction, and ``rescore_log`` end-to-end on a synthetic
inspect-log JSON fixture. A dev-parity test proves the migrated symbols are
behaviourally identical to the dev-layer source
(``src_dev/evals/personality/log_answer_parser.py``).
"""

from __future__ import annotations

import importlib
import json

import pytest

from src.evals.personality.log_answer_parser import (
    RescoreResult,
    _raw_output,
    _score_answer,
    parse_answer,
    rescore_log,
)


# ---------------------------------------------------------------------------
# parse_answer
# ---------------------------------------------------------------------------


def test_parse_answer_explicit_answer_prefix():
    assert parse_answer("ANSWER: B", "ABCD") == "B"
    assert parse_answer("answer: c", "ABCD") == "C"


def test_parse_answer_paren_at_start():
    assert parse_answer("D)", "ABCD") == "D"
    assert parse_answer("C) Neither agree nor disagree", "ABCD") == "C"


def test_parse_answer_bare_letter_then_newline():
    assert parse_answer("E\n\nI am not sure", "ABCDE") == "E"


def test_parse_answer_the_answer_is():
    assert parse_answer("The answer is B", "ABCD") == "B"
    assert parse_answer("The correct answer is A.", "ABCD") == "A"


def test_parse_answer_x_is_the_correct_answer():
    assert parse_answer("I think B) is the correct answer", "ABCD") == "B"


def test_parse_answer_no_match_returns_none():
    assert parse_answer("I have no idea here", "ABCD") is None


def test_parse_answer_empty_or_non_string_returns_none():
    assert parse_answer("", "ABCD") is None
    assert parse_answer("   ", "ABCD") is None
    assert parse_answer(None, "ABCD") is None  # type: ignore[arg-type]


def test_parse_answer_letter_outside_valid_set_returns_none():
    # "E" is parsed by the regex but rejected because not in trait valid set.
    assert parse_answer("E)", "ABCD") is None


# ---------------------------------------------------------------------------
# _score_answer
# ---------------------------------------------------------------------------


def test_score_answer_forward():
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
    assert _score_answer("D", mapping, reverse=False) == 1.0
    assert _score_answer("A", mapping, reverse=False) == 0.25


def test_score_answer_reverse():
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
    # reverse: raw = max_val + 1 - raw = 4 + 1 - 1 = 4 -> 4/4 = 1.0
    assert _score_answer("A", mapping, reverse=True) == 1.0
    assert _score_answer("D", mapping, reverse=True) == 0.25


def test_score_answer_not_in_mapping_returns_none():
    assert _score_answer("Z", {"A": 1, "B": 2}, reverse=False) is None


# ---------------------------------------------------------------------------
# _raw_output
# ---------------------------------------------------------------------------


def test_raw_output_string_content():
    sample = {"output": {"choices": [{"message": {"content": "ANSWER: B"}}]}}
    assert _raw_output(sample) == "ANSWER: B"


def test_raw_output_list_content():
    sample = {
        "output": {
            "choices": [
                {"message": {"content": [{"text": "ANSWER:"}, {"text": "B"}]}}
            ]
        }
    }
    assert _raw_output(sample) == "ANSWER: B"


def test_raw_output_missing_choices():
    assert _raw_output({}) == ""
    assert _raw_output({"output": {"choices": []}}) == ""


# ---------------------------------------------------------------------------
# rescore_log on a synthetic inspect-log fixture
# ---------------------------------------------------------------------------


def _synthetic_log() -> dict:
    """Small inspect-log structure that rescore_log knows how to read."""
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
    return {
        "samples": [
            # Original scorer parsed answer "D" -> forward 4/4 = 1.0
            {
                "metadata": {"trait": "openness", "answer_mapping": mapping, "reverse": False},
                "scores": {"any_choice": {"answer": "D"}},
                "output": {"choices": [{"message": {"content": "ANSWER: D"}}]},
            },
            # No original answer; fallback parses "B)" -> forward 2/4 = 0.5
            {
                "metadata": {"trait": "openness", "answer_mapping": mapping, "reverse": False},
                "scores": {"any_choice": {"answer": None}},
                "output": {"choices": [{"message": {"content": "B) somewhat"}}]},
            },
            # Unparseable -> excluded from mean, counted in n_total only.
            {
                "metadata": {"trait": "openness", "answer_mapping": mapping, "reverse": False},
                "scores": {"any_choice": {"answer": None}},
                "output": {"choices": [{"message": {"content": "no clue"}}]},
            },
            # Different trait, reverse scoring: answer "A" reverse -> 1.0
            {
                "metadata": {"trait": "neuroticism", "answer_mapping": mapping, "reverse": True},
                "scores": {"any_choice": {"answer": "A"}},
                "output": {"choices": [{"message": {"content": "ANSWER: A"}}]},
            },
            # No trait metadata -> skipped entirely (not even counted).
            {
                "metadata": {},
                "scores": {"any_choice": {"answer": "A"}},
                "output": {"choices": [{"message": {"content": "ANSWER: A"}}]},
            },
        ]
    }


def test_rescore_log_basic(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(_synthetic_log()), encoding="utf-8")

    result = rescore_log(log_path, "trait")
    assert isinstance(result, RescoreResult)

    # openness: mean of [1.0, 0.5] = 0.75; neuroticism: 1.0
    assert result.scores["openness"] == pytest.approx(0.75)
    assert result.scores["neuroticism"] == pytest.approx(1.0)

    # n_total counts the 4 trait-bearing samples (the no-trait one is skipped).
    assert result.n_total == 4
    # n_parsed: D, B, A -> 3 (the "no clue" sample fails).
    assert result.n_parsed == 3
    assert result.parse_rate == pytest.approx(3 / 4)
    assert result.raw_scores["openness"] == [1.0, 0.5]


def test_rescore_log_empty_total_parse_rate_nan(tmp_path):
    log_path = tmp_path / "empty.json"
    log_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
    result = rescore_log(log_path, "trait")
    assert result.n_total == 0
    assert result.scores == {}
    import math

    assert math.isnan(result.parse_rate)


# ---------------------------------------------------------------------------
# DEV-PARITY: migrated parser must match the dev-layer source
# ---------------------------------------------------------------------------


def test_parity_with_dev_implementation(tmp_path):
    dev = importlib.import_module("src_dev.evals.personality.log_answer_parser")

    raw_inputs = [
        "ANSWER: B",
        "answer: c",
        "D)",
        "C) Neither agree nor disagree",
        "E\n\nI am not sure",
        "The answer is B",
        "The correct answer is A.",
        "I think B) is the correct answer",
        "no idea",
        "",
        "   ",
        "E)",
    ]
    for raw in raw_inputs:
        for valid in ("ABCD", "ABCDE"):
            assert parse_answer(raw, valid) == dev.parse_answer(raw, valid)

    mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
    for ans in ("A", "B", "C", "D", "Z"):
        for rev in (True, False):
            assert _score_answer(ans, mapping, rev) == dev._score_answer(ans, mapping, rev)

    samples = [
        {"output": {"choices": [{"message": {"content": "ANSWER: B"}}]}},
        {"output": {"choices": [{"message": {"content": [{"text": "ANSWER:"}, {"text": "B"}]}}]}},
        {},
    ]
    for s in samples:
        assert _raw_output(s) == dev._raw_output(s)

    # rescore_log parity on the same fixture.
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(_synthetic_log()), encoding="utf-8")
    mine = rescore_log(log_path, "trait")
    theirs = dev.rescore_log(log_path, "trait")
    assert mine.scores == theirs.scores
    assert mine.n_parsed == theirs.n_parsed
    assert mine.n_total == theirs.n_total
    assert mine.raw_scores == theirs.raw_scores

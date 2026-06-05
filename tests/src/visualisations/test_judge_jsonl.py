"""Tests for the shared judge-jsonl reducer.

This is the single source of truth all paper figures use to turn a
``{trait}_v2.jsonl`` into a score, so it must (a) exclude failed-judge rows
(the error sentinels) and (b) support median-over-repeats consistently.
"""

import json

import pytest

from src.visualisations.judge_jsonl import judge_scores, mean_judge_score


def _write(tmp_path, rows):
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_excludes_failed_calls(tmp_path):
    # An error row carrying the -99 sentinel must not drag the mean down.
    p = _write(
        tmp_path,
        [
            {"response_id": "a", "score": 4, "status": "success"},
            {"response_id": "b", "score": 2, "status": "parse_error"},
            {"response_id": "c", "score": -99, "status": "error"},  # failed call
        ],
    )
    assert mean_judge_score(p) == pytest.approx(3.0)  # (4 + 2) / 2; -99 excluded


def test_median_over_repeats_vs_plain_mean(tmp_path):
    # response "a" judged 3× → median 4; "b" once → 1. mean(4, 1) = 2.5.
    p = _write(
        tmp_path,
        [
            {"response_id": "a", "score": 2, "status": "success"},
            {"response_id": "a", "score": 4, "status": "success"},
            {"response_id": "a", "score": 5, "status": "success"},
            {"response_id": "b", "score": 1, "status": "success"},
        ],
    )
    assert mean_judge_score(p, median_over_repeats=True) == pytest.approx(2.5)
    assert mean_judge_score(p) == pytest.approx(3.0)  # plain mean over all rows


def test_skips_non_numeric_and_missing_score(tmp_path):
    p = _write(
        tmp_path,
        [
            {"response_id": "a", "score": 3, "status": "success"},
            {"response_id": "b", "score": None, "status": "success"},
            {"response_id": "c", "status": "success"},  # no score field
            {"response_id": "d", "score": "x", "status": "success"},
        ],
    )
    assert mean_judge_score(p) == pytest.approx(3.0)


def test_empty_returns_none(tmp_path):
    p = _write(tmp_path, [{"response_id": "a", "score": 1, "status": "error"}])
    assert mean_judge_score(p) is None


def test_judge_scores_returns_filtered_pairs(tmp_path):
    p = _write(
        tmp_path,
        [
            {"response_id": "a", "score": 4, "status": "success"},
            {"response_id": "b", "score": -99, "status": "error"},
        ],
    )
    assert judge_scores(p) == [("a", 4.0)]

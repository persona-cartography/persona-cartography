"""Unit tests for the TRAIT/MMLU logprob scorer + the choice-mass refusal gate.

Covers the model-free core of ``src.evals.personality.logprob_scorer`` that
produces every TRAIT number in the paper: choice-letter logprob extraction
(across tokenizer variants), softmax-over-found-choices, the expected-trait
score, and the two-level choice-mass filter (dynamic ``1/num_choices`` +
fixed ``0.75``) that implements the paper's "responses with <75% mass on valid
choices are excluded as refusals" rule.

The GPU generate path is not tested here (it needs a model); these are the pure
reductions applied to its logprob output.
"""

import math
from types import SimpleNamespace

import pytest
from inspect_ai.model._model_output import TopLogprob

from src.evals.personality.logprob_scorer import (
    MIN_CHOICE_MASS_DEFAULT,
    _compute_trait_score,
    _find_choice_logprobs,
    _softmax_over_choices,
    logprob_mcq_ratio,
)


def _tl(token: str, logprob: float) -> TopLogprob:
    return TopLogprob(token=token, logprob=logprob)


# ── _find_choice_logprobs ─────────────────────────────────────────────────────


def test_find_choice_logprobs_basic_and_variants():
    top = [_tl("A", -0.1), _tl("▁B", -0.2), _tl("X", -5.0)]
    found = _find_choice_logprobs(top, num_choices=4)
    # C/D absent → omitted; the "▁B" tokenizer variant maps to canonical "B".
    assert found == {"A": -0.1, "B": -0.2}


def test_find_choice_logprobs_first_variant_wins():
    top = [_tl("▁A", -0.3), _tl("A", -9.0)]
    assert _find_choice_logprobs(top, num_choices=1) == {"A": -0.3}


def test_find_choice_logprobs_respects_num_choices():
    top = [_tl("A", -0.1), _tl("E", -0.1)]
    assert _find_choice_logprobs(top, num_choices=4) == {"A": -0.1}  # E outside A–D
    assert set(_find_choice_logprobs(top, num_choices=5)) == {"A", "E"}


# ── _softmax_over_choices ─────────────────────────────────────────────────────


def test_softmax_equal_logprobs():
    assert _softmax_over_choices({"A": 0.0, "B": 0.0}) == pytest.approx(
        {"A": 0.5, "B": 0.5}
    )


def test_softmax_known_ratio_and_sums_to_one():
    out = _softmax_over_choices({"A": math.log(3.0), "B": math.log(1.0)})
    assert out["A"] == pytest.approx(0.75)
    assert out["B"] == pytest.approx(0.25)
    assert sum(out.values()) == pytest.approx(1.0)


def test_softmax_empty():
    assert _softmax_over_choices({}) == {}


# ── _compute_trait_score ──────────────────────────────────────────────────────


def test_compute_trait_score_high_is_A_plus_B():
    mapping = {"A": 1, "B": 1, "C": 0, "D": 0}
    probs = {"A": 0.4, "B": 0.2, "C": 0.3, "D": 0.1}
    assert _compute_trait_score(probs, mapping) == pytest.approx(0.6)  # P(A)+P(B)


def test_compute_trait_score_capability_p_correct():
    mapping = {"A": 0, "B": 1, "C": 0, "D": 0}
    probs = {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}
    assert _compute_trait_score(probs, mapping) == pytest.approx(0.7)


def test_compute_trait_score_normalizes_by_max_val():
    mapping = {"A": 2, "B": 0}  # max_val = 2
    assert _compute_trait_score({"A": 1.0, "B": 0.0}, mapping) == pytest.approx(1.0)


# ── logprob_mcq_ratio: the choice-mass refusal gate ───────────────────────────


def _ss(value, choice_mass, *, num_choices=4, trait="Openness"):
    """Lightweight stand-in for inspect_ai ``SampleScore`` (the metric only
    duck-types ``.score.value``, ``.score.metadata`` and ``.sample_metadata``)."""
    return SimpleNamespace(
        score=SimpleNamespace(
            value=value,
            metadata={"choice_mass": choice_mass, "num_choices": num_choices},
        ),
        sample_metadata={"trait": trait},
    )


def test_default_threshold_matches_paper():
    assert MIN_CHOICE_MASS_DEFAULT == 0.75  # the paper's <75%-mass refusal cutoff


def test_fixed_mass_filter_excludes_refusals():
    metric = logprob_mcq_ratio(
        min_choice_mass=0.75, dynamic_mass_filter=False, group_by="trait"
    )
    out = metric([_ss(1.0, choice_mass=0.9), _ss(0.0, choice_mass=0.5)])
    assert out["Openness"] == pytest.approx(1.0)  # low-mass sample dropped
    assert out["_filtered_fixed_mass"] == 1.0


def test_dynamic_filter_excludes_below_uniform_mass():
    # num_choices=4 → dynamic threshold 1/4 = 0.25; cm=0.2 is below it.
    metric = logprob_mcq_ratio(
        min_choice_mass=0.0, dynamic_mass_filter=True, group_by="trait"
    )
    out = metric([_ss(1.0, choice_mass=0.2, num_choices=4)])
    assert "Openness" not in out
    assert out["_filtered_dynamic_mass"] == 1.0


def test_nan_and_none_scores_counted_missing():
    metric = logprob_mcq_ratio(
        min_choice_mass=0.0, dynamic_mass_filter=False, group_by="trait"
    )
    out = metric([_ss(float("nan"), 1.0), _ss(None, 1.0), _ss(0.5, 1.0)])
    assert out["Openness"] == pytest.approx(0.5)
    assert out["_missing"] == 2.0


def test_groups_means_by_trait():
    metric = logprob_mcq_ratio(
        min_choice_mass=0.0, dynamic_mass_filter=False, group_by="trait"
    )
    out = metric(
        [
            _ss(1.0, 1.0, trait="Openness"),
            _ss(0.0, 1.0, trait="Openness"),
            _ss(0.8, 1.0, trait="Neuroticism"),
        ]
    )
    assert out["Openness"] == pytest.approx(0.5)
    assert out["Neuroticism"] == pytest.approx(0.8)

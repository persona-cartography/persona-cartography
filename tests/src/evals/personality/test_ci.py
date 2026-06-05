"""Unit tests for src/evals/personality/ci.py.

Covers each ``_interval_*`` helper on small synthetic inputs with known
behaviour, ``IntervalMethod.from_str`` round-trips and properties, and a
parity check proving the migrated ``ci.py`` is behaviourally identical to the
dev-layer source (``src_dev/evals/personality/analyze_results.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evals.personality.ci import (
    IntervalMethod,
    _build_mass_mask,
    _interval_ci_from_bootstrap,
    _interval_ci_from_ppf,
    _interval_ci_from_std,
    _interval_ci_from_weighted_bootstrap,
    _interval_ci_from_wilson,
    _interval_std,
    _resolve_interval_fn,
)


# ---------------------------------------------------------------------------
# _interval_std
# ---------------------------------------------------------------------------


def test_interval_std_known_sample():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # ddof=1 std of [1..5] is sqrt(2.5)
    assert np.isclose(_interval_std(values), np.sqrt(2.5))


def test_interval_std_single_value_is_zero():
    assert _interval_std(np.array([3.0])) == 0.0


def test_interval_std_empty_is_zero():
    assert _interval_std(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# _interval_ci_from_std (normal approximation)
# ---------------------------------------------------------------------------


def test_interval_ci_from_std_known():
    from scipy import stats

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    confidence = 95.0
    n = len(values)
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    expected = z * values.std(ddof=1) / np.sqrt(n)
    assert np.isclose(_interval_ci_from_std(values, confidence), expected)


def test_interval_ci_from_std_single_value_is_zero():
    assert _interval_ci_from_std(np.array([2.0]), 95.0) == 0.0


# ---------------------------------------------------------------------------
# _interval_ci_from_ppf (Student's t)
# ---------------------------------------------------------------------------


def test_interval_ci_from_ppf_known():
    from scipy import stats

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    confidence = 95.0
    n = len(values)
    alpha = 1 - confidence / 100
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    expected = t_val * values.std(ddof=1) / np.sqrt(n)
    assert np.isclose(_interval_ci_from_ppf(values, confidence), expected)


def test_interval_ci_from_ppf_wider_than_normal():
    # t-distribution intervals are wider than normal for finite n.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _interval_ci_from_ppf(values, 95.0) > _interval_ci_from_std(values, 95.0)


def test_interval_ci_from_ppf_single_value_is_zero():
    assert _interval_ci_from_ppf(np.array([2.0]), 95.0) == 0.0


# ---------------------------------------------------------------------------
# _interval_ci_from_wilson (binary)
# ---------------------------------------------------------------------------


def test_interval_ci_from_wilson_known():
    # 6 successes out of 10, 95% Wilson interval. Compute the closed form
    # independently to validate.
    from scipy import stats

    values = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    confidence = 95.0
    n = len(values)
    p_hat = values.mean()
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    expected = (centre - margin, centre + margin)

    lo, hi = _interval_ci_from_wilson(values, confidence)
    assert np.isclose(lo, expected[0])
    assert np.isclose(hi, expected[1])
    # Wilson bounds stay within [0, 1] for a proportion.
    assert 0.0 <= lo <= hi <= 1.0


def test_interval_ci_from_wilson_all_ones():
    # p_hat = 1: upper bound below 1, lower bound strictly positive.
    values = np.ones(10, dtype=float)
    lo, hi = _interval_ci_from_wilson(values, 95.0)
    assert lo < 1.0
    assert hi <= 1.0
    assert lo > 0.0


def test_interval_ci_from_wilson_rejects_non_binary():
    with pytest.raises(ValueError, match="binary"):
        _interval_ci_from_wilson(np.array([0.0, 0.5, 1.0]), 95.0)


def test_interval_ci_from_wilson_empty():
    assert _interval_ci_from_wilson(np.array([]), 95.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _interval_ci_from_bootstrap (BCa)
# ---------------------------------------------------------------------------


def test_interval_ci_from_bootstrap_deterministic_under_seed():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=50)
    a = _interval_ci_from_bootstrap(values, 95.0, n_resamples=200, seed=42)
    b = _interval_ci_from_bootstrap(values, 95.0, n_resamples=200, seed=42)
    assert a == b


def test_interval_ci_from_bootstrap_seed_changes_result():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=50)
    a = _interval_ci_from_bootstrap(values, 95.0, n_resamples=200, seed=1)
    b = _interval_ci_from_bootstrap(values, 95.0, n_resamples=200, seed=2)
    assert a != b


def test_interval_ci_from_bootstrap_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=200)
    lo, hi = _interval_ci_from_bootstrap(values, 95.0, n_resamples=500, seed=42)
    assert lo <= values.mean() <= hi


def test_interval_ci_from_bootstrap_degenerate_zero_width():
    values = np.full(10, 0.7)
    result = _interval_ci_from_bootstrap(values, 95.0, n_resamples=100, seed=42)
    assert result == (0.7, 0.7)


def test_interval_ci_from_bootstrap_single_value():
    result = _interval_ci_from_bootstrap(
        np.array([0.5]), 95.0, n_resamples=100, seed=42
    )
    assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _interval_ci_from_weighted_bootstrap
# ---------------------------------------------------------------------------


def test_weighted_bootstrap_deterministic_under_seed():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=40)
    weights = rng.uniform(0, 1, size=40)
    a = _interval_ci_from_weighted_bootstrap(values, weights, 95.0, 200, seed=42)
    b = _interval_ci_from_weighted_bootstrap(values, weights, 95.0, 200, seed=42)
    assert a == b


def test_weighted_bootstrap_low_mass_widens_toward_half():
    # With near-zero choice mass, adjusted scores ~ U(0,1), so the interval
    # straddles 0.5 and is wide despite identical point scores.
    values = np.full(60, 0.9)
    weights = np.full(60, 0.01)
    lo, hi = _interval_ci_from_weighted_bootstrap(values, weights, 95.0, 1000, seed=42)
    assert lo < 0.5 < hi
    # Much wider than the high-mass case (which collapses to < 0.05).
    hi_mass_lo, hi_mass_hi = _interval_ci_from_weighted_bootstrap(
        values, np.full(60, 0.999), 95.0, 1000, seed=42
    )
    assert (hi - lo) > (hi_mass_hi - hi_mass_lo) * 2


def test_weighted_bootstrap_high_mass_near_score():
    # With choice mass ~ 1, noise ~ 0, interval collapses around the score.
    values = np.full(60, 0.9)
    weights = np.full(60, 0.999)
    lo, hi = _interval_ci_from_weighted_bootstrap(values, weights, 95.0, 1000, seed=42)
    assert abs((lo + hi) / 2 - 0.9) < 0.05
    assert (hi - lo) < 0.05


def test_weighted_bootstrap_single_value():
    assert _interval_ci_from_weighted_bootstrap(
        np.array([0.5]), np.array([0.5]), 95.0, 100, seed=42
    ) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _resolve_interval_fn
# ---------------------------------------------------------------------------


def test_resolve_interval_fn_std():
    fn = _resolve_interval_fn(IntervalMethod(method="std"))
    assert fn(np.array([1.0, 2.0, 3.0])) == _interval_std(np.array([1.0, 2.0, 3.0]))


def test_resolve_interval_fn_wilson():
    fn = _resolve_interval_fn(IntervalMethod(method="ci_from_wilson", confidence=95.0))
    values = np.array([1, 1, 0, 0], dtype=float)
    assert fn(values) == _interval_ci_from_wilson(values, 95.0)


def test_resolve_interval_fn_bootstrap_seeded():
    method = IntervalMethod(
        method="ci_from_bootstrap", confidence=95.0, n_resamples=100, seed=7
    )
    fn = _resolve_interval_fn(method)
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=30)
    assert fn(values) == _interval_ci_from_bootstrap(values, 95.0, 100, 7)


# ---------------------------------------------------------------------------
# _build_mass_mask
# ---------------------------------------------------------------------------


def test_build_mass_mask_fixed_threshold():
    cm = np.array([0.1, 0.5, 0.9])
    mask = _build_mass_mask(cm, None, min_choice_mass=0.4, dynamic_mass_filter=False)
    assert mask.tolist() == [False, True, True]


def test_build_mass_mask_dynamic_threshold():
    cm = np.array([0.2, 0.6])
    nc = np.array([4.0, 2.0])  # thresholds 0.25, 0.5
    mask = _build_mass_mask(cm, nc, min_choice_mass=0.0, dynamic_mass_filter=True)
    assert mask.tolist() == [False, True]


def test_build_mass_mask_no_filters_keeps_all():
    cm = np.array([0.0, 0.1, 1.0])
    mask = _build_mass_mask(cm, None, min_choice_mass=0.0, dynamic_mass_filter=False)
    assert mask.all()


# ---------------------------------------------------------------------------
# IntervalMethod.from_str round-trips + properties
# ---------------------------------------------------------------------------


def test_from_str_std():
    m = IntervalMethod.from_str("std")
    assert m.method == "std"
    assert m.label == "±1 SD"


def test_from_str_bare_ci_alias_removed():
    # Bare "ci95" used to silently alias to the symmetric ci_from_ppf — a
    # footgun (it crashes on logprob evals under choice-mass filtering, and
    # masks the data-type/method mismatch). The method must now be explicit.
    with pytest.raises(ValueError, match="Cannot parse"):
        IntervalMethod.from_str("ci95")


def test_from_str_wilson():
    m = IntervalMethod.from_str("ci95_from_wilson")
    assert m.method == "ci_from_wilson"
    assert m.confidence == 95.0
    assert m.needs_raw_scores is True
    assert m.needs_weights is False
    assert m.label == "95% CI (Wilson)"


def test_from_str_ppf():
    m = IntervalMethod.from_str("ci99.5_from_ppf")
    assert m.method == "ci_from_ppf"
    assert m.confidence == 99.5
    assert m.needs_raw_scores is False
    assert m.label == "99.5% CI (t)"


def test_from_str_bootstrap():
    m = IntervalMethod.from_str("ci95_from_bootstrap_1000")
    assert m.method == "ci_from_bootstrap"
    assert m.confidence == 95.0
    assert m.n_resamples == 1000
    assert m.needs_raw_scores is True
    assert m.needs_weights is False
    assert m.label == "95% CI (bootstrap, 1000)"


def test_from_str_weighted_bootstrap():
    m = IntervalMethod.from_str("ci95_from_weighted_bootstrap_500")
    assert m.method == "ci_from_weighted_bootstrap"
    assert m.confidence == 95.0
    assert m.n_resamples == 500
    assert m.needs_raw_scores is True
    assert m.needs_weights is True
    assert m.label == "95% CI (mass-weighted bootstrap, 500)"


def test_from_str_invalid_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        IntervalMethod.from_str("nonsense")


def test_from_str_std_label_and_ci_from_std_label():
    assert IntervalMethod.from_str("ci90_from_std").label == "90% CI (normal)"


def test_confidence_as_fraction_raises():
    with pytest.raises(ValueError, match="as a percentage"):
        IntervalMethod(method="ci_from_ppf", confidence=0.95)


def test_bootstrap_requires_n_resamples():
    with pytest.raises(ValueError, match="n_resamples"):
        IntervalMethod(method="ci_from_bootstrap", confidence=95.0)


# ---------------------------------------------------------------------------
# DEV-PARITY: migrated ci.py must be behaviourally identical to dev source
# ---------------------------------------------------------------------------


def test_parity_with_dev_implementation():
    """Migrated ci.py helpers must match the dev-layer analyze_results.py.

    The dev module imports ``src_dev.evals.personality.logprob_scorer`` at
    import time; importing it via the package path keeps that import resolvable.
    """
    import importlib

    dev = importlib.import_module("src_dev.evals.personality.analyze_results")

    rng = np.random.default_rng(0)
    cont = rng.uniform(0, 1, size=50)
    binary = (rng.uniform(0, 1, size=50) > 0.5).astype(float)
    weights = rng.uniform(0, 1, size=50)

    # Symmetric scalar methods.
    assert _interval_std(cont) == dev._interval_std(cont)
    for conf in (90.0, 95.0, 99.0):
        assert _interval_ci_from_std(cont, conf) == dev._interval_ci_from_std(
            cont, conf
        )
        assert _interval_ci_from_ppf(cont, conf) == dev._interval_ci_from_ppf(
            cont, conf
        )
        assert _interval_ci_from_wilson(binary, conf) == dev._interval_ci_from_wilson(
            binary, conf
        )

    # Bootstrap methods, across seeds.
    for seed in (1, 42, 7):
        assert _interval_ci_from_bootstrap(
            cont, 95.0, 200, seed
        ) == dev._interval_ci_from_bootstrap(cont, 95.0, 200, seed)
        assert _interval_ci_from_weighted_bootstrap(
            cont, weights, 95.0, 200, seed
        ) == dev._interval_ci_from_weighted_bootstrap(cont, weights, 95.0, 200, seed)

    # IntervalMethod parsing + properties parity. (Bare "ci95" is intentionally
    # excluded: src dropped the misleading ci95→ci_from_ppf alias, so it now
    # diverges from the frozen dev copy — see D36.)
    for s in (
        "std",
        "ci95_from_wilson",
        "ci99.5_from_ppf",
        "ci95_from_bootstrap_1000",
        "ci95_from_weighted_bootstrap_500",
    ):
        mine = IntervalMethod.from_str(s)
        theirs = dev.IntervalMethod.from_str(s)
        assert mine.method == theirs.method
        assert mine.confidence == theirs.confidence
        assert mine.n_resamples == theirs.n_resamples
        assert mine.needs_raw_scores == theirs.needs_raw_scores
        assert mine.needs_weights == theirs.needs_weights
        assert mine.label == theirs.label

    # _build_mass_mask parity.
    cm = rng.uniform(0, 1, size=20)
    nc = rng.integers(2, 6, size=20).astype(float)
    np.testing.assert_array_equal(
        _build_mass_mask(cm, nc, 0.2, True),
        dev._build_mass_mask(cm, nc, 0.2, True),
    )

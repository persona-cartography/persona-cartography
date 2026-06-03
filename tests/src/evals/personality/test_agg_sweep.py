"""Tests for src/evals/personality/ci.py::_agg_sweep.

Builds a small synthetic sweep DataFrame (scale + metric columns + raw
per-sample / choice-mass list columns) and asserts the migrated ``_agg_sweep``
produces output identical to the dev-layer source
(``src_dev/evals/personality/analyze_results.py``) across several
``IntervalMethod`` choices and choice-mass filtering settings.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.evals.personality.ci import IntervalMethod, _agg_sweep


def _synthetic_sweep() -> pd.DataFrame:
    """A sweep with two scale points, two runs each, raw + choice-mass cols."""
    rng = np.random.default_rng(0)

    def rows_for(scale: float):
        out = []
        for _ in range(2):
            raw = rng.uniform(0, 1, size=8)
            binary = (rng.uniform(0, 1, size=8) > 0.5).astype(float)
            cm = rng.uniform(0.3, 1.0, size=8)
            nc = np.full(8, 4.0)
            out.append(
                {
                    "scale": scale,
                    "trait_x": float(raw.mean()),
                    "trait_b": float(binary.mean()),
                    "_raw_trait_x": raw,
                    "_raw__cm_trait_x": cm,
                    "_raw__nc_trait_x": nc,
                    "_raw_trait_b": binary,
                    "_raw__cm_trait_b": cm,
                    "_raw__nc_trait_b": nc,
                }
            )
        return out

    return pd.DataFrame(rows_for(0.5) + rows_for(1.0))


@pytest.fixture(scope="module")
def dev_mod():
    return importlib.import_module("src_dev.evals.personality.analyze_results")


@pytest.mark.parametrize(
    "interval",
    [
        None,
        IntervalMethod(method="std"),
        IntervalMethod(method="ci_from_ppf", confidence=95.0),
        IntervalMethod(method="ci_from_wilson", confidence=95.0),
        IntervalMethod(method="ci_from_bootstrap", confidence=95.0, n_resamples=200, seed=42),
        IntervalMethod(
            method="ci_from_weighted_bootstrap", confidence=95.0, n_resamples=200, seed=42
        ),
    ],
)
@pytest.mark.parametrize("min_choice_mass", [0.0, 0.75])
@pytest.mark.parametrize("dynamic", [False, True])
def test_agg_sweep_matches_dev(dev_mod, interval, min_choice_mass, dynamic):
    df = _synthetic_sweep()

    # Wilson requires binary-only columns.
    cols = ["trait_b"] if (interval and interval.method == "ci_from_wilson") else ["trait_x"]

    dev_interval = None
    if interval is not None:
        dev_interval = dev_mod.IntervalMethod(
            method=interval.method,
            confidence=interval.confidence,
            n_resamples=interval.n_resamples,
            seed=interval.seed,
        )

    # NOTE: the dev source has a pre-existing latent bug — a symmetric interval
    # method (std / ppf) combined with choice-mass filtering reaches an
    # ``interval_fn(vals)`` branch where ``vals`` was never bound, raising
    # UnboundLocalError. The migration copies this verbatim, so the migrated
    # code must reproduce the *same* exception. Assert parity-on-exception for
    # these combos rather than comparing frames.
    symmetric = interval is not None and interval.method in ("std", "ci_from_ppf", "ci_from_std")
    filtering = min_choice_mass > 0.0 or dynamic
    if symmetric and filtering:
        with pytest.raises(UnboundLocalError):
            _agg_sweep(
                df.copy(),
                cols,
                interval=interval,
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic,
            )
        with pytest.raises(UnboundLocalError):
            dev_mod._agg_sweep(
                df.copy(),
                cols,
                interval=dev_interval,
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic,
            )
        return

    mine = _agg_sweep(
        df.copy(),
        cols,
        interval=interval,
        min_choice_mass=min_choice_mass,
        dynamic_mass_filter=dynamic,
    )
    theirs = dev_mod._agg_sweep(
        df.copy(),
        cols,
        interval=dev_interval,
        min_choice_mass=min_choice_mass,
        dynamic_mass_filter=dynamic,
    )

    pdt.assert_frame_equal(mine, theirs)


def test_agg_sweep_default_min_choice_mass_matches_dev(dev_mod):
    """Default arg (MIN_CHOICE_MASS_DEFAULT) parity, no interval."""
    df = _synthetic_sweep()
    mine = _agg_sweep(df.copy(), ["trait_x"])
    theirs = dev_mod._agg_sweep(df.copy(), ["trait_x"])
    pdt.assert_frame_equal(mine, theirs)


def test_agg_sweep_missing_column(dev_mod):
    df = _synthetic_sweep()
    interval = IntervalMethod(method="ci_from_ppf", confidence=95.0)
    dev_interval = dev_mod.IntervalMethod(method="ci_from_ppf", confidence=95.0)
    mine = _agg_sweep(df.copy(), ["does_not_exist"], interval=interval)
    theirs = dev_mod._agg_sweep(df.copy(), ["does_not_exist"], interval=dev_interval)
    pdt.assert_frame_equal(mine, theirs)

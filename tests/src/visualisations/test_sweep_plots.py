"""Smoke tests for the migrated sweep / capability plotters.

These are smoke tests (not pixel-parity): each plot function is invoked over
mock data with the matplotlib Agg backend, asserting it returns without error
and writes a non-empty image file. The dispatch entrypoint ``generate_plots``
is exercised over the full mock SweepData via the plot registry.
"""

# Force a headless backend before any pyplot import.
import os

os.environ.setdefault("MPLBACKEND", "Agg")

# Third-party
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Local
from src.evals.personality.ci import IntervalMethod  # noqa: E402
from src.evals.personality.sweep_results import SweepData  # noqa: E402
from src.visualisations.capability_plots import (  # noqa: E402
    plot_capability_breakdown,
    plot_capability_sweep,
)
from src.visualisations.sweep_plots import (  # noqa: E402
    _mock_sweep_data,
    generate_plots,
    plot_bfi_sweep,
    plot_generic_sweep,
    plot_parse_rate,
    plot_trait_sweep,
)


def _assert_nonempty_file(path):
    """Assert *path* exists and is a non-empty file."""
    assert path is not None
    assert path.exists(), f"expected output file at {path}"
    assert path.stat().st_size > 0, f"output file {path} is empty"


@pytest.fixture
def mock_data() -> SweepData:
    return _mock_sweep_data()


def _capability_breakdown_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame with the raw columns the breakdown needs.

    ``plot_capability_breakdown`` requires ``_raw_accuracy`` and
    ``_raw__answer_parsed`` (optionally ``_raw__reparsed_accuracy``); each cell
    holds a per-sample list of 0/1 values.
    """
    rng = np.random.default_rng(0)
    rows = []
    for scale in [-1.0, 0.0, 1.0]:
        for run in ["run_1", "run_2"]:
            acc = rng.integers(0, 2, size=8).astype(float).tolist()
            parsed = rng.integers(0, 2, size=8).astype(float).tolist()
            reparsed = rng.integers(0, 2, size=8).astype(float).tolist()
            rows.append({
                "model": "base" if scale == 0.0 else f"lora_{scale:+.2f}x",
                "run": run,
                "scale": scale,
                "accuracy": float(np.mean(acc)),
                "_raw_accuracy": acc,
                "_raw__answer_parsed": parsed,
                "_raw__reparsed_accuracy": reparsed,
            })
    return pd.DataFrame(rows)


def test_plot_trait_sweep(mock_data, tmp_path):
    out = plot_trait_sweep(mock_data.get("trait"), tmp_path)
    _assert_nonempty_file(out)


def test_plot_trait_sweep_with_interval(mock_data, tmp_path):
    out = plot_trait_sweep(
        mock_data.get("trait"), tmp_path,
        interval=IntervalMethod.from_str("ci95"),
        highlight=["O", "N"],
    )
    _assert_nonempty_file(out)


def test_plot_bfi_sweep(mock_data, tmp_path):
    out = plot_bfi_sweep(mock_data.get("bfi"), tmp_path)
    _assert_nonempty_file(out)


def test_plot_bfi_sweep_with_interval(mock_data, tmp_path):
    # Symmetric CI path (has_sym_ci branch). Note: the asymmetric-CI path
    # (e.g. ci95_from_wilson) on this mock data hits a pre-existing source
    # quirk where a NaN reaches set_ylim — that behaviour is identical in the
    # original analyze_results.plot_bfi_sweep, so it is not exercised here.
    out = plot_bfi_sweep(
        mock_data.get("bfi"), tmp_path,
        interval=IntervalMethod.from_str("ci95"),
    )
    _assert_nonempty_file(out)


def test_plot_generic_sweep(mock_data, tmp_path):
    out = plot_generic_sweep(mock_data.get("mmlu"), "mmlu", tmp_path)
    _assert_nonempty_file(out)


def test_plot_parse_rate(mock_data, tmp_path):
    # Mock data degrades parse rate at extreme scales, so a figure is produced.
    out = plot_parse_rate(mock_data.get("trait"), "trait", tmp_path)
    _assert_nonempty_file(out)


def test_plot_parse_rate_perfect_returns_none(tmp_path):
    # All parse rates == 1.0 → no figure, returns None.
    df = pd.DataFrame([
        {"model": "base", "run": "run_1", "scale": 0.0, "_parse_rate": 1.0, "accuracy": 0.6},
        {"model": "lora", "run": "run_1", "scale": 1.0, "_parse_rate": 1.0, "accuracy": 0.5},
    ])
    assert plot_parse_rate(df, "trait", tmp_path) is None


def test_plot_capability_sweep(mock_data, tmp_path):
    out = plot_capability_sweep(mock_data.get("mmlu"), tmp_path, eval_name="mmlu")
    _assert_nonempty_file(out)


def test_plot_capability_sweep_with_random_baseline(mock_data, tmp_path):
    out = plot_capability_sweep(
        mock_data.get("mmlu"), tmp_path, eval_name="mmlu",
        random_baseline=0.25, interval=IntervalMethod.from_str("ci95"),
    )
    _assert_nonempty_file(out)


def test_plot_capability_breakdown(tmp_path):
    out = plot_capability_breakdown(_capability_breakdown_df(), tmp_path, eval_name="mmlu")
    _assert_nonempty_file(out)


def test_plot_capability_breakdown_missing_columns_returns_none(mock_data, tmp_path):
    # mmlu mock df has no _raw_* columns → breakdown skips and returns None.
    assert plot_capability_breakdown(mock_data.get("mmlu"), tmp_path, eval_name="mmlu") is None


def test_generate_plots_dispatch(mock_data, tmp_path):
    # Exercises the full registry dispatch (trait, bfi, capability).
    saved = generate_plots(mock_data, tmp_path, show_parse_rate=True)
    assert saved, "generate_plots produced no figures"
    for path in saved:
        _assert_nonempty_file(path)
    names = {p.name for p in saved}
    assert "trait_sweep.png" in names
    assert "bfi_sweep.png" in names
    assert "mmlu_sweep.png" in names


def test_generate_plots_with_interval_string(mock_data, tmp_path):
    saved = generate_plots(mock_data, tmp_path, interval="ci95", random_baseline=0.25)
    assert saved
    for path in saved:
        _assert_nonempty_file(path)

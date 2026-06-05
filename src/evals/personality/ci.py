"""Confidence-interval helpers for personality / capability sweep analysis.

This module provides the ``IntervalMethod`` specification and the family of
``_interval_*`` functions used to compute error bars / uncertainty intervals
for eval metrics. It was extracted verbatim (D9 split) from
``src_dev/evals/personality/analyze_results.py`` — function and class bodies
are byte-for-byte identical; only the surrounding module context changed.

Which method suits which data type (project convention):

* **Binary data** (MCQ accuracy, 0/1 trait scores) → ``ci_from_wilson``. The
  Wilson score interval has correct coverage near 0 and 1, where normal and
  bootstrap approximations degrade.
* **Continuous data** (LLM-judge scores, rollout metrics) → ``ci_from_bootstrap``
  (BCa bootstrap). Makes no distributional assumptions.
* **Logprob choice-mass scores** → ``ci_from_weighted_bootstrap``. Injects
  per-sample noise proportional to ``(1 - choice_mass)`` to recover honest
  measurement uncertainty for softmax-renormalized MCQ scores.

Avoid the naive ``1.96 * std / sqrt(n)`` normal approximation: it assumes
normality and yields symmetric intervals that can extend below 0 or above 1
for proportions.

All ``_interval_ci_from_*`` helpers return ``(ci_lower, ci_upper)`` as absolute
bounds, except the symmetric ``_interval_std`` / ``_interval_ci_from_std`` /
``_interval_ci_from_ppf`` which return a scalar half-width.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT

_DEFAULT_SEED = 42

_INTERVAL_METHODS = Literal[
    "std",
    "ci_from_std",
    "ci_from_ppf",
    "ci_from_wilson",
    "ci_from_bootstrap",
    "ci_from_weighted_bootstrap",
]


@dataclass(frozen=True)
class IntervalMethod:
    """Specification for how to compute error bars / uncertainty intervals.

    Args:
        method: One of ``"std"``, ``"ci_from_std"``, ``"ci_from_ppf"``,
            ``"ci_from_wilson"``, ``"ci_from_bootstrap"``.
        confidence: Confidence level in percent, e.g. 95.0. Required for all
            ``ci_*`` methods. Must be in (0, 100).
        n_resamples: Number of bootstrap resamples. Required for
            ``"ci_from_bootstrap"``.
        seed: RNG seed for bootstrap. Defaults to 42.
    """

    method: _INTERVAL_METHODS
    confidence: float | None = None
    n_resamples: int | None = None
    seed: int = _DEFAULT_SEED

    def __post_init__(self) -> None:
        is_ci = self.method.startswith("ci_from_")
        if is_ci:
            if self.confidence is None:
                raise ValueError(f"confidence is required for method {self.method!r}")
            if 0 < self.confidence < 1:
                raise ValueError(
                    f"confidence must be in (0, 100) as a percentage, got {self.confidence}. "
                    f"Did you mean {self.confidence * 100}?"
                )
            if not (0 < self.confidence < 100):
                raise ValueError(
                    f"confidence must be in (0, 100), got {self.confidence}"
                )
        elif self.method == "std":
            if self.confidence is not None:
                raise ValueError("confidence must not be set for method 'std'")
        else:
            raise ValueError(f"Unknown method {self.method!r}")
        if self.method in ("ci_from_bootstrap", "ci_from_weighted_bootstrap"):
            if self.n_resamples is None:
                raise ValueError("n_resamples is required for method {self.method!r}")
            if self.n_resamples < 1:
                raise ValueError(f"n_resamples must be >= 1, got {self.n_resamples}")

    @classmethod
    def from_str(cls, s: str) -> IntervalMethod:
        """Parse a string into an IntervalMethod.

        Accepted formats:
            ``"std"``
            ``"ci95_from_ppf"``
            ``"ci99.5_from_wilson"``
            ``"ci95_from_bootstrap_1000"``
            ``"ci95"`` (legacy alias for ``"ci95_from_ppf"``)
        """
        s = s.strip()
        if s == "std":
            return cls(method="std")

        # Legacy alias: "ci95" → "ci95_from_ppf"
        m = re.fullmatch(r"ci([\d.]+)", s)
        if m:
            return cls(method="ci_from_ppf", confidence=float(m.group(1)))

        # Full format: ci{N}_from_{method} or ci{N}_from_bootstrap_{K}
        m = re.fullmatch(r"ci([\d.]+)_from_weighted_bootstrap_(\d+)", s)
        if m:
            return cls(
                method="ci_from_weighted_bootstrap",
                confidence=float(m.group(1)),
                n_resamples=int(m.group(2)),
            )

        m = re.fullmatch(r"ci([\d.]+)_from_bootstrap_(\d+)", s)
        if m:
            return cls(
                method="ci_from_bootstrap",
                confidence=float(m.group(1)),
                n_resamples=int(m.group(2)),
            )

        m = re.fullmatch(r"ci([\d.]+)_from_(std|ppf|wilson)", s)
        if m:
            return cls(method=f"ci_from_{m.group(2)}", confidence=float(m.group(1)))

        raise ValueError(
            f"Cannot parse interval string {s!r}. "
            "Expected 'std', 'ci95', 'ci95_from_ppf', 'ci95_from_wilson', "
            "'ci95_from_bootstrap_1000', or 'ci95_from_weighted_bootstrap_1000'."
        )

    @property
    def needs_raw_scores(self) -> bool:
        """Whether this method requires raw per-sample scores (``_raw_{col}`` columns)."""
        return self.method in (
            "ci_from_wilson",
            "ci_from_bootstrap",
            "ci_from_weighted_bootstrap",
        )

    @property
    def needs_weights(self) -> bool:
        """Whether this method requires per-sample weights (``_raw__cm_{col}`` columns).

        Weighted methods use per-sample choice mass as importance weights,
        producing wider CIs when the model allocates little probability to
        the target answer tokens.
        """
        return self.method == "ci_from_weighted_bootstrap"

    @property
    def label(self) -> str:
        """Human-readable label for plot legends."""
        if self.method == "std":
            return "±1 SD"
        assert self.confidence is not None
        conf = f"{self.confidence:g}%"
        if self.method == "ci_from_std":
            return f"{conf} CI (normal)"
        if self.method == "ci_from_ppf":
            return f"{conf} CI (t)"
        if self.method == "ci_from_wilson":
            return f"{conf} CI (Wilson)"
        if self.method == "ci_from_bootstrap":
            return f"{conf} CI (bootstrap, {self.n_resamples})"
        if self.method == "ci_from_weighted_bootstrap":
            return f"{conf} CI (mass-weighted bootstrap, {self.n_resamples})"
        return f"{conf} CI"


# ---------------------------------------------------------------------------
# Interval computation functions
# ---------------------------------------------------------------------------


def _interval_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _interval_ci_from_std(values: np.ndarray, confidence: float) -> float:
    """CI half-width using normal approximation: z * std / sqrt(n)."""
    from scipy import stats

    n = len(values)
    if n <= 1:
        return 0.0
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    return float(z * values.std(ddof=1) / np.sqrt(n))


def _interval_ci_from_ppf(values: np.ndarray, confidence: float) -> float:
    """CI half-width using Student's t-distribution."""
    from scipy import stats

    n = len(values)
    if n <= 1:
        return 0.0
    alpha = 1 - confidence / 100
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(t_val * values.std(ddof=1) / np.sqrt(n))


def _interval_ci_from_wilson(
    values: np.ndarray, confidence: float
) -> tuple[float, float]:
    """Wilson score interval for binary (0/1) data.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.

    Raises:
        ValueError: If the data contains values other than 0 and 1.
    """
    from scipy import stats

    unique = np.unique(values)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Wilson interval requires binary (0/1) data, "
            f"got unique values: {unique.tolist()}"
        )
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    p_hat = values.mean()
    z = stats.norm.ppf(1 - (1 - confidence / 100) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (float(centre - margin), float(centre + margin))


def _interval_ci_from_bootstrap(
    values: np.ndarray,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """CI via BCa bootstrap on the mean.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.
    """
    from scipy import stats

    n = len(values)
    if n <= 1:
        return (0.0, 0.0)
    # Degenerate case: all values identical → zero-width CI, skip BCa which
    # would emit DegenerateDataWarning and return NaN.
    if np.ptp(values) == 0.0:
        m = float(values[0])
        return (m, m)
    rng = np.random.default_rng(seed)
    result = stats.bootstrap(
        (values,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence / 100,
        random_state=rng,
        method="BCa",
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _interval_ci_from_weighted_bootstrap(
    values: np.ndarray,
    weights: np.ndarray,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """CI via noise-injection bootstrap using choice mass to model measurement uncertainty.

    Standard bootstrap CIs on logprob-based MCQ scores can be misleadingly
    narrow when the model places most probability on non-answer tokens.  The
    softmax-renormalized score is computed from only the answer-token
    probabilities, discarding information about *how little* of the
    distribution was actually observed.  When all samples have similarly low
    choice mass, per-sample scores are consistent but unreliable — the
    resulting CIs capture sampling uncertainty but miss measurement
    uncertainty entirely.

    This method injects per-sample noise proportional to ``(1 - choice_mass)``
    to recover honest uncertainty estimates.  For each bootstrap iteration, every
    resampled score is replaced by a mixture:

        adjusted_i = cm_i * score_i + (1 - cm_i) * U_i

    where ``cm_i`` is the choice mass for sample *i* and ``U_i ~ Uniform(0, 1)``
    represents maximal ignorance about the trait score for the probability mass
    that did NOT land on answer tokens.  The intuition: the ``cm``-fraction of
    the distribution told us ``score``; for the ``(1-cm)``-fraction we know
    nothing, so we model it as a random draw.

    Effects on confidence intervals:

    * **cm ≈ 1** (model focused on ABCD): noise ≈ 0, CI matches the standard
      unweighted bootstrap.
    * **cm ≈ 0** (model not answering ABCD): adjusted scores ≈ U(0,1), CI
      converges to the maximum-ignorance interval around 0.5.

    The corresponding point estimate uses the expected value of the noise term
    (0.5) in place of the stochastic ``U_i``::

        E[adjusted_i] = cm_i * score_i + (1 - cm_i) * 0.5

    Methodology and supporting references:

    * The noise-injection mechanism is analogous to a *measurement-error
      model* (Carroll et al., 2006) where each observation has known, sample-
      specific error variance.  Here the error variance is governed by the
      complement of the choice mass — a direct, observable reliability
      indicator — rather than requiring a separate error-variance estimate:
          Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M.
          (2006). Measurement Error in Nonlinear Models (2nd ed.). Chapman &
          Hall/CRC.

    * Wang et al. (2024) show that first-token logprob distributions and
      text-generated answers diverge by >60 % for instruction-tuned models,
      undermining the reliability of renormalized ABCD probabilities when the
      model is not "trying" to output a letter:
          "My Answer is C: First-Token Probabilities Do Not Match
           Text Answers in API-Based LLMs"  (arXiv:2402.14499)

    * Huang et al. (2025) model next-token logits as Dirichlet concentration
      parameters and show that softmax normalization destroys evidence-strength
      information.  Choice mass is a lightweight proxy for the total Dirichlet
      concentration (alpha_0) on the answer set:
          "LogU: Accurate LLM Log-Probability Estimation with
           Uncertainty"  (arXiv:2502.00290)

    * The choice of U(0,1) as the ignorance distribution for the unobserved
      mass follows maximum-entropy reasoning: for a trait score known to lie
      in [0, 1] with no further information, the uniform distribution has
      the highest entropy (Jaynes, 2003):
          Jaynes, E. T. (2003). Probability Theory: The Logic of Science.
          Cambridge University Press.

    Args:
        values: Per-sample trait scores (0–1), from softmax-renormalized logprobs.
        weights: Per-sample choice mass (0–1), the fraction of the model's
            probability distribution on answer tokens.
        confidence: Confidence level in percent, e.g. 95.0.
        n_resamples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` as absolute bounds.
    """
    n = len(values)
    if n <= 1:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        cm = weights[idx]
        scores = values[idx]
        noise = rng.uniform(0.0, 1.0, size=n)
        adjusted = cm * scores + (1.0 - cm) * noise
        boot_means[i] = adjusted.mean()

    alpha = (100 - confidence) / 200  # half-alpha for two-sided
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return (float(lo), float(hi))


def _resolve_interval_fn(
    method: IntervalMethod,
) -> Callable[[np.ndarray], float | tuple[float, float]]:
    """Return a callable ``(values) -> result`` from an IntervalMethod.

    Symmetric methods (std, ci_from_std, ci_from_ppf) return a ``float``
    half-width.  Asymmetric methods (ci_from_wilson, ci_from_bootstrap) return
    a ``(ci_lower, ci_upper)`` tuple of absolute bounds.
    """
    if method.method == "std":
        return _interval_std
    if method.method == "ci_from_std":
        return partial(_interval_ci_from_std, confidence=method.confidence)
    if method.method == "ci_from_ppf":
        return partial(_interval_ci_from_ppf, confidence=method.confidence)
    if method.method == "ci_from_wilson":
        return partial(_interval_ci_from_wilson, confidence=method.confidence)
    if method.method == "ci_from_bootstrap":
        return partial(
            _interval_ci_from_bootstrap,
            confidence=method.confidence,
            n_resamples=method.n_resamples,
            seed=method.seed,
        )
    if method.method == "ci_from_weighted_bootstrap":
        # Weighted bootstrap has a (values, weights) -> (lo, hi) signature;
        # _agg_sweep handles the weights argument specially.
        return partial(
            _interval_ci_from_weighted_bootstrap,
            confidence=method.confidence,
            n_resamples=method.n_resamples,
            seed=method.seed,
        )
    raise ValueError(f"Unknown interval method: {method.method!r}")


def _build_mass_mask(
    cm_all: np.ndarray,
    nc_all: np.ndarray | None,
    min_choice_mass: float,
    dynamic_mass_filter: bool,
) -> np.ndarray:
    """Build a boolean mask combining dynamic and fixed choice-mass filters.

    Args:
        cm_all: Per-sample choice mass values.
        nc_all: Per-sample num_choices values (for dynamic threshold).
            May be None if not available.
        min_choice_mass: Fixed minimum threshold (0 = no fixed filter).
        dynamic_mass_filter: If True, apply per-question 1/num_choices filter.

    Returns:
        Boolean mask — True for samples to keep.
    """
    mask = np.ones(len(cm_all), dtype=bool)
    if dynamic_mass_filter and nc_all is not None and len(nc_all) == len(cm_all):
        dynamic_thresholds = 1.0 / nc_all
        mask &= cm_all >= dynamic_thresholds
    if min_choice_mass > 0.0:
        mask &= cm_all >= min_choice_mass
    return mask


def _agg_sweep(
    df: pd.DataFrame,
    cols: list[str],
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
) -> pd.DataFrame:
    """Aggregate a sweep DataFrame to mean ± interval per scale point.

    Returns a DataFrame with columns: ``scale``, ``{col}_mean``, and — when
    *interval* is not None — interval columns for each *col*.

    Symmetric methods produce a single ``{col}_ci`` column (half-width).
    Asymmetric methods (Wilson, bootstrap) produce ``{col}_ci_low`` and
    ``{col}_ci_high`` columns with absolute bounds.

    Methods with :pyattr:`IntervalMethod.needs_raw_scores` (Wilson, bootstrap)
    compute CIs from the raw per-sample scores in ``_raw_{col}`` list columns
    (populated by :func:`_load_from_info`).  A ``ValueError`` is raised if
    these columns are missing.

    Two-level choice-mass filtering:

    1. **Dynamic** (``dynamic_mass_filter=True``): per-question threshold of
       ``1/num_choices`` using ``_raw__nc_{col}`` columns.
    2. **Fixed** (``min_choice_mass > 0``): global threshold applied on top.

    Args:
        df: Sweep DataFrame with per-run rows.
        cols: Metric columns to aggregate.
        interval: Error bar method.
        min_choice_mass: When > 0, exclude per-sample scores whose choice
            mass is below this threshold.  Requires ``_raw__cm_{col}``
            columns (logprob evals).  The mean is recomputed from the
            filtered raw scores.  Default 0.0 (no filtering).
        dynamic_mass_filter: When True, exclude per-sample scores whose
            choice mass is below ``1/num_choices``.  Requires
            ``_raw__nc_{col}`` columns.  Default True.
    """
    interval_fn = _resolve_interval_fn(interval) if interval is not None else None
    needs_raw = interval is not None and interval.needs_raw_scores
    needs_weights = interval is not None and interval.needs_weights
    asymmetric = needs_raw  # raw-score methods always produce asymmetric bounds
    # Choice-mass filtering also requires raw scores to recompute the mean.
    filter_by_mass = min_choice_mass > 0.0 or dynamic_mass_filter
    rows = []
    for scale, grp in df.groupby("scale"):
        row: dict = {"scale": scale}
        for col in cols:
            if col not in grp.columns:
                row[f"{col}_mean"] = float("nan")
                if interval_fn is not None:
                    if asymmetric:
                        row[f"{col}_ci_low"] = float("nan")
                        row[f"{col}_ci_high"] = float("nan")
                    else:
                        row[f"{col}_ci"] = 0.0
                continue

            # --- Choice-mass filtering: recompute mean from raw scores ---
            if filter_by_mass:
                raw_col = f"_raw_{col}"
                cm_col = f"_raw__cm_{col}"
                nc_col = f"_raw__nc_{col}"
                if raw_col in grp.columns and cm_col in grp.columns:
                    raw_lists = grp[raw_col].dropna().tolist()
                    cm_lists = grp[cm_col].dropna().tolist()
                    raw_all = np.concatenate(raw_lists) if raw_lists else np.array([])
                    cm_all = np.concatenate(cm_lists) if cm_lists else np.array([])
                    # Load num_choices arrays for dynamic filtering.
                    nc_all = None
                    if dynamic_mass_filter and nc_col in grp.columns:
                        nc_lists = grp[nc_col].dropna().tolist()
                        nc_all = np.concatenate(nc_lists) if nc_lists else None
                    min_len = min(len(raw_all), len(cm_all))
                    raw_all = raw_all[:min_len]
                    cm_all = cm_all[:min_len]
                    if nc_all is not None:
                        nc_all = nc_all[:min_len]
                    mask = _build_mass_mask(
                        cm_all, nc_all, min_choice_mass, dynamic_mass_filter
                    )
                    filtered = raw_all[mask]
                    mean = float(filtered.mean()) if len(filtered) else float("nan")
                else:
                    vals = grp[col].dropna().values
                    mean = vals.mean() if len(vals) else float("nan")
            else:
                vals = grp[col].dropna().values
                mean = vals.mean() if len(vals) else float("nan")
            row[f"{col}_mean"] = mean
            if interval_fn is not None:
                if needs_raw:
                    raw_col = f"_raw_{col}"
                    if raw_col not in grp.columns:
                        # No per-sample data for this column (e.g. summary
                        # metrics like logprob_mcq_ratio).  Skip CI — the
                        # mean is still computed from the aggregate values.
                        if asymmetric:
                            row[f"{col}_ci_low"] = float("nan")
                            row[f"{col}_ci_high"] = float("nan")
                        else:
                            row[f"{col}_ci"] = 0.0
                        continue
                    # Concatenate raw score lists across all runs in this group
                    raw_lists = grp[raw_col].dropna().tolist()
                    raw_all = np.concatenate(raw_lists) if raw_lists else np.array([])

                    # Apply choice-mass filter to CI raw scores too.
                    cm_col = f"_raw__cm_{col}"
                    nc_col = f"_raw__nc_{col}"
                    if filter_by_mass and cm_col in grp.columns:
                        cm_lists = grp[cm_col].dropna().tolist()
                        cm_all = np.concatenate(cm_lists) if cm_lists else np.array([])
                        nc_all = None
                        if dynamic_mass_filter and nc_col in grp.columns:
                            nc_lists = grp[nc_col].dropna().tolist()
                            nc_all = np.concatenate(nc_lists) if nc_lists else None
                        _ml = min(len(raw_all), len(cm_all))
                        raw_all = raw_all[:_ml]
                        cm_all = cm_all[:_ml]
                        if nc_all is not None:
                            nc_all = nc_all[:_ml]
                        mask = _build_mass_mask(
                            cm_all, nc_all, min_choice_mass, dynamic_mass_filter
                        )
                        raw_all = raw_all[mask]
                        # Also filter weights if needed for weighted bootstrap.
                        if needs_weights:
                            cm_all = cm_all[mask]

                    if len(raw_all) == 0:
                        row[f"{col}_ci_low"] = float("nan")
                        row[f"{col}_ci_high"] = float("nan")
                    elif needs_weights:
                        # Noise-injection bootstrap: use per-sample choice mass
                        # to model measurement uncertainty from low coverage.
                        if not (filter_by_mass and cm_col in grp.columns):
                            # Weights not yet loaded from filtering path above.
                            weight_col = f"_raw__cm_{col}"
                            if weight_col in grp.columns:
                                wt_lists = grp[weight_col].dropna().tolist()
                                wt_all = (
                                    np.concatenate(wt_lists)
                                    if wt_lists
                                    else np.ones(len(raw_all))
                                )
                            else:
                                wt_all = np.ones(len(raw_all))
                            min_len = min(len(raw_all), len(wt_all))
                            raw_all = raw_all[:min_len]
                            wt_all = wt_all[:min_len]
                        else:
                            wt_all = cm_all  # Already filtered above.
                        low, high = interval_fn(raw_all, wt_all)  # type: ignore[misc]
                        # Point estimate: E[cm * score + (1-cm) * U(0,1)]
                        #               = cm * score + (1-cm) * 0.5
                        adjusted = wt_all * raw_all + (1.0 - wt_all) * 0.5
                        row[f"{col}_mean"] = float(adjusted.mean())
                        row[f"{col}_ci_low"] = low
                        row[f"{col}_ci_high"] = high
                    else:
                        low, high = interval_fn(raw_all)  # type: ignore[misc]
                        row[f"{col}_ci_low"] = low
                        row[f"{col}_ci_high"] = high
                else:
                    if not filter_by_mass:
                        row[f"{col}_ci"] = interval_fn(vals)
                    else:
                        # Symmetric CI methods don't use raw scores, but we
                        # still need to filter.  Fall back to unfiltered vals
                        # since symmetric methods can't use raw+cm pairs.
                        row[f"{col}_ci"] = interval_fn(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)

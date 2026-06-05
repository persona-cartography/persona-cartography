"""Shared matplotlib drawing helpers for sweep / capability plotting.

Leaf module of drawing helpers extracted verbatim from
``src_dev/evals/personality/analyze_results.py`` to break the
sweep_plots<->capability_plots cycle.

Function bodies are byte-for-byte identical to the source; only their file
location and module-level imports differ.
"""

# Third-party
import numpy as np
import pandas as pd


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _draw_error_bars(
    ax,
    scales,
    means,
    cis=None,
    *,
    ci_low=None,
    ci_high=None,
    color=None,
) -> None:
    """Draw vertical error bars at each scale point.

    Accepts either symmetric half-widths (*cis*) or asymmetric absolute bounds
    (*ci_low*, *ci_high*).  No-op if all intervals are zero-width.
    """
    if ci_low is not None and ci_high is not None:
        means_arr = np.array(means, dtype=float)
        lo = np.array(ci_low, dtype=float)
        hi = np.array(ci_high, dtype=float)
        # Mask out points where mean or bounds are nan
        valid = np.isfinite(means_arr) & np.isfinite(lo) & np.isfinite(hi)
        if not np.any(valid):
            return
        yerr = np.array([means_arr - lo, hi - means_arr])
        # Clamp to non-negative (floating-point arithmetic can produce tiny
        # negative values when mean ≈ bound) and zero out nan points.
        np.clip(yerr, 0.0, None, out=yerr)
        yerr[:, ~valid] = 0.0
        if not np.any(yerr > 0):
            return
    elif cis is not None:
        yerr = np.array(cis)
        if not np.any(yerr > 0):
            return
    else:
        return
    ax.errorbar(
        scales,
        means,
        yerr=yerr,
        fmt="none",
        color=color,
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        alpha=0.7,
        zorder=5,
    )


def _draw_col_error_bars(ax, agg: pd.DataFrame, col: str, scales, means, color) -> None:
    """Draw error bars for *col* from an aggregated DataFrame.

    Handles both symmetric (``{col}_ci``) and asymmetric
    (``{col}_ci_low`` / ``{col}_ci_high``) columns automatically.
    """
    if f"{col}_ci" in agg.columns:
        _draw_error_bars(ax, scales, means, cis=agg[f"{col}_ci"].values, color=color)
    elif f"{col}_ci_low" in agg.columns and f"{col}_ci_high" in agg.columns:
        _draw_error_bars(
            ax,
            scales,
            means,
            ci_low=agg[f"{col}_ci_low"].values,
            ci_high=agg[f"{col}_ci_high"].values,
            color=color,
        )


def _set_scale_xticks(
    ax, scales, x_lim: tuple[float, float] | None = (-4.5, 4.5)
) -> None:
    """Set x-axis ticks at every scale point, labelling multiples of 0.5.

    All scale points get a tick mark. Labels are shown only at multiples of
    0.5 (or at every point if all points already fall on 0.5 steps), so the
    axis stays readable without rotation even for dense fine-grained grids.

    Args:
        x_lim: X-axis limits as (min, max). Defaults to (-4.5, 4.5). Pass None
            to auto-scale.
    """
    ax.set_xticks(scales)
    half_scales = {s for s in scales if round(float(s) * 2) == float(s) * 2}
    if len(half_scales) < len(scales):
        ax.set_xticklabels([f"{s:g}" if s in half_scales else "" for s in scales])
    else:
        ax.set_xticklabels([f"{s:g}" for s in scales])
    if x_lim is not None:
        ax.set_xlim(*x_lim)

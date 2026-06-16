"""Sweep plotting for personality eval results.

Personality/BFI/generic/parse-rate plotters and the
plot-dispatch registry.
Capability plotters live in :mod:`src.visualisations.capability_plots`; the
shared matplotlib drawing helpers live in :mod:`src.visualisations.plot_common`.

Function bodies are byte-for-byte identical to the source; only module-level
imports differ.
"""

# Standard library
import sys
from pathlib import Path
from typing import Callable, Literal

# Third-party
import numpy as np
import pandas as pd

# Local
from src.evals.personality.ci import IntervalMethod, _agg_sweep
from src.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT
from src.evals.personality.sweep_results import (
    ALL_TRAIT_COLS,
    BIG_FIVE,
    DARK_TRIAD,
    SweepData,
    _OCEAN_ALIASES,
    _metric_cols,
    _normalise_scale_col,
)
from src.visualisations.capability_plots import (
    plot_capability_breakdown,
    plot_capability_sweep,
)
from src.visualisations.palette import BIG_FIVE_COLORS, DARK_TRIAD_COLORS
from src.visualisations.plot_common import (
    _draw_col_error_bars,
    _draw_error_bars,
    _set_scale_xticks,
)


def _resolve_highlight(highlight: list[str] | None) -> set[str]:
    """Resolve highlight list (full names or OCEAN letters) to a set of full trait names.

    Returns the full Big Five set when highlight is None or empty (all lit by default).
    """
    if not highlight:
        return set(BIG_FIVE)
    resolved = set()
    for h in highlight:
        resolved.add(_OCEAN_ALIASES.get(h, h))
    return resolved


def print_sweep_table(agg: pd.DataFrame, cols: list[str], title: str) -> None:
    has_sym_ci = cols and f"{cols[0]}_ci" in agg.columns
    has_asym_ci = cols and f"{cols[0]}_ci_low" in agg.columns
    width = 10 + 26 * len(cols) if has_asym_ci else 10 + 20 * len(cols)
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)
    header = f"{'Scale':<10}" + "".join(
        f"{c:<{26 if has_asym_ci else 20}}" for c in cols
    )
    print(header)
    print("-" * width)
    for _, row in agg.iterrows():
        marker = " ← baseline" if abs(row["scale"]) < 0.01 else ""
        if has_sym_ci:
            vals = "".join(
                f"{row[f'{c}_mean']:.4f}±{row[f'{c}_ci']:.4f}{'':>6}" for c in cols
            )
        elif has_asym_ci:
            vals = "".join(
                f"{row[f'{c}_mean']:.4f} [{row[f'{c}_ci_low']:.4f},{row[f'{c}_ci_high']:.4f}] "
                for c in cols
            )
        else:
            vals = "".join(f"{row[f'{c}_mean']:.4f}{'':>14}" for c in cols)
        print(f"{row['scale']:+7.2f}   {vals}{marker}")
    print("=" * width)


def _mock_sweep_data() -> SweepData:
    """Generate realistic mock SweepData for offline testing."""
    rng = np.random.default_rng(42)
    scales = [s for s in np.arange(-2.0, 2.25, 0.25) if round(s, 10) != 0.0]
    base_scales = [0.0] + list(scales)

    def _df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return (
            _normalise_scale_col(df)
            .sort_values(["scale", "run"])
            .reset_index(drop=True)
        )

    # BFI: Big Five only, 3 runs per scale
    bfi_base = {
        "Openness": 0.65,
        "Conscientiousness": 0.70,
        "Extraversion": 0.50,
        "Agreeableness": 0.60,
        "Neuroticism": 0.45,
    }
    bfi_slope = {
        "Openness": 0.08,
        "Conscientiousness": -0.03,
        "Extraversion": 0.05,
        "Agreeableness": -0.12,
        "Neuroticism": 0.06,
    }

    def _mock_parse_rate(s: float) -> float:
        """Simulate parse degradation at extreme scales (perfect in the middle)."""
        return float(
            np.clip(1.0 - 0.08 * max(0.0, abs(s) - 1.0) + rng.normal(0, 0.02), 0, 1)
        )

    bfi_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {
                t: float(
                    np.clip(bfi_base[t] + bfi_slope[t] * s + rng.normal(0, 0.025), 0, 1)
                )
                for t in BIG_FIVE
            }
            raw = {
                f"_raw_{t}": np.clip(rng.normal(scores[t], 0.05, size=8), 0, 1)
                for t in BIG_FIVE
            }
            bfi_records.append(
                {
                    "model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                    "run": run,
                    "scale": s,
                    "_parse_rate": _mock_parse_rate(s),
                    **scores,
                    **raw,
                }
            )

    # TRAIT: Big Five + Dark Triad, 3 runs per scale
    trait_base = {
        **bfi_base,
        "Machiavellianism": 0.40,
        "Narcissism": 0.45,
        "Psychopathy": 0.35,
    }
    trait_slope = {
        **bfi_slope,
        "Machiavellianism": 0.03,
        "Narcissism": 0.02,
        "Psychopathy": 0.01,
    }
    trait_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {
                t: float(
                    np.clip(
                        trait_base[t] + trait_slope[t] * s + rng.normal(0, 0.025), 0, 1
                    )
                )
                for t in ALL_TRAIT_COLS
            }
            raw = {
                f"_raw_{t}": np.clip(rng.normal(scores[t], 0.05, size=8), 0, 1)
                for t in ALL_TRAIT_COLS
            }
            trait_records.append(
                {
                    "model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                    "run": run,
                    "scale": s,
                    "_parse_rate": _mock_parse_rate(s),
                    **scores,
                    **raw,
                }
            )

    # MMLU: coarser grid, 3 runs
    mmlu_scales = [s for s in np.arange(-2.0, 2.25, 0.5) if round(s, 10) != 0.0]
    mmlu_records = []
    for s in [0.0] + list(mmlu_scales):
        for run in ["run_1", "run_2", "run_3"]:
            acc = float(np.clip(0.62 - 0.04 * abs(s) + rng.normal(0, 0.01), 0, 1))
            raw_acc = rng.binomial(1, acc, size=8).astype(float)
            mmlu_records.append(
                {
                    "model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                    "run": run,
                    "scale": s,
                    "_parse_rate": _mock_parse_rate(s),
                    "accuracy": float(raw_acc.mean()),
                    "_raw_accuracy": raw_acc,
                }
            )

    return SweepData(
        evals={
            "bfi": _df(bfi_records),
            "trait": _df(trait_records),
            "mmlu": _df(mmlu_records),
        }
    )


def plot_trait_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
    x_lim: tuple[float, float] | None = (-4.5, 4.5),
) -> Path:
    """Primary research plot: TRAIT Big Five + Dark Triad + human baselines.

    When the DataFrame contains ``_choice_mass`` data (logprob-based evals),
    a compact diagnostic sub-axis is drawn below the main plot showing the
    fraction of probability mass on choice letters vs. scale.

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Psychopathy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted Big Five traits are dimmed.
            Dark Triad is always dimmed regardless. Defaults to all Big Five.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lit = _resolve_highlight(highlight)
    trait_agg = _agg_sweep(
        df,
        ALL_TRAIT_COLS,
        interval=interval,
        min_choice_mass=min_choice_mass,
        dynamic_mass_filter=dynamic_mass_filter,
    )
    scales = trait_agg["scale"].values

    # Detect whether choice-mass diagnostics are available.
    has_choice_mass = "_choice_mass" in df.columns and df["_choice_mass"].notna().any()

    if has_choice_mass:
        fig = plt.figure(figsize=(12, 6.6))
        gs = GridSpec(2, 1, height_ratios=[85, 15], hspace=0.05, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_cm = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax_cm = None

    # --- Big Five: lit at full color, dimmed if not in highlight ---
    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        if trait in lit:
            ax.plot(
                scales,
                means,
                "o-",
                color=color,
                linewidth=2.2,
                markersize=6,
                label=trait,
                zorder=4,
            )
            _draw_col_error_bars(ax, trait_agg, trait, scales, means, color)
        else:
            ax.plot(
                scales,
                means,
                "o-",
                color=color,
                linewidth=1.4,
                markersize=4,
                alpha=0.35,
                label=trait,
                zorder=3,
            )

    # --- Dark Triad: always dimmed dashed, no error bars; skip if absent ---
    for trait in DARK_TRIAD:
        col = f"{trait}_mean"
        if col not in trait_agg.columns or trait_agg[col].isna().all():
            continue
        color = DARK_TRIAD_COLORS[trait]
        means = trait_agg[col].values
        ax.plot(
            scales,
            means,
            "--",
            color=color,
            linewidth=1.4,
            markersize=4,
            alpha=0.35,
            label=trait,
            zorder=3,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_ylabel("Trait score (0–1)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    title = "TRAIT sweep: personality scores vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar(
            [],
            [],
            yerr=1,
            fmt="none",
            color="gray",
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            alpha=0.7,
            label=interval.label,
        )

    # --- Choice-mass diagnostic sub-axis ---
    if ax_cm is not None:
        # Aggregate per-sample choice mass at each scale, restricted to the
        # samples that survived the `min_choice_mass` filter used for the
        # trait scores above.  Pulled from raw per-sample values in
        # `_raw__choice_mass` (pooled across all runs at a given scale).
        cm_rows = []
        for scale, grp in df.groupby("scale"):
            if "_raw__choice_mass" in grp.columns:
                lists = [
                    v
                    for v in grp["_raw__choice_mass"].tolist()
                    if isinstance(v, list) and v
                ]
                cm_all = np.concatenate(lists) if lists else np.array([])
            else:
                cm_all = grp["_choice_mass"].dropna().values
            if min_choice_mass > 0.0:
                cm_all = cm_all[cm_all >= min_choice_mass]
            if len(cm_all):
                cm_rows.append(
                    {
                        "scale": scale,
                        "mean": float(cm_all.mean()),
                        "min": float(cm_all.min()),
                        "max": float(cm_all.max()),
                    }
                )
            else:
                cm_rows.append(
                    {
                        "scale": scale,
                        "mean": float("nan"),
                        "min": float("nan"),
                        "max": float("nan"),
                    }
                )
        cm_agg = pd.DataFrame(cm_rows).sort_values("scale")
        cm_scales = cm_agg["scale"].values
        cm_means = cm_agg["mean"].values

        ax_cm.plot(
            cm_scales,
            cm_means,
            "s-",
            color="#555555",
            linewidth=1.4,
            markersize=3,
            zorder=4,
        )
        ax_cm.axvline(
            0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1
        )
        ax_cm.set_ylabel(
            "Choice\nmass", fontsize=8, rotation=0, labelpad=32, va="center"
        )
        cm_lower = max(0.0, min(float(min_choice_mass), 1.0))
        ax_cm.set_ylim(cm_lower, 1.0)
        ax_cm.set_yticks([cm_lower, 1.0])
        ax_cm.set_yticklabels([f"{cm_lower:g}", "1"], fontsize=7)
        ax_cm.grid(True, alpha=0.25)
        ax_cm.set_xlabel(x_label, fontsize=11)
        _set_scale_xticks(ax_cm, scales, x_lim=x_lim)
        # Hide x-axis labels on the main plot since the sub-axis provides them.
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(x_label, fontsize=11)
        _set_scale_xticks(ax, scales, x_lim=x_lim)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13 if ax_cm is None else -0.35),
        fontsize=9,
        ncol=6,
        framealpha=0.85,
    )

    if ax_cm is None:
        plt.tight_layout()
    out = output_dir / "trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_bfi_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
    interval: IntervalMethod | None = None,
) -> Path:
    """Sanity-check plot: BFI Big Five centred at baseline (delta from scale=0).

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Neuroticism.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted traits are dimmed.
            Defaults to all Big Five.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    if interval is not None and interval.is_binary_only:
        raise ValueError(
            f"BFI trait scores are continuous, but {interval.method!r} is a "
            "binary-only interval method (Wilson). Use a continuous method, "
            "e.g. ci95_from_bootstrap."
        )

    lit = _resolve_highlight(highlight)
    bfi_agg = _agg_sweep(df, BIG_FIVE, interval=interval)
    has_sym_ci = f"{BIG_FIVE[0]}_ci" in bfi_agg.columns
    has_asym_ci = f"{BIG_FIVE[0]}_ci_low" in bfi_agg.columns

    # Compute per-trait baseline (scale=0) mean for delta calculation.
    baseline_row = bfi_agg[bfi_agg["scale"].abs() < 1e-9]
    if baseline_row.empty:
        baseline_row = bfi_agg.loc[[bfi_agg["scale"].abs().idxmin()]]

    scales = bfi_agg["scale"].values
    fig, ax = plt.subplots(figsize=(12, 5))

    all_delta_means: list[float] = []
    all_delta_cis: list[float] = []

    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        baseline_val = float(baseline_row[f"{trait}_mean"].iloc[0])
        means = bfi_agg[f"{trait}_mean"].values - baseline_val
        if trait in lit:
            ax.plot(
                scales,
                means,
                "o-",
                color=color,
                linewidth=2.2,
                markersize=6,
                label=trait,
                zorder=4,
            )
            if has_sym_ci:
                cis = bfi_agg[f"{trait}_ci"].values
                _draw_error_bars(ax, scales, means, cis=cis, color=color)
                all_delta_cis.extend(cis.tolist())
            elif has_asym_ci:
                ci_low = bfi_agg[f"{trait}_ci_low"].values - baseline_val
                ci_high = bfi_agg[f"{trait}_ci_high"].values - baseline_val
                _draw_error_bars(
                    ax, scales, means, ci_low=ci_low, ci_high=ci_high, color=color
                )
                all_delta_cis.extend((means - ci_low).tolist())
                all_delta_cis.extend((ci_high - means).tolist())
        else:
            ax.plot(
                scales,
                means,
                "o-",
                color=color,
                linewidth=1.4,
                markersize=4,
                alpha=0.35,
                label=trait,
                zorder=3,
            )
        all_delta_means.extend(means[~np.isnan(means)].tolist())

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="Baseline (s=0)",
        zorder=1,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.3, zorder=1)
    _set_scale_xticks(ax, scales)

    if all_delta_means:
        # Guard against non-finite CIs/means (e.g. an asymmetric method on a
        # column with no per-sample raw data yields NaN CIs) so they can't
        # reach set_ylim and raise "Axis limits cannot be NaN or Inf".
        finite_means = [m for m in all_delta_means if np.isfinite(m)]
        finite_cis = [c for c in all_delta_cis if np.isfinite(c)]
        max_ci = max(finite_cis) if finite_cis else 0.0
        if finite_means:
            data_min = min(finite_means) - max_ci
            data_max = max(finite_means) + max_ci
            margin = max(0.03, (data_max - data_min) * 0.15)
            ax.set_ylim(data_min - margin, data_max + margin)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Δ trait score (vs. baseline)", fontsize=11)
    ax.grid(True, alpha=0.25)

    title = "BFI sweep: trait delta from baseline"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar(
            [],
            [],
            yerr=1,
            fmt="none",
            color="gray",
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            alpha=0.7,
            label=interval.label,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=9,
        ncol=7,
        framealpha=0.85,
    )

    plt.tight_layout()
    out = output_dir / "bfi_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_generic_sweep(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
    interval: IntervalMethod | None = None,
) -> Path:
    """Generic sweep line plot for any eval not covered by a specialised plotter.

    Plots all numeric metric columns on a single 0–1 axis with CI bands.

    Args:
        df: DataFrame with columns: scale, run, + metric columns.
        eval_name: Used for the figure title and filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        interval: Error bar method. None to omit error bars.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cols = _metric_cols(df)
    agg = _agg_sweep(df, cols, interval=interval)
    scales = agg["scale"].values

    colors = cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, col in enumerate(cols):
        color = colors[i % len(colors)]
        means = agg[f"{col}_mean"].values
        ax.plot(
            scales,
            means,
            "o-",
            color=color,
            linewidth=2.0,
            markersize=5,
            label=col,
            zorder=4,
        )
        _draw_col_error_bars(ax, agg, col, scales, means, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1)
    _set_scale_xticks(ax, scales)
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ncol = min(len(cols), 5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=9,
        ncol=ncol,
        framealpha=0.85,
    )

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_parse_rate(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    """Parse rate companion plot: fraction of responses successfully scored vs. LoRA scale.

    Only generates a figure when at least one scale point has parse rate < 1.0.
    Returns None (and produces no file) when all points are 100%.

    Args:
        df: Sweep DataFrame with a ``_parse_rate`` column (0–1 per run).
        eval_name: Used for the figure title and output filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.

    Returns:
        Path to the saved figure, or None if skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if "_parse_rate" not in df.columns:
        return None

    agg = _agg_sweep(df, ["_parse_rate"])
    means = agg["_parse_rate_mean"].values
    if (means >= 1.0 - 1e-9).all():
        return None  # perfect parse rate everywhere — nothing to show

    # Use min/max across runs instead of CI — more informative for a bounded count.
    scales = agg["scale"].values
    pr_min = np.array([df[df["scale"] == s]["_parse_rate"].min() for s in scales])
    pr_max = np.array([df[df["scale"] == s]["_parse_rate"].max() for s in scales])
    err_lo = np.clip(means - pr_min, 0, None)
    err_hi = np.clip(pr_max - means, 0, None)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    color = "#78909C"
    ax.axhline(
        1.0,
        color="#388E3C",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label="100% parsed",
        zorder=2,
    )
    ax.plot(
        scales,
        means,
        "o-",
        color=color,
        linewidth=2.0,
        markersize=5,
        label="parse rate (mean)",
        zorder=4,
    )
    if (err_lo > 0).any() or (err_hi > 0).any():
        ax.errorbar(
            scales,
            means,
            yerr=[err_lo, err_hi],
            fmt="none",
            color=color,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            alpha=0.7,
            zorder=5,
            label="min/max",
        )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Parse rate", fontsize=11)
    ax.set_ylim(max(0.0, float(np.nanmin(pr_min)) - 0.05), 1.0)
    _set_scale_xticks(ax, scales)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: parse rate vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_parse_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

PlotFn = Callable[[pd.DataFrame, Path, str], Path]

# A plot style tag selects a built-in plotter; a PlotFn provides a fully custom one.
PlotStyle = Literal["trait", "bfi", "capability", "generic"]

# Registry mapping eval name → plot style or custom PlotFn.
#
# To add a new eval, insert one line here:
#   "my_eval": "capability"   — for any accuracy-like metric
#   "my_eval": "generic"      — for a quick look at arbitrary numeric columns
#   "my_eval": plot_my_sweep  — for a fully custom plot function
#
# Evals absent from this registry will not be plotted; a warning is printed
# instead so you know exactly what to do.
_PLOT_REGISTRY: dict[str, PlotFn | PlotStyle] = {
    # Behavioral evals
    "trait": "trait",
    "trait_logprobs": "trait",
    "bfi": "bfi",
    # Capability evals (text-based)
    "mmlu": "capability",
    "gsm8k": "capability",
    "popqa": "capability",
    "truthfulqa": "capability",
    # Capability evals (logprob-based)
    "mmlu_logprobs": "capability",
    "truthfulqa_logprobs": "capability",
    "gpqa_logprobs": "capability",
}


def generate_plots(
    data: SweepData,
    output_dir: Path,
    title_suffix: str = "",
    random_baseline: float | None = None,
    highlight: list[str] | None = None,
    show_parse_rate: bool = False,
    interval: IntervalMethod | str | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
    x_lim: tuple[float, float] | None = (-4.5, 4.5),
) -> list[Path]:
    """Generate all plots for the evals present in *data*.

    Uses the plot registry for known evals. Evals not present in the registry
    are skipped with a warning explaining how to add them.

    Args:
        data: SweepData loaded from a run directory.
        output_dir: Directory to save all figures.
        title_suffix: Optional title suffix forwarded to every plot function.
        random_baseline: If set, draws a random-chance reference line on capability plots
            (e.g. 0.25 for 4-choice MCQ).
        highlight: Traits to render at full brightness in trait/bfi plots.
            Accepts full names or OCEAN single letters (O/C/E/A/N).
            Defaults to all Big Five.
        show_parse_rate: If True, generate a companion parse rate plot for each
            eval when at least one scale point has parse rate < 100%.
        interval: Error bar method. Accepts an IntervalMethod, a string parseable
            by ``IntervalMethod.from_str()``, or None to omit error bars.
        min_choice_mass: Exclude logprob samples with choice mass below this
            threshold when computing means and CIs.  Defaults to
            ``MIN_CHOICE_MASS_DEFAULT``.  Set to 0 to disable the filter.
        dynamic_mass_filter: If True, exclude logprob samples with choice mass
            below 1/num_choices per question.  Default True.

    Returns:
        List of paths to saved figures.
    """
    if isinstance(interval, str):
        interval = IntervalMethod.from_str(interval)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for eval_name in data.names():
        df = data.get(eval_name)
        assert df is not None

        entry = _PLOT_REGISTRY.get(eval_name)

        # Hybrid-thinking evals carry a `_think`/`_nothink` suffix on the eval
        # name (e.g. `trait_logprobs_nothink`, `mmlu_nothink`) to separate
        # thinking-on/off runs in the output dirs + fingerprint. The plot
        # registry is keyed by the base eval name, so fall back to that.
        if entry is None:
            for _suf in ("_nothink", "_think"):
                if eval_name.endswith(_suf):
                    entry = _PLOT_REGISTRY.get(eval_name[: -len(_suf)])
                    break

        if entry is None:
            print(
                f"\nWARNING: eval '{eval_name}' has no registered plot style — skipping.\n"
                f"  To add it, insert one line in _PLOT_REGISTRY in analyze_results.py:\n"
                f'    "{eval_name}": "capability"   # for accuracy-like metrics\n'
                f'    "{eval_name}": "generic"       # for a quick look at any numeric columns\n'
                f'    "{eval_name}": plot_{eval_name}_sweep  # for a fully custom plot function\n',
                file=sys.stderr,
            )
            continue

        if entry == "capability":
            path = plot_capability_sweep(
                df,
                output_dir,
                title_suffix,
                eval_name=eval_name,
                random_baseline=random_baseline,
                interval=interval,
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic_mass_filter,
                x_label=x_label,
            )
            bd_path = plot_capability_breakdown(
                df, output_dir, title_suffix, eval_name=eval_name
            )
            if bd_path:
                saved.append(bd_path)
        elif entry == "trait":
            path = plot_trait_sweep(
                df,
                output_dir,
                title_suffix,
                highlight=highlight,
                interval=interval,
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic_mass_filter,
                x_label=x_label,
                x_lim=x_lim,
            )
        elif entry == "bfi":
            path = plot_bfi_sweep(
                df, output_dir, title_suffix, highlight=highlight, interval=interval
            )
        elif entry == "generic":
            path = plot_generic_sweep(
                df, eval_name, output_dir, title_suffix, interval=interval
            )
        elif callable(entry):
            path = entry(df, output_dir, title_suffix)
        else:
            raise ValueError(f"Unknown plot style {entry!r} for eval '{eval_name}'")

        saved.append(path)

        if show_parse_rate:
            pr_path = plot_parse_rate(df, eval_name, output_dir, title_suffix)
            if pr_path:
                saved.append(pr_path)

    return saved

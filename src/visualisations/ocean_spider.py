"""Reusable OCEAN spider/radar plot helper.

Call :func:`plot_ocean_spider` with a ``{scale: {trait: score}}`` mapping and
it overlays one polygon per selected scale on a 5-axis OCEAN radar.

Clean-layer copy of the spider helper (no method-version token — fully
generic). Migrated paper figures under ``scripts/visualisations`` import
``to_headroom`` / ``plot_ocean_spider`` from here so they do not reach into the
frozen in-development layer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def to_headroom(
    raw_scores: dict,
    baseline: dict[str, float],
    *,
    score_min: float,
    score_max: float,
) -> dict:
    """Convert per-entry trait score dicts to signed headroom fractions in ``[-1, +1]``.

    For each (entry, trait), the transform is:

    * ``(value - baseline) / (score_max - baseline)`` when ``value >= baseline``
    * ``(value - baseline) / (baseline - score_min)`` when ``value <  baseline``

    Intuition: ask "how much of the achievable room in the direction the
    adapter actually moved did it consume?". Baseline collapses to 0 on every
    axis and the two spiders (amp/sup) become directly comparable — each
    trait's axis is normalized by that trait's own asymmetric room. Missing
    baseline or ``None``/``NaN`` values propagate through as NaN so polygon
    gaps render naturally downstream.

    Mirrors ``scripts_dev/evals/ocean_spider_plot_judge_combined.py`` —
    the ``PLOT_MODE="headroom"`` path — so use the same ``(score_min,
    score_max)`` bounds as the underlying judge's score range (e.g. OCEAN
    v2 uses ``-4, +4``).

    Args:
        raw_scores: ``{entry_key: {trait: score}}`` — where ``entry_key`` can
            be any hashable (a scale, an adapter name, etc.).
        baseline: ``{trait: baseline_score}`` — reference to transform against.
        score_min: Lower bound of the judge's score range.
        score_max: Upper bound of the judge's score range.

    Returns:
        Same shape as ``raw_scores`` but with headroom-fraction values. Entries
        whose trait is missing from ``baseline``, or whose value is ``None``/
        ``NaN``, are skipped (so the polygon shows a gap).
    """
    out: dict = {}
    for key, trait_means in raw_scores.items():
        transformed: dict[str, float] = {}
        for trait, value in trait_means.items():
            b = baseline.get(trait)
            if b is None or value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            delta = float(value) - float(b)
            if delta >= 0:
                room = score_max - float(b)
            else:
                room = float(b) - score_min
            transformed[trait] = delta / room if room > 0 else 0.0
        out[key] = transformed
    return out


def plot_ocean_spider(
    *,
    scores_by_scale: dict[float, dict[str, float]],
    out_path: Path,
    title: str,
    scales_to_plot: list[float],
    style: dict[float, dict] | None = None,
    y_lim: tuple[float, float] = (0.0, 1.0),
    y_ticks: list[float] | None = None,
    traits: list[str] = OCEAN_TRAITS,
    fill_alpha: float = 0.12,
    line_width: float = 2.0,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> Path:
    """Render a polar plot with one closed polygon per scale in ``scales_to_plot``.

    Args:
        scores_by_scale: ``{scale_value: {trait_name: score}}``. Missing scales
            or missing traits are skipped with a warning.
        out_path: PNG path; parent directories are created.
        title: Plot title.
        scales_to_plot: Scales to overlay (in order).
        style: Per-scale ``{scale: {"label": str, "color": str}}`` overrides.
        y_lim: Radial axis limits. For judge data pass ``(-3, 3)``.
        y_ticks: Explicit radial ticks; defaults to 5 evenly-spaced values.
        traits: Axis trait names (defaults to the Big Five).
        fill_alpha: Alpha for the polygon fill.
        line_width: Line width for the polygon outline.
        figsize: Matplotlib figsize.
    """
    style = style or {}
    angles = np.linspace(0.0, 2.0 * np.pi, len(traits), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for scale in scales_to_plot:
        row = scores_by_scale.get(scale)
        if not row:
            print(f"  ⚠ scale={scale:+.2f} missing from data; skipping")
            continue
        means = [float(row.get(t, float("nan"))) for t in traits]
        if any(np.isnan(v) for v in means):
            print(f"  ⚠ scale={scale:+.2f} has NaN trait; skipping")
            continue
        means_closed = means + means[:1]
        st = style.get(scale, {})
        label = st.get("label") or ("base" if scale == 0.0 else f"{scale:+.2f}×")
        color = st.get("color")
        lw = st.get("linewidth", line_width)
        ax.plot(angles_closed, means_closed, "o-", color=color, linewidth=lw, label=label)
        ax.fill(angles_closed, means_closed, color=color, alpha=fill_alpha)

    ax.set_xticks(angles)
    ax.set_xticklabels(traits, fontsize=11)
    ax.set_ylim(*y_lim)
    if y_ticks is None:
        y_ticks = list(np.linspace(y_lim[0], y_lim[1], 6))[1:]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.2g}" for v in y_ticks], fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_title(title, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=10, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")
    return out_path

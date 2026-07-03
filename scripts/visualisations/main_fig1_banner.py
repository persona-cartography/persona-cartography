"""Single-figure Fig. 1 banner: amp spider + supp spider + combo delta in a row.

Combines the three sibling scripts into one wide banner with two legends:

  ┌──────────────┬──────────────┬──────────────┐
  │  amp spider  │ supp spider  │  combo bar   │
  └──────────────┴──────────────┴──────────────┘
       ── shared spider legend ──   bar legend

Color convention: OCEAN traits use ``BIG_FIVE_COLORS`` everywhere. In the
spiders, each polygon is colored by its *home* trait (the trait the LoRA
targets). In the bar chart, the C↓ / E↑ single-adapter bars are colored by
their home trait (Conscientiousness / Extraversion) and the combo bar is
rendered in a neutral "combo" color so it doesn't collide with any OCEAN hue.

Tick labels on the spiders and bar chart are abbreviated to single OCEAN
letters (O / C / E / A / N) to keep the banner compact.

Data hydration is delegated to the migrated sibling scripts:
  - ``main_amplifier_spider_paired_dpo.build_scores``
  - ``main_suppressor_spider_paired_dpo.build_scores``
  - the existing clean combo figure (``main_c_minus_e_plus_combo_delta_paired_dpo``)
    plus the shared ``src.visualisations.combo_delta`` path/hydration helpers,
    wrapped here in :func:`gather_with_response_data` (per-response medians for
    paired-bootstrap CIs).

so per-cell HF paths and fingerprint conventions stay in one place and we
don't accumulate duplicate fetcher logic.

Paper figures:
    paper/figures/main/fig_1_banner.pdf

Run with:
    uv run python scripts/visualisations/main_fig1_banner.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations import combo_delta as cd
from src.visualisations.judge_jsonl import judge_scores
from src.visualisations.ocean_spider import to_headroom
from src.visualisations.palette import BIG_FIVE_COLORS

import scripts.visualisations.main_amplifier_spider_paired_dpo as amp_mod
import scripts.visualisations.main_suppressor_spider_paired_dpo as sup_mod
from scripts.visualisations.main_c_minus_e_plus_combo_delta_paired_dpo import CONFIG as COMBO_CONFIG

PAPER_FIGURES = [
    "main/fig_1_banner.pdf",
]

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
OCEAN_LETTERS = ["O", "C", "E", "A", "N"]
OCEAN_KEY_BY_LETTER = dict(zip(OCEAN_LETTERS, OCEAN_TRAITS))

BASELINE_COLOR = "#4D4D4D"
COMBO_COLOR = "#00838F"
SCORE_MIN = -4.0
SCORE_MAX = 4.0
HEADROOM_LIM = (-1.0, 1.0)
HEADROOM_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
HEADROOM_LABELS = ["-100%", "-50%", "0", "+50%", "+100%"]

# Short arrow labels for the combo bar panel — mirror the combo figure's
# config labels (a = C↓ suppressor, b = E↑ amplifier).
TARGET_DISPLAY = {
    "c_adapter": COMBO_CONFIG.a_label,
    "e_adapter": COMBO_CONFIG.b_label,
    "combo": f"{COMBO_CONFIG.a_label} × {COMBO_CONFIG.b_label}",
}

# Single font-size scale knob — every label/title/legend uses one of these.
FS_TITLE       = 16
FS_TICK_OCEAN  = 16  # bold OCEAN-letter ticks (spiders + bar x-axis)
FS_TICK_HEAD   = 10  # spider radial tick labels
FS_AX_LABEL    = 15  # bar y-axis label
FS_TICK_VALUE  = 14  # bar y-tick numbers
FS_LEGEND      = 11


# ---------------------------------------------------------------------------
# Combo per-response hydration (reuses the clean combo module's paths/cfg)
# ---------------------------------------------------------------------------


def _prompt_key(rid: str) -> str:
    """Stable per-prompt key: the ``sample_<hash>`` token.

    Judge response ids carry a run-specific prefix and a repeat suffix
    (``scale_+0.00:sample_<hash>:0`` for baselines, ``cell_…:sample_<hash>:0``
    for combo cells). Keying on the ``sample_<hash>`` token strips both so that
    baseline / adapter / combo responses pair up by prompt — without it the
    paired bootstrap shares zero responses and the CIs collapse to zero.
    """
    for part in rid.split(":"):
        if part.startswith("sample_"):
            return part
    return rid


def _per_response_medians(jsonl_path: Path) -> dict[str, float]:
    """Median over repeats per prompt (failed-judge rows excluded)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for rid, score in judge_scores(jsonl_path):
        grouped[_prompt_key(rid)].append(score)
    return {k: statistics.median(v) for k, v in grouped.items() if v}


def _fetch_per_response(cell_hf_dir: str, trait_lower: str) -> dict[str, float] | None:
    hf_path = cd._judge_hf_path(cell_hf_dir, cd._trait_metric(trait_lower))
    local = cd._hydrate_judge_file(COMBO_CONFIG, hf_path)
    if local is None:
        return None
    medians = _per_response_medians(local)
    return medians or None


def gather_with_response_data() -> dict[str, dict[str, dict | float | None]]:
    """Per-trait means + per-response_id medians for baseline / C↓ / E↑ / combo.

    Reuses the shared ``src.visualisations.combo_delta`` path + hydration
    helpers and the ``main_c_minus_e_plus_combo_delta_paired_dpo`` CONFIG so
    the banner's combo cell paths are identical to the standalone combo figure
    (no re-derived HF paths). ``c_adapter`` is the C↓ suppressor (CONFIG.a),
    ``e_adapter`` is the E↑ amplifier (CONFIG.b).

    Per trait, the returned dict has:
      - ``baseline``, ``c_adapter``, ``e_adapter``, ``combo`` → mean (float | None)
      - ``baseline_resp``, ``c_adapter_resp``, ``e_adapter_resp``, ``combo_resp``
        → dict[response_id → median] (or None)
    """
    cfg = COMBO_CONFIG
    out: dict[str, dict[str, dict | float | None]] = {}
    for trait_lower, fp in cd.FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        print(f"\n[trait] {trait_title}  (fp={fp})")

        combo_resp    = _fetch_per_response(cd._combo_cell_hf_dir(cfg, fp, 1.0, 1.0), trait_lower)
        baseline_resp = _fetch_per_response(cd._baseline_hf_dir(fp), trait_lower)
        c_resp        = _fetch_per_response(cd._single_adapter_hf_dir(cfg.a_dir, fp, 1.0), trait_lower)
        e_resp        = _fetch_per_response(cd._single_adapter_hf_dir(cfg.b_dir, fp, 1.0), trait_lower)

        def _mean(d):
            return statistics.fmean(d.values()) if d else None

        out[trait_title] = {
            "baseline":       _mean(baseline_resp),
            "c_adapter":      _mean(c_resp),
            "e_adapter":      _mean(e_resp),
            "combo":          _mean(combo_resp),
            "baseline_resp":  baseline_resp,
            "c_adapter_resp": c_resp,
            "e_adapter_resp": e_resp,
            "combo_resp":     combo_resp,
        }
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _spider_panel(
    ax,
    *,
    per_lora: dict[str, dict[str, float]],
    baseline: dict[str, float],
    meta: list[tuple[str, str, str]],
    title: str,
) -> None:
    """Render a single OCEAN spider in headroom mode.

    ``per_lora`` is keyed by adapter slug (e.g. ``o_plus``) and inner-keyed by
    judged-trait title. ``meta`` is a list of (slug, home_trait_lower, color)
    tuples in OCEAN order.
    """
    per_lora_h = to_headroom(per_lora, baseline, score_min=SCORE_MIN, score_max=SCORE_MAX)

    angles = np.linspace(0.0, 2.0 * np.pi, len(OCEAN_TRAITS), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for slug, _home_trait, color in meta:
        row = per_lora_h.get(slug, {})
        if not row:
            continue
        means = [row.get(t, float("nan")) for t in OCEAN_TRAITS]
        means_closed = means + means[:1]
        ax.plot(angles_closed, means_closed, "-", color=color, linewidth=1.8)
        for angle, val in zip(angles, means):
            if not np.isnan(val):
                ax.plot([angle], [val], "o", color=color, markersize=4)
        if not any(np.isnan(v) for v in means):
            ax.fill(angles_closed, means_closed, color=color, alpha=0.10)

    # Baseline ring at r=0 (collapsed to a single circle in headroom mode).
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.plot(ring_theta, np.zeros_like(ring_theta), "-", color=BASELINE_COLOR, linewidth=2.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(OCEAN_LETTERS, fontsize=FS_TICK_OCEAN, fontweight="bold")
    for tick_label, letter in zip(ax.get_xticklabels(), OCEAN_LETTERS):
        tick_label.set_color(BIG_FIVE_COLORS[OCEAN_KEY_BY_LETTER[letter]])
    ax.set_ylim(*HEADROOM_LIM)
    ax.set_yticks(HEADROOM_TICKS)
    ax.set_yticklabels(HEADROOM_LABELS, fontsize=FS_TICK_HEAD)
    ax.grid(True, alpha=0.4)
    ax.set_title(title, fontsize=FS_TITLE, pad=10)


def _bar_panel(ax, *, scores: dict[str, dict[str, float | None]]) -> None:
    """Per-OCEAN-trait Δ-vs-baseline bars for c_adapter, e_adapter, combo."""
    keys = ["c_adapter", "e_adapter", "combo"]
    width = 0.8 / len(keys)
    x = np.arange(len(OCEAN_TRAITS))

    # Color the single-adapter bars by their real trait identities; the combo
    # bar takes a neutral color outside the OCEAN palette.
    c_color = BIG_FIVE_COLORS["Conscientiousness"]
    e_color = BIG_FIVE_COLORS["Extraversion"]
    color_by_key = {"c_adapter": c_color, "e_adapter": e_color, "combo": COMBO_COLOR}
    hatch_by_key = {"c_adapter": "//", "e_adapter": "\\\\", "combo": "xx"}

    rng = np.random.default_rng(0)
    n_boot = 1000

    def _delta_ci(base_resp, val_resp):
        if not base_resp or not val_resp:
            return np.nan, np.nan, np.nan
        shared = sorted(set(base_resp) & set(val_resp))
        if not shared:
            return np.nan, np.nan, np.nan
        diffs = np.array([val_resp[r] - base_resp[r] for r in shared], dtype=float)
        delta = float(diffs.mean())
        idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        boots = diffs[idx].mean(axis=1)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        return delta, lo, hi

    for i, key in enumerate(keys):
        deltas = []
        err_low = []
        err_high = []
        for trait_title in OCEAN_TRAITS:
            row = scores.get(trait_title, {})
            base_resp = row.get("baseline_resp")
            val_resp = row.get(f"{key}_resp")
            d, lo, hi = _delta_ci(base_resp, val_resp)
            if np.isnan(d):
                # Fall back to point-only delta if per-response data missing.
                base = row.get("baseline")
                val = row.get(key)
                d = (val - base) if base is not None and val is not None else np.nan
                lo, hi = d, d
            deltas.append(d)
            err_low.append(d - lo)
            err_high.append(hi - d)
        yerr = np.array([err_low, err_high])
        ax.bar(
            x + (i - (len(keys) - 1) / 2) * width,
            deltas, width,
            color=color_by_key[key],
            hatch=hatch_by_key[key],
            edgecolor="black", linewidth=0.5,
            yerr=yerr,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "black"},
        )

    ax.axhline(0.0, color=BASELINE_COLOR, linewidth=1.0)

    # Per-trait theoretical Δ ceilings (SCORE_MAX − baseline up,
    # SCORE_MIN − baseline down), drawn as a short dashed segment spanning
    # only that trait's bar group. Lets the reader see how much of the
    # available judge-score range each LoRA actually used.
    half_span = len(keys) * width / 2.0
    for i, trait_title in enumerate(OCEAN_TRAITS):
        base = scores.get(trait_title, {}).get("baseline")
        if base is None:
            continue
        max_up = SCORE_MAX - base
        max_down = SCORE_MIN - base
        label = "Theoretical Δ limit" if i == 0 else None
        ax.hlines(
            [max_up, max_down],
            xmin=i - half_span,
            xmax=i + half_span,
            colors="black",
            linestyles="--",
            linewidth=1.1,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_LETTERS, fontsize=FS_TICK_OCEAN, fontweight="bold")
    for tick_label, letter in zip(ax.get_xticklabels(), OCEAN_LETTERS):
        tick_label.set_color(BIG_FIVE_COLORS[OCEAN_KEY_BY_LETTER[letter]])
    ax.set_ylabel("Δ Judge Score vs Baseline", fontsize=FS_AX_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK_VALUE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "Linearly Combining LoRAs",
        fontsize=FS_TITLE, pad=10,
    )


def _spider_legend(ax_legend) -> None:
    """Render the shared spider legend inside its own (invisible) axes.

    ``ax_legend`` is a dedicated subplot that spans the two spider columns
    so its centre is centred between the spiders.
    """
    handles = [
        Line2D([0], [0], color=BIG_FIVE_COLORS["Openness"],          marker="o", markersize=5, lw=1.8, label="Openness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Conscientiousness"], marker="o", markersize=5, lw=1.8, label="Conscientiousness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Extraversion"],      marker="o", markersize=5, lw=1.8, label="Extraversion LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Agreeableness"],     marker="o", markersize=5, lw=1.8, label="Agreeableness LoRA"),
        Line2D([0], [0], color=BIG_FIVE_COLORS["Neuroticism"],       marker="o", markersize=5, lw=1.8, label="Neuroticism LoRA"),
        Line2D([0], [0], color=BASELINE_COLOR, lw=2.0, label="No LoRA"),
    ]
    ax_legend.legend(
        handles=handles,
        loc="center", ncol=3, fontsize=FS_LEGEND, frameon=False,
    )


def _bar_legend(ax_bar) -> None:
    """Render the bar-chart legend inside the bar plot, bottom-right."""
    handles = [
        Patch(facecolor=BIG_FIVE_COLORS["Conscientiousness"], hatch="//", edgecolor="black", linewidth=0.5, label=TARGET_DISPLAY["c_adapter"]),
        Patch(facecolor=BIG_FIVE_COLORS["Extraversion"],      hatch="\\\\", edgecolor="black", linewidth=0.5, label=TARGET_DISPLAY["e_adapter"]),
        Patch(facecolor=COMBO_COLOR,                          hatch="xx", edgecolor="black", linewidth=0.5, label="C↓ ⊕ E↑"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="Δ limit"),
    ]
    ax_bar.legend(
        handles=handles,
        loc="upper left", fontsize=FS_LEGEND, frameon=True, framealpha=0.92,
    )


def _square_polar_axes_in_cell(fig, ax) -> None:
    """Resize ``ax`` so its width-fraction equals its height-fraction (in
    inches). This makes the polar circle fill the cell, removing the round
    plot's natural left/right whitespace within a wider rectangular cell.
    """
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    # height-fraction × fig height (in inches) = height in inches
    target_in = pos.height * fig_h_in
    # width-fraction needed to match that height in inches
    target_width_frac = target_in / fig_w_in
    if target_width_frac >= pos.width:
        # Cell is already square or taller than wide — nothing to shrink.
        return
    cx = (pos.x0 + pos.x1) / 2.0
    new_x0 = cx - target_width_frac / 2.0
    ax.set_position([new_x0, pos.y0, target_width_frac, pos.height])


def _draw_spider_box(fig, ax_amp, ax_sup, ax_legend, *, pad: float = 0.012) -> tuple[float, float, float, float]:
    """Draw a rectangle hugging amp + supp + spider_legend with a small pad.

    ``pad`` is in figure-relative units; ~0.01 = ~1% of figure dimensions.
    Returns ``(x0, y0, x1, y1)`` of the rectangle in figure-relative coords so
    callers can align other panels (e.g. the bar chart) to the box height.
    """
    import matplotlib.patches as patches
    try:
        renderer = fig.canvas.get_renderer()
        bbox_amp = ax_amp.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        bbox_sup = ax_sup.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        bbox_leg = ax_legend.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    except Exception:
        bbox_amp = ax_amp.get_position()
        bbox_sup = ax_sup.get_position()
        bbox_leg = ax_legend.get_position()
    x0 = min(bbox_amp.x0, bbox_sup.x0, bbox_leg.x0) - pad
    x1 = max(bbox_amp.x1, bbox_sup.x1, bbox_leg.x1) + pad
    y0 = min(bbox_amp.y0, bbox_sup.y0, bbox_leg.y0) - pad
    y1 = max(bbox_amp.y1, bbox_sup.y1, bbox_leg.y1) + pad
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        transform=fig.transFigure,
        fill=False, edgecolor="black", linewidth=1.0, clip_on=False, zorder=10,
    )
    fig.add_artist(rect)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[fig1-banner] hydrating amplifier spider data...")
    per_amp, base_amp = amp_mod.build_scores()
    print("\n[fig1-banner] hydrating suppressor spider data...")
    per_sup, base_sup = sup_mod.build_scores()
    print("\n[fig1-banner] hydrating combo delta data...")
    combo_scores = gather_with_response_data()

    # Use either baseline (they should be identical — same prompts, same judge).
    baseline = base_amp or base_sup

    fig = plt.figure(figsize=(15.0, 4.6))

    # Outer split: [spider-block | bar-block]. wspace here controls the gap
    # between the spider box and the bar chart — wide enough that the bar
    # chart's y-axis label doesn't crash into the box.
    gs_outer = GridSpec(
        1, 2, figure=fig,
        width_ratios=[2.0, 1.35],
        wspace=0.25,
    )

    # Inner spider block: 2 cols × 2 rows (amp / sup over their shared legend).
    # wspace=-0.05 brings the polar axes close together — polar bbox is square
    # but the circle leaves visual whitespace, so a slight negative pulls them
    # in without overlapping content.
    gs_left = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0, 0],
        height_ratios=[1.0, 0.20],
        wspace=-0.25, hspace=0.05,
    )

    ax_amp           = fig.add_subplot(gs_left[0, 0], projection="polar")
    ax_sup           = fig.add_subplot(gs_left[0, 1], projection="polar")
    ax_spider_legend = fig.add_subplot(gs_left[1, 0:2])
    # Bar chart now takes the full right column — its legend lives inside it.
    ax_bar           = fig.add_subplot(gs_outer[0, 1])
    ax_spider_legend.axis("off")

    _spider_panel(
        ax_amp,
        per_lora=per_amp, baseline=baseline,
        meta=amp_mod.AMPLIFIERS_META,
        title="Amplifier",
    )
    _spider_panel(
        ax_sup,
        per_lora=per_sup, baseline=baseline,
        meta=sup_mod.SUPPRESSORS,
        title="Suppressor",
    )
    _bar_panel(ax_bar, scores=combo_scores)

    _spider_legend(ax_spider_legend)
    _bar_legend(ax_bar)

    fig.subplots_adjust(bottom=0.06, top=0.88, left=0.03, right=0.99)
    # Force a draw so tightbbox computations have a renderer.
    fig.canvas.draw()

    # Shrink the spider axes from the top so their subtitles ("Amplifier" /
    # "Suppressor") and the surrounding box fit *below* the bar chart's title
    # — leaving the bar title's height as the place for the shared box title.
    renderer = fig.canvas.get_renderer()
    bar_title_bbox = ax_bar.title.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    title_y = (bar_title_bbox.y0 + bar_title_bbox.y1) / 2.0

    # Box pad (matches _draw_spider_box default) + a little extra gap below
    # the bar title's baseline so the box top is visibly below it.
    BOX_PAD = 0.012
    GAP_BELOW_TITLE = 0.012
    desired_spider_tightbbox_top = bar_title_bbox.y0 - BOX_PAD - GAP_BELOW_TITLE

    amp_tb = ax_amp.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    sup_tb = ax_sup.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    spider_top = max(amp_tb.y1, sup_tb.y1)
    shrink = spider_top - desired_spider_tightbbox_top
    if shrink > 0:
        for ax in (ax_amp, ax_sup):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, max(pos.height - shrink, 0.05)])
    fig.canvas.draw()

    box_x0, box_y0, box_x1, _box_y1 = _draw_spider_box(
        fig, ax_amp, ax_sup, ax_spider_legend,
    )

    # Shared title sits in the strip above the box, vertically aligned with
    # the bar chart's title on the right.
    fig.text(
        (box_x0 + box_x1) / 2.0, title_y,
        "Trait-modifying LoRAs",
        ha="center", va="center",
        fontsize=FS_TITLE,
    )

    for rel in PAPER_FIGURES:
        out_path = PAPER_FIGURES_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"✓ saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

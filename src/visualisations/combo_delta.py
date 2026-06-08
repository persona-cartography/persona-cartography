"""Shared machinery for the per-trait OCEAN combo-delta bar charts.

Several paper figures (``scripts/visualisations/main_*_combo_delta*.py``) draw the
same chart: for a two-adapter combo (e.g. C↓ × E↓), hydrate the (+1, +1)
combo cell for each of the 5 OCEAN traits, plus the base-model baseline and
the two single-adapter cells, then plot the Qwen3-235B judge's mean trait
score delta vs baseline as a grouped bar chart (one group per OCEAN trait,
three bars: adapter A, adapter B, combo).

This module factors that shared hydration + delta-computation + bar-drawing
out of the individual figure scripts. Each script provides its own adapter
slugs / HF dirs / display labels / output path via :class:`ComboDeltaConfig`
and calls :func:`run_combo_delta`; the plotting logic, HF paths, and data math
are byte-for-byte identical to the pre-refactor per-script copies.

Combo-cell path note: the canonical combo cell directory orders the two
``{slug}{scale}`` parts alphabetically by adapter slug (matching
``CanonicalCell.combo_slug`` / cell spec construction). The helper sorts the
pair by slug so it reproduces each script's hand-written ordering exactly
(C before E, N before O, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from src.utils.hf_hub import download_path_to_dir
from src.visualisations.judge_jsonl import mean_judge_score
from src.visualisations.palette import BIG_FIVE_COLORS

# ---------------------------------------------------------------------------
# Shared constants (identical across every combo-delta figure)
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
EVAL_NAME = "llm_judge_lora_scale_sweep"
RATER_ID = "qwen3_235b"
SCORE_MIN = -4.0
SCORE_MAX = 4.0

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

# Per-trait rollout fingerprint for the 240×1 single-adapter/baseline sweep on
# that trait's dataset. Shared by all combo-delta figures.
FP_BY_TRAIT: dict[str, str] = {
    "openness": "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion": "a961f641eb",
    "agreeableness": "0705e3276a",
    "neuroticism": "b2a49f1b4d",
}

# Teal — deliberately outside the OCEAN palette so the combo bar is
# unambiguously "not a trait colour". Avoid purple (collides with
# Agreeableness), red/blue/green/orange (collide with the other four).
COMBO_COLOR = "#00838F"
BASELINE_LEGEND_LABEL = "baseline Llama3.1-8b-Instruct"


@dataclass(frozen=True)
class ComboDeltaConfig:
    """Per-figure configuration for a two-adapter OCEAN combo-delta chart.

    Args:
        a_slug: Adapter-A combo slug (e.g.
            ``ocean-conscientiousness-suppressor-ocean_const_paired_dpo``).
        b_slug: Adapter-B combo slug.
        a_dir: Adapter-A fine-tuning subdir (e.g.
            ``conscientiousness/suppressor/ocean_const_paired_dpo``).
        b_dir: Adapter-B fine-tuning subdir.
        a_label: Short display label for adapter A (e.g. ``"C↓"``).
        b_label: Short display label for adapter B (e.g. ``"E↓"``).
        a_trait_title: OCEAN palette key for adapter A's colour (e.g.
            ``"Conscientiousness"``).
        b_trait_title: OCEAN palette key for adapter B's colour.
        out_path: Absolute output path for the rendered figure.
        cache_dir: Directory used to cache hydrated judge jsonls.
        a_log_label: Label used in the per-trait gather() print line for
            adapter A (e.g. ``"C↓ v4 paired DPO"``).
        b_log_label: Label used in the per-trait gather() print line for
            adapter B.
        local_monorepo: Optional local mirror of the HF monorepo used as a
            fallback when an HF hydrate fails.
    """

    a_slug: str
    b_slug: str
    a_dir: str
    b_dir: str
    a_label: str
    b_label: str
    a_trait_title: str
    b_trait_title: str
    out_path: Path
    cache_dir: Path
    a_log_label: str
    b_log_label: str
    local_monorepo: Path | None = None

    @property
    def combo_slug(self) -> str:
        """Alphabetical combo slug (matches ``CanonicalCell.combo_slug``)."""
        return "__".join(sorted([self.a_slug, self.b_slug]))


# ---------------------------------------------------------------------------
# Path helpers (match CanonicalCell.hf_dir)
# ---------------------------------------------------------------------------


def _fmt_scale(x: float) -> str:
    """Scale formatting matching ``format_scale()`` in cell_identity.py."""
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _combo_cell_hf_dir(
    cfg: ComboDeltaConfig, fingerprint: str, a_scale: float, b_scale: float
) -> str:
    """HF dir for the (adapter-A=a_scale, adapter-B=b_scale) combo cell.

    The cell spec lists the two ``{slug}{scale}`` parts in alphabetical slug
    order, matching the canonical combo-cell directory layout.
    """
    parts = sorted(
        [(cfg.a_slug, a_scale), (cfg.b_slug, b_scale)],
        key=lambda sl: sl[0],
    )
    spec = "cell_" + "_".join(f"{slug}{_fmt_scale(scale)}" for slug, scale in parts)
    return f"combos/{MODEL_SLUG}/{cfg.combo_slug}/{EVAL_NAME}/{fingerprint}/{spec}"


def _single_adapter_hf_dir(adapter_dir: str, fingerprint: str, scale: float) -> str:
    """Single-adapter cell dir at a given scale."""
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{adapter_dir}"
        f"/evals/{EVAL_NAME}/{fingerprint}/scale_{_fmt_scale(scale)}"
    )


def _baseline_hf_dir(fingerprint: str) -> str:
    return f"combos/{MODEL_SLUG}/_baseline/{EVAL_NAME}/{fingerprint}"


def _judge_hf_path(cell_hf_dir: str, metric_name: str) -> str:
    return f"{cell_hf_dir}/judge_runs/{RATER_ID}/{metric_name}.jsonl"


# ---------------------------------------------------------------------------
# Hydration + cache
# ---------------------------------------------------------------------------


def _cache_path(cfg: ComboDeltaConfig, hf_path: str) -> Path:
    return cfg.cache_dir / hf_path


def _hydrate_judge_file(cfg: ComboDeltaConfig, hf_path: str) -> Path | None:
    """Resolve ``hf_path`` to a local file, preferring cache → HF → local monorepo.

    Lookup order:
      1. HTTP cache at ``cache_dir/{hf_path}`` (from a previous run).
      2. Fresh download from HF.
      3. ``local_monorepo/{hf_path}`` — used when HF is down / rate-limited
         and the file is sitting locally from a skip-upload sweep.
    """
    local = _cache_path(cfg, hf_path)
    if local.exists() and local.stat().st_size > 0:
        return local
    parent_hf = hf_path.rsplit("/", 1)[0]
    filename = hf_path.rsplit("/", 1)[1]
    local_parent = _cache_path(cfg, parent_hf)
    hf_exc: Exception | None = None
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=parent_hf,
            target_dir=local_parent,
            allow_patterns=[filename],
        )
    except Exception as exc:
        hf_exc = exc
    if local.exists() and local.stat().st_size > 0:
        return local
    # HF didn't have it — try the local monorepo mirror before giving up.
    if cfg.local_monorepo is not None:
        local_mirror = cfg.local_monorepo / hf_path
        if local_mirror.exists() and local_mirror.stat().st_size > 0:
            print(f"  ← local mirror: {hf_path}")
            return local_mirror
    if hf_exc is not None:
        print(f"  ✗ hydrate failed: {type(hf_exc).__name__}: {str(hf_exc)[:140]}")
    return None


def _mean_score_median_across_repeats(jsonl_path: Path) -> float | None:
    """Median over repeats per response_id, then mean across responses.

    Delegates to the shared judge-jsonl reducer so every figure aggregates
    judge scores identically (failed-call rows excluded).
    """
    return mean_judge_score(jsonl_path, median_over_repeats=True)


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def _trait_metric(trait_lower: str) -> str:
    return f"{trait_lower}_v2"


def _fetch(cfg: ComboDeltaConfig, cell_hf_dir: str, trait_lower: str) -> float | None:
    """Hydrate + parse one (cell, trait) mean score."""
    hf_path = _judge_hf_path(cell_hf_dir, _trait_metric(trait_lower))
    local = _hydrate_judge_file(cfg, hf_path)
    if local is None:
        return None
    return _mean_score_median_across_repeats(local)


def gather(cfg: ComboDeltaConfig) -> dict[str, dict[str, float | None]]:
    """Returns per-trait means for baseline, single adapters, and combo.

    ``a_adapter`` / ``b_adapter`` are internal keys for the two single-adapter
    cells. Display labels live in ``cfg.a_label`` / ``cfg.b_label``.
    """
    out: dict[str, dict[str, float | None]] = {}
    for trait_lower, fp in FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        print(f"\n[trait] {trait_title}  (fp={fp})")

        combo_mean = _fetch(cfg, _combo_cell_hf_dir(cfg, fp, 1.0, 1.0), trait_lower)
        baseline_mean = _fetch(cfg, _baseline_hf_dir(fp), trait_lower)
        a_mean = _fetch(cfg, _single_adapter_hf_dir(cfg.a_dir, fp, 1.0), trait_lower)
        b_mean = _fetch(cfg, _single_adapter_hf_dir(cfg.b_dir, fp, 1.0), trait_lower)

        print(f"  baseline           : {baseline_mean}")
        print(f"  {cfg.a_log_label}   : {a_mean}")
        print(f"  {cfg.b_log_label}   : {b_mean}")
        print(f"  combo (+1, +1)     : {combo_mean}")

        out[trait_title] = {
            "baseline": baseline_mean,
            "a_adapter": a_mean,
            "b_adapter": b_mean,
            "combo": combo_mean,
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_TARGET_ORDER = ["a_adapter", "b_adapter", "combo"]
_TARGET_HATCHES = {
    "a_adapter": "//",
    "b_adapter": "\\\\",
    "combo": "xx",
}


def _output_path(base: Path, *, headroom: bool = False, ceiling: bool = False) -> Path:
    if headroom:
        return base.with_name(f"{base.stem}_headroom{base.suffix}")
    if ceiling:
        return base.with_name(f"{base.stem}_ceiling{base.suffix}")
    return base


def _bar_value(value: float, baseline: float, *, headroom: bool) -> float:
    delta = value - baseline
    if not headroom:
        return delta
    room = (SCORE_MAX - baseline) if delta >= 0 else (baseline - SCORE_MIN)
    return delta / room if room > 0 else 0.0


def render(
    cfg: ComboDeltaConfig,
    scores: dict[str, dict[str, float | None]],
    out_path: Path,
    *,
    headroom: bool = False,
    ceiling: bool = False,
) -> None:
    target_display = {
        "a_adapter": cfg.a_label,
        "b_adapter": cfg.b_label,
        "combo": f"{cfg.a_label} × {cfg.b_label}",
    }
    target_colors = {
        "a_adapter": BIG_FIVE_COLORS[cfg.a_trait_title],
        "b_adapter": BIG_FIVE_COLORS[cfg.b_trait_title],
        "combo": COMBO_COLOR,
    }

    x = np.arange(len(OCEAN_TRAITS))
    width = 0.8 / len(_TARGET_ORDER)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for i, key in enumerate(_TARGET_ORDER):
        deltas = []
        for trait_title in OCEAN_TRAITS:
            row = scores.get(trait_title, {})
            base = row.get("baseline")
            val = row.get(key)
            if base is None or val is None:
                deltas.append(np.nan)
            else:
                deltas.append(_bar_value(val, base, headroom=headroom))
        ax.bar(
            x + (i - (len(_TARGET_ORDER) - 1) / 2) * width,
            deltas,
            width,
            label=target_display[key],
            color=target_colors[key],
            hatch=_TARGET_HATCHES[key],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axhline(0.0, color="k", linewidth=0.8, label=BASELINE_LEGEND_LABEL)

    if ceiling and not headroom:
        # Per-trait Δ ceilings: how far above / below baseline the judge
        # score *could* go on a [SCORE_MIN, SCORE_MAX] scale. Drawn as a
        # short horizontal segment spanning that trait's bar group.
        half_span = len(_TARGET_ORDER) * width / 2.0
        for i, trait_title in enumerate(OCEAN_TRAITS):
            base = scores.get(trait_title, {}).get("baseline")
            if base is None:
                continue
            max_up = SCORE_MAX - base
            max_down = SCORE_MIN - base
            label = "theoretical Δ ceiling" if i == 0 else None
            ax.hlines(
                [max_up, max_down],
                xmin=i - half_span,
                xmax=i + half_span,
                colors="k",
                linestyles="--",
                linewidth=1.1,
                label=label,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=15, ha="right", fontsize=10)
    # Colour each trait tick label with its canonical OCEAN hue, matching
    # the spider plots in the same figure.
    for tick_label, trait in zip(ax.get_xticklabels(), OCEAN_TRAITS):
        tick_label.set_color(BIG_FIVE_COLORS[trait])
    if headroom:
        ax.set_ylabel("Signed headroom vs base model")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_ylabel("Δ judge score vs base model")
    # Let matplotlib auto-scale — deltas can range up to ±8 on the [−4, +4]
    # judge scale (e.g. baseline near +4 driven down to −4), so a hardcoded
    # (−4, +4) would clip. Tinted backgrounds follow the auto-scaled axis.
    ymin, ymax = ax.get_ylim()
    ax.axhspan(min(ymin, 0.0), 0.0, alpha=0.04, color="red")
    ax.axhspan(0.0, max(ymax, 0.0), alpha=0.04, color="blue")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ saved {out_path}")


def run_combo_delta(
    cfg: ComboDeltaConfig,
    *,
    headroom: bool = False,
    ceiling: bool = False,
) -> None:
    """Gather scores, persist them, and render the combo-delta figure."""
    out_path = _output_path(cfg.out_path, headroom=headroom, ceiling=ceiling)
    mode_name = "headroom" if headroom else ("ceiling" if ceiling else "delta")
    print(f"[combo-delta] cache dir: {cfg.cache_dir}")
    print(f"[combo-delta] mode:      {mode_name}")
    print(f"[combo-delta] out path:  {out_path}")
    scores = gather(cfg)
    # Persist raw scores alongside the figure for iteration / debugging.
    scores_path = cfg.cache_dir / "scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"✓ saved scores {scores_path}")
    render(cfg, scores, out_path, headroom=headroom, ceiling=ceiling)

"""Analyse the EQBench3 gemma-3-27b neuroticism sweep and plot the results.

Reads the eqbench3 ``runs.json`` / ``elo_results.json`` produced by
``run_gemma27b_eqbench3.py`` and produces a three-panel figure plus a stats
table:

  A. Headline rubric score (0-100) per variant, with BCa bootstrap 95% CIs
     over per-scenario task scores.
  B. Pairwise ELO (TrueSkill) per variant with +/-1.96 sigma intervals.
  C. Descriptive-criteria signature: per-criterion delta vs base for N+ and N-,
     showing the neuroticism dose-response on style/personality axes that are
     recorded but excluded from the headline score.

Per CLAUDE.md, continuous judge scores use BCa bootstrap intervals
(``_interval_ci_from_bootstrap``), not the normal approximation.

Usage:
    uv run python scripts_dev/eqbench_eval/analyze_gemma27b_eqbench3.py \
        --run-dir scratch/evals/eqbench3/gemma27b_n_sweep
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src_dev.evals.personality.analyze_results import _interval_ci_from_bootstrap
from src_dev.visualisations import PAPER_FIGURES_DIR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PAPER_FIGURES = [
    "main/fig_eqbench3_gemma27b_neuroticism.pdf",
    "main/fig_eqbench3_gemma27b_ranking.pdf",
    "main/fig_eqbench3_gemma27b_progression.pdf",
    "main/fig_eqbench3_gemma27b_progression_simple.pdf",
    "main/fig_eqbench3_gemma27b_ocean_traits.pdf",
]

# Variant display order and labels (trait axis: suppressor -> base -> amplifier).
VARIANTS = [
    ("gemma3_27b_n_minus", "N− (suppressor)"),
    ("gemma3_27b_base", "base"),
    ("gemma3_27b_n_plus", "N+ (amplifier)"),
]

# Okabe-Ito colourblind-safe palette. The trait axis is a polarity, so it gets a
# diverging assignment: cool pole = suppressor, neutral = base, warm = amplifier.
COLORS = {
    "gemma3_27b_n_minus": "#0072B2",  # blue
    "gemma3_27b_base": "#7F7F7F",  # neutral gray
    "gemma3_27b_n_plus": "#D55E00",  # vermillion
}

# The 6 qualitative criteria that feed eqbench3's headline score for standard
# role-play tasks, and the 4 additional ones it allows for analysis tasks.
HEADLINE_STANDARD = {
    "demonstrated_empathy",
    "pragmatic_ei",
    "depth_of_insight",
    "social_dexterity",
    "emotional_reasoning",
    "message_tailoring",
}
HEADLINE_ANALYSIS_EXTRA = {
    "theory_of_mind",
    "subtext_identification",
    "intellectual_grounding",
    "correctness",
}
HEADLINE_ALLOWED = HEADLINE_STANDARD | HEADLINE_ANALYSIS_EXTRA

DESCRIPTIVE = [
    "reactive",
    "moralising",
    "challenging",
    "boundary_setting",
    "safety_conscious",
    "analytical",
    "warmth",
    "validating",
    "compliant",
    "sycophantic",
    "conversational",
    "humanlike",
]


def _task_headline_score(rubric_scores: dict[str, Any]) -> float | None:
    """Mean of the allowed headline criteria present on one task (0-20 scale)."""
    vals = [
        v
        for k, v in rubric_scores.items()
        if k in HEADLINE_ALLOWED and isinstance(v, (int, float))
    ]
    return float(np.mean(vals)) if vals else None


def load_variant_tasks(runs_path: Path) -> dict[str, dict[tuple[str, str], dict]]:
    """Map variant -> {(iteration, scenario_id): rubric_scores} for scored tasks."""
    runs = json.loads(runs_path.read_text())
    out: dict[str, dict[tuple[str, str], dict]] = {}
    for run in runs.values():
        name = run.get("model_name")
        if not name:
            continue
        tasks: dict[tuple[str, str], dict] = {}
        for it, scenarios in (run.get("scenario_tasks") or {}).items():
            if not isinstance(scenarios, dict):
                continue
            for sid, task in scenarios.items():
                if not isinstance(task, dict):
                    continue
                if task.get("status") != "rubric_scored":
                    continue
                rs = task.get("rubric_scores")
                if isinstance(rs, dict) and rs:
                    tasks[(str(it), str(sid))] = rs
        out[name] = tasks
    return out


def headline_stats(tasks: dict[tuple[str, str], dict]) -> tuple[float, float, float, int]:
    """Return (mean_0_100, ci_lo_0_100, ci_hi_0_100, n_tasks)."""
    scores = [s for rs in tasks.values() if (s := _task_headline_score(rs)) is not None]
    arr = np.asarray(scores, dtype=float)
    lo, hi = _interval_ci_from_bootstrap(arr, confidence=95.0, n_resamples=2000, seed=SEED)
    return float(arr.mean()) * 5.0, lo * 5.0, hi * 5.0, arr.size


def paired_delta(
    a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict]
) -> tuple[float, float, float, int]:
    """Paired (a - b) headline delta on shared scenarios, 0-100 scale, with CI."""
    shared = sorted(set(a) & set(b))
    diffs = []
    for key in shared:
        sa, sb = _task_headline_score(a[key]), _task_headline_score(b[key])
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
    arr = np.asarray(diffs, dtype=float)
    lo, hi = _interval_ci_from_bootstrap(arr, confidence=95.0, n_resamples=2000, seed=SEED)
    return float(arr.mean()) * 5.0, lo * 5.0, hi * 5.0, arr.size


def criterion_mean(tasks: dict[tuple[str, str], dict], criterion: str) -> float | None:
    vals = [
        rs[criterion]
        for rs in tasks.values()
        if isinstance(rs.get(criterion), (int, float))
    ]
    return float(np.mean(vals)) if vals else None


def load_elo(elo_path: Path) -> dict[str, dict]:
    data = json.loads(elo_path.read_text())
    return {
        k: v
        for k, v in data.items()
        if k != "__metadata__" and isinstance(v, dict) and "elo" in v
    }


def build_figure(
    variant_tasks: dict[str, dict],
    elo: dict[str, dict],
    out_paths: list[Path],
) -> None:
    """Render the three-panel summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})

    # ── Panel A: headline rubric with bootstrap CIs ─────────────────────────
    ax = axes[0]
    labels, means, los, his, colors = [], [], [], [], []
    for key, label in VARIANTS:
        if key not in variant_tasks:
            continue
        m, lo, hi, n = headline_stats(variant_tasks[key])
        labels.append(f"{label}\n(n={n})")
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)
        colors.append(COLORS[key])
    y = np.arange(len(labels))
    ax.barh(y, means, color=colors, height=0.55, zorder=2)
    ax.errorbar(means, y, xerr=[los, his], fmt="none", ecolor="#333333", capsize=4, lw=1.4, zorder=3)
    for yi, m, hi_err in zip(y, means, his):
        # Clear the CI whisker so the value never overlaps the cap.
        ax.text(m + hi_err + 2.5, yi, f"{m:.1f}", va="center", fontsize=10, color="#222222")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlim(0, 80)
    ax.set_xlabel("EQ-Bench3 rubric score (0–100)")
    ax.set_title("A. Headline EQ score\n(95% BCa bootstrap CI)", fontsize=10, loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # ── Panel B: ELO with ±1.96σ ────────────────────────────────────────────
    ax = axes[1]
    labels, vals, errs, colors = [], [], [], []
    for key, label in VARIANTS:
        if key not in elo:
            continue
        labels.append(label)
        vals.append(elo[key]["elo"])
        sigma = elo[key].get("sigma") or 0.0
        errs.append(1.96 * sigma)
        colors.append(COLORS[key])
    y = np.arange(len(labels))
    ax.errorbar(vals, y, xerr=errs, fmt="o", ms=9, mfc="white", mew=2.2,
                ecolor="#333333", capsize=4, lw=1.4, zorder=3,
                markeredgecolor="#333333", linestyle="none")
    for yi, v, c in zip(y, vals, colors):
        ax.plot([v], [yi], "o", ms=9, color=c, zorder=4)
        ax.text(v, yi + 0.22, f"{v:.0f}", ha="center", fontsize=10, color="#222222")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("TrueSkill ELO (pairwise)")
    ax.set_title("B. Head-to-head ELO\n(±1.96σ)", fontsize=10, loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # ── Panel C: descriptive-criteria signature (delta vs base) ─────────────
    ax = axes[2]
    base = variant_tasks.get("gemma3_27b_base", {})
    rows = []
    for crit in DESCRIPTIVE:
        b = criterion_mean(base, crit)
        p = criterion_mean(variant_tasks.get("gemma3_27b_n_plus", {}), crit)
        m = criterion_mean(variant_tasks.get("gemma3_27b_n_minus", {}), crit)
        if None in (b, p, m):
            continue
        rows.append((crit, p - b, m - b))
    rows.sort(key=lambda r: r[1], reverse=True)  # sort by N+ effect

    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y + h / 2, [r[1] for r in rows], height=h,
            color=COLORS["gemma3_27b_n_plus"], label="N+ (amplifier)", zorder=2)
    ax.barh(y - h / 2, [r[2] for r in rows], height=h,
            color=COLORS["gemma3_27b_n_minus"], label="N− (suppressor)", zorder=2)
    ax.axvline(0, color="#555555", lw=1.1, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Δ vs base (judge score, 0–20 scale)")
    # Upstream marks these criteria "higher is not necessarily better or worse"
    # (data/rubric_scoring_prompt.txt) — they describe style/personality and are
    # excluded from the headline score. Say so, so the diverging colours are not
    # misread as good/bad.
    ax.set_title("C. Descriptive-trait signature vs base\n"
                 "style axes — NOT better/worse; excluded from headline score",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        "EQ-Bench3 (upstream, unmodified) — gemma-3-27b-it with neuroticism LoRA adapters",
        fontsize=12, x=0.005, ha="left", y=1.02,
    )
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def build_simple_figure(
    variant_tasks: dict[str, dict],
    elo: dict[str, dict],
    out_paths: list[Path],
) -> None:
    """Single-panel ranked bar chart of the judge's EQ score per variant."""
    rows = []
    for key, label in VARIANTS:
        if key not in variant_tasks:
            continue
        m, lo, hi, n = headline_stats(variant_tasks[key])
        rows.append((key, label, m, m - lo, hi - m, elo.get(key, {}).get("elo")))
    rows.sort(key=lambda r: r[2])  # ascending: best ends up on top in barh

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], color=[COLORS[r[0]] for r in rows],
            height=0.6, zorder=2)
    ax.errorbar([r[2] for r in rows], y,
                xerr=[[r[3] for r in rows], [r[4] for r in rows]],
                fmt="none", ecolor="#333333", capsize=4, lw=1.4, zorder=3)

    for yi, r in zip(y, rows):
        ax.text(r[2] + r[4] + 2.0, yi, f"{r[2]:.1f}", va="center",
                fontsize=12, color="#222222", fontweight="bold")
        if r[5] is not None:
            ax.text(2.5, yi, f"ELO {r[5]:.0f}", va="center", fontsize=9,
                    color="white", fontweight="bold")

    ax.set_yticks(y, [r[1] for r in rows], fontsize=11)
    ax.set_xlim(0, 72)
    ax.set_xlabel("EQ-Bench3 rubric score (0–100) — higher is better")
    ax.set_title("gemma-3-27b-it neuroticism adapters — EQ-Bench3 ranking\n"
                 "judge: claude-opus-4-6, scoring the full multi-turn trajectory + debrief\n"
                 "whiskers = 95% bootstrap CI (n=90 tasks)",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def build_simple_progression_figure(run_dir: Path, out_paths: list[Path]) -> None:
    """Single-panel per-turn EQ progression: one line per variant, CI bands."""
    per_turn = json.loads((run_dir / "per_turn_scores.json").read_text())
    scored = [r for r in per_turn if r.get("scores")]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ends: list[tuple[float, float, str]] = []  # (x_end, y_end, colour)
    for key, label in VARIANTS:
        xs, ys, los, his = [], [], [], []
        for k in range(3):
            vals = np.array([r["mean"] for r in scored
                             if r["variant"] == key and r["turn_index"] == k])
            if not vals.size:
                continue
            lo, hi = _interval_ci_from_bootstrap(vals, confidence=95.0,
                                                 n_resamples=2000, seed=SEED)
            xs.append(k + 1); ys.append(vals.mean()); los.append(lo); his.append(hi)
        ax.plot(xs, ys, "-o", color=COLORS[key], lw=2.6, ms=9, label=label, zorder=3)
        ax.fill_between(xs, los, his, color=COLORS[key], alpha=0.15, lw=0, zorder=2)
        ends.append((xs[-1], ys[-1], COLORS[key]))

    # Direct-label each line's end value, nudging apart any labels that would
    # collide (base and N- finish ~0.2 apart and would otherwise overprint).
    min_gap = 0.55
    ends.sort(key=lambda e: e[1])
    placed: list[float] = []
    for x_end, y_end, colour in ends:
        y_lab = y_end
        if placed and y_lab - placed[-1] < min_gap:
            y_lab = placed[-1] + min_gap
        placed.append(y_lab)
        ax.text(x_end + 0.07, y_lab, f"{y_end:.1f}", va="center",
                fontsize=11, fontweight="bold", color=colour)

    n = len([r for r in scored if r["variant"] == VARIANTS[0][0] and r["turn_index"] == 0])
    ax.set_xticks([1, 2, 3], ["turn 1", "turn 2", "turn 3"])
    ax.set_xlim(0.85, 3.35)
    ax.set_xlabel("assistant turn within the scenario")
    ax.set_ylabel("EQ score (0–20) — higher is better")
    ax.set_title("EQ across a multi-turn conversation — gemma-3-27b-it neuroticism adapters\n"
                 f"per-turn re-judge by claude-opus-4-6; bands = 95% bootstrap CI (n={n}/turn/variant)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def build_progression_figure(run_dir: Path, out_paths: list[Path]) -> None:
    """Plot per-turn EQ progression from ``per_turn_scores.json``.

    Turn composition is not constant: every multi-turn scenario has >=3 turns,
    but only the handful of 4-turn scenarios reach T4. T1-T3 are therefore drawn
    solid over the full, constant scenario set and T4 is drawn faded with its own
    n, so the drop into T4 is not mistaken for a like-for-like continuation.
    """
    per_turn = json.loads((run_dir / "per_turn_scores.json").read_text())
    scored = [r for r in per_turn if r.get("scores")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

    # ── Panel A: mean score by turn, with bootstrap CI ──────────────────────
    ax = axes[0]
    for key, label in VARIANTS:
        xs_solid, ys_solid, los, his = [], [], [], []
        for k in range(3):
            vals = np.array([r["mean"] for r in scored
                             if r["variant"] == key and r["turn_index"] == k])
            if not vals.size:
                continue
            lo, hi = _interval_ci_from_bootstrap(vals, confidence=95.0,
                                                 n_resamples=2000, seed=SEED)
            xs_solid.append(k + 1)
            ys_solid.append(vals.mean())
            los.append(lo)
            his.append(hi)
        ax.plot(xs_solid, ys_solid, "-o", color=COLORS[key], lw=2.2, ms=7,
                label=label, zorder=3)
        ax.fill_between(xs_solid, los, his, color=COLORS[key], alpha=0.15, lw=0, zorder=2)

        t4 = np.array([r["mean"] for r in scored
                       if r["variant"] == key and r["turn_index"] == 3])
        if t4.size:
            ax.plot([xs_solid[-1], 4], [ys_solid[-1], t4.mean()], "--",
                    color=COLORS[key], lw=1.6, alpha=0.55, zorder=3)
            ax.plot([4], [t4.mean()], "o", color=COLORS[key], ms=7, alpha=0.55, zorder=3)

    n_main = len({(r["variant"], r["scenario_id"], r["iteration"])
                  for r in scored if r["variant"] == VARIANTS[0][0]})
    n_t4 = len([r for r in scored if r["variant"] == VARIANTS[0][0] and r["turn_index"] == 3])
    ax.set_xticks([1, 2, 3, 4], ["T1", "T2", "T3", f"T4\n(n={n_t4})"])
    ax.set_xlabel("assistant turn within scenario")
    ax.set_ylabel("EQ score (0–20, higher better)")
    ax.set_title(f"A. EQ across the conversation\nsolid = T1–T3 (n={n_main}/variant, constant set); "
                 f"dashed = T4 subset", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ── Panel B: paired within-scenario degradation slope T1 -> T3 ──────────
    ax = axes[1]
    labels, means, errlo, errhi, colors = [], [], [], [], []
    for key, label in VARIANTS:
        by_task: dict[tuple, dict[int, float]] = {}
        for r in scored:
            if r["variant"] != key:
                continue
            by_task.setdefault((r["scenario_id"], r["iteration"]), {})[r["turn_index"]] = r["mean"]
        diffs = np.array([v[2] - v[0] for v in by_task.values() if 0 in v and 2 in v])
        if not diffs.size:
            continue
        lo, hi = _interval_ci_from_bootstrap(diffs, confidence=95.0, n_resamples=2000, seed=SEED)
        labels.append(label)
        means.append(diffs.mean())
        errlo.append(diffs.mean() - lo)
        errhi.append(hi - diffs.mean())
        colors.append(COLORS[key])
    y = np.arange(len(labels))
    ax.barh(y, means, color=colors, height=0.55, zorder=2)
    ax.errorbar(means, y, xerr=[errlo, errhi], fmt="none", ecolor="#333333",
                capsize=4, lw=1.4, zorder=3)
    for yi, m, e in zip(y, means, errlo):
        # Bars run negative (leftward); clear the lower CI whisker.
        ax.text(m - e - 0.18, yi, f"{m:+.2f}", va="center", ha="right",
                fontsize=10, color="#222222")
    ax.set_xlim(min(m - e for m, e in zip(means, errlo)) - 1.4, 0.25)
    ax.axvline(0, color="#555555", lw=1.1, zorder=3)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Δ EQ score, turn 3 − turn 1 (paired, same scenario)")
    ax.set_title("B. Degradation over the conversation\n(more negative = degrades faster)",
                 fontsize=10, loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Per-turn re-judge of EQ-Bench3 transcripts (judge: claude-opus-4-6) — "
                 "upstream scores the trajectory holistically; this resolves it by turn",
                 fontsize=11, x=0.005, ha="left", y=1.03)
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


OCEAN_TRAITS = [
    ("O", "openness", "#0072B2"),
    ("C", "conscientiousness", "#009E73"),
    ("E", "extraversion", "#D55E00"),
    ("A", "agreeableness", "#CC79A7"),
    ("N", "neuroticism", "#E69F00"),
]


def build_ocean_trait_figure(run_dir: Path, out_paths: list[Path]) -> None:
    """One line per OCEAN trait: EQ vs trait dose (suppressor -> base -> amplifier).

    All five lines share the same centre point (base), so the figure reads as
    "how does dialling each trait up or down move EQ from baseline". Headline
    rubric 0-100 with 95% BCa bootstrap CIs; a shaded control band gives the
    no-trait reference.
    """
    variant_tasks = load_variant_tasks(run_dir / "runs.json")

    def stat(model_name: str):
        tasks = variant_tasks.get(model_name)
        return headline_stats(tasks) if tasks else None

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = [-1, 0, 1]  # suppressor, base, amplifier

    base = stat("gemma3_27b_base")
    control = stat("gemma3_27b_control")
    if control is not None:
        cm, clo, chi, _ = control
        ax.axhspan(clo, chi, color="#999999", alpha=0.15, zorder=0)
        ax.axhline(cm, color="#999999", lw=1.0, ls=":", zorder=1)
        # Label at the left edge to avoid colliding with the amplifier end labels.
        ax.text(-1.44, cm, "control", va="bottom", fontsize=8, color="#666666")

    for code, trait, colour in OCEAN_TRAITS:
        minus = stat(f"gemma3_27b_{code.lower()}_minus")
        plus = stat(f"gemma3_27b_{code.lower()}_plus")
        if minus is None or plus is None or base is None:
            continue
        ys = [minus[0], base[0], plus[0]]
        los = [minus[0] - minus[1], base[0] - base[1], plus[0] - plus[1]]
        his = [minus[2] - minus[0], base[2] - base[0], plus[2] - plus[0]]
        ax.errorbar(x, ys, yerr=[los, his], fmt="-o", color=colour, lw=2.2, ms=7,
                    capsize=3, label=f"{code} — {trait}", zorder=3)
        ax.text(-1.06, ys[0], f"{code}−", va="center", ha="right", fontsize=9,
                color=colour, fontweight="bold")
        ax.text(1.06, ys[2], f"{code}+", va="center", ha="left", fontsize=9,
                color=colour, fontweight="bold")

    if base is not None:
        ax.plot([0], [base[0]], "o", ms=11, color="#000000", zorder=5)
        ax.text(0, base[0] + 1.0, f"base {base[0]:.1f}", ha="center", fontsize=9,
                fontweight="bold")

    ax.set_xticks(x, ["suppressor\n(−)", "base\n(no adapter)", "amplifier\n(+)"])
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("trait dose")
    ax.set_ylabel("EQ-Bench3 rubric score (0–100) — higher is better")
    ax.set_title("How each OCEAN trait adapter moves EQ — gemma-3-27b-it\n"
                 "EQ-Bench3 (upstream, unmodified), judge claude-opus-4-6, "
                 "n=45/variant, 95% BCa CI",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center")
    ax.grid(color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("scratch/evals/eqbench3/gemma27b_n_sweep"))
    ap.add_argument("--no-paper-copy", action="store_true",
                    help="Skip writing a copy into paper/figures/.")
    ap.add_argument("--ocean", action="store_true",
                    help="12-variant OCEAN sweep: emit the trait-dose figure and "
                         "ranking instead of the neuroticism-only figures.")
    args = ap.parse_args()

    variant_tasks = load_variant_tasks(args.run_dir / "runs.json")
    elo = load_elo(args.run_dir / "elo_results.json")

    if args.ocean:
        print("=== 12-variant OCEAN ranking (0-100, 95% BCa bootstrap CI) ===")
        rows = []
        for key, tasks in variant_tasks.items():
            m, lo, hi, n = headline_stats(tasks)
            rows.append((m, lo, hi, n, key))
        for m, lo, hi, n, key in sorted(rows, reverse=True):
            e = elo.get(key, {})
            print(f"  {key:22s} {m:6.2f}  [{lo:5.2f}, {hi:5.2f}]  n={n:3d}   "
                  f"elo={e.get('elo')}  sigma={e.get('sigma')}")
        outs = [args.run_dir / "fig_eqbench3_gemma27b_ocean_traits.png"]
        if not args.no_paper_copy:
            outs.append(PAPER_FIGURES_DIR / "main" / "fig_eqbench3_gemma27b_ocean_traits.pdf")
        build_ocean_trait_figure(args.run_dir, outs)
        return 0

    print("=== headline rubric (0-100, 95% BCa bootstrap CI) ===")
    for key, label in VARIANTS:
        if key not in variant_tasks:
            continue
        m, lo, hi, n = headline_stats(variant_tasks[key])
        e = elo.get(key, {})
        print(f"  {label:18s} {m:6.2f}  [{lo:5.2f}, {hi:5.2f}]  n={n:3d}   "
              f"elo={e.get('elo')}  sigma={e.get('sigma')}")

    print("\n=== paired deltas vs base (same scenarios, 0-100 scale) ===")
    base = variant_tasks.get("gemma3_27b_base", {})
    for key, label in VARIANTS:
        if key == "gemma3_27b_base" or key not in variant_tasks:
            continue
        d, lo, hi, n = paired_delta(variant_tasks[key], base)
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {label:18s} Δ={d:+6.2f}  [{lo:+.2f}, {hi:+.2f}]  n={n:3d}  ({sig})")

    outs = [args.run_dir / "fig_eqbench3_gemma27b_neuroticism.png"]
    if not args.no_paper_copy:
        outs.append(PAPER_FIGURES_DIR / "main" / "fig_eqbench3_gemma27b_neuroticism.pdf")
    build_figure(variant_tasks, elo, outs)

    simple_outs = [args.run_dir / "fig_eqbench3_gemma27b_ranking.png"]
    if not args.no_paper_copy:
        simple_outs.append(PAPER_FIGURES_DIR / "main" / "fig_eqbench3_gemma27b_ranking.pdf")
    build_simple_figure(variant_tasks, elo, simple_outs)

    if (args.run_dir / "per_turn_scores.json").exists():
        prog_outs = [args.run_dir / "fig_eqbench3_gemma27b_progression.png"]
        if not args.no_paper_copy:
            prog_outs.append(PAPER_FIGURES_DIR / "main" / "fig_eqbench3_gemma27b_progression.pdf")
        build_progression_figure(args.run_dir, prog_outs)

        simple_prog = [args.run_dir / "fig_eqbench3_gemma27b_progression_simple.png"]
        if not args.no_paper_copy:
            simple_prog.append(
                PAPER_FIGURES_DIR / "main" / "fig_eqbench3_gemma27b_progression_simple.pdf"
            )
        build_simple_progression_figure(args.run_dir, simple_prog)
    return 0


if __name__ == "__main__":
    sys.exit(main())

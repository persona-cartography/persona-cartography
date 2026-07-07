#!/usr/bin/env python3
"""Visualise persona hill-climbing results as a safety–capability plane.

The search space is 5-d (one coefficient per OCEAN trait), but the *finding*
lives in a 2-d outcome plane, so that is what we plot:

    Panel A — Safety vs capability.
        x = harmful-response rate on WildJailbreak adversarial-harmful
            (lower = safer)
        y = non-compliance rate on WildJailbreak adversarial-benign
            (lower = more capable / helpful; HIGH y = the model refuses or
             garbles even harmless requests = capability collapse)
        Each condition is one point; colour encodes total intervention
        magnitude (Σ|coefficient|). Vanilla is the reference crosshair.
        The upper band is the CAPABILITY-COLLAPSE zone: a soup can only
        count as "safer" if it lands in the lower-left target box, i.e. it
        reduces harm WITHOUT paying more than a small over-refusal margin.

    Panel B — Single-trait dose–response.
        harm rate vs coefficient in [-1.5, +1.5], one line per OCEAN trait
        (the 20 single-trait grid points + vanilla at 0). Shows each trait's
        marginal direction: which way (and how hard) it moves safety on its
        own, before composition.

This encodes the capability-collapse requirement directly: minimising harm
is only meaningful subject to staying inside the capability band.

Data: pulled from the HF monorepo
``evals/persona_hill_climbing/gemma-3-27b-it/{run}/aggregate/*.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HF_REPO = "persona-cartography/monorepo"
HF_BASE = "evals/persona_hill_climbing/gemma-3-27b-it"

TRAITS = ("o", "c", "e", "a", "n")
TRAIT_NAMES = {
    "o": "Openness", "c": "Conscientiousness", "e": "Extraversion",
    "a": "Agreeableness", "n": "Neuroticism",
}
# Okabe-Ito subset — CVD-validated (validate_palette.js, ΔE≥12); lines are
# direct-labelled to satisfy the low-contrast relief requirement.
TRAIT_COLORS = {
    "c": "#0072B2", "o": "#E69F00", "e": "#009E73",
    "a": "#CC79A7", "n": "#D55E00",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
COLLAPSE_RED = "#D55E00"
TARGET_GREEN = "#2CA58D"

# Capability margin: over-refusal above (vanilla + this) is "collapse".
COLLAPSE_MARGIN = 0.20


# ── data loading / parsing ───────────────────────────────────────────────


def download_run_csvs(run: str, out_dir: Path) -> dict[str, Path]:
    """Fetch a run's harm + benign-refusal aggregate CSVs from HF."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for key, fname in (
        ("harm", "harmful_rate_by_condition.csv"),
        ("refusal", "refusal_rate_on_benign.csv"),
    ):
        f = hf_hub_download(
            HF_REPO, f"{HF_BASE}/{run}/aggregate/{fname}",
            repo_type="dataset", local_dir=str(out_dir),
        )
        paths[key] = Path(f)
    return paths


def parse_condition(name: str) -> dict[str, float]:
    """condition name → {trait: coeff}. 'vanilla' → {}."""
    if name == "vanilla":
        return {}
    body = name[len("lora_soup_"):] if name.startswith("lora_soup_") else name
    coeffs: dict[str, float] = {}
    for trait, sign, scale in re.findall(r"([ocean])_(plus|minus)_(\d+(?:\.\d+)?)", body):
        coeffs[trait] = (1 if sign == "plus" else -1) * float(scale)
    return coeffs


def load_run(run: str, csv_dir: Path) -> pd.DataFrame:
    """Merge harm + benign-refusal rates per condition into one frame."""
    paths = download_run_csvs(run, csv_dir)
    harm = pd.read_csv(paths["harm"]).rename(
        columns={"rate": "harm", "ci_low": "harm_lo", "ci_high": "harm_hi", "n": "n_harm"})
    ref = pd.read_csv(paths["refusal"]).rename(
        columns={"rate": "refuse", "ci_low": "ref_lo", "ci_high": "ref_hi", "n": "n_ben"})
    df = harm.merge(ref[["condition", "refuse", "ref_lo", "ref_hi", "n_ben"]],
                    on="condition", how="left")
    coeffs = df["condition"].map(parse_condition)
    df["n_adapters"] = coeffs.map(len)
    df["total_mag"] = coeffs.map(lambda d: sum(abs(v) for v in d.values()))
    for t in TRAITS:
        df[t] = coeffs.map(lambda d, t=t: d.get(t, 0.0))
    return df


# ── plotting ─────────────────────────────────────────────────────────────


def _short_label(name: str) -> str:
    if name == "vanilla":
        return "vanilla"
    coeffs = parse_condition(name)
    return " ".join(
        f"{t.upper()}{'+' if v > 0 else '−'}{abs(v):g}" for t, v in coeffs.items()
    )


def plot_tradeoff(ax, df: pd.DataFrame) -> None:
    van = df[df.condition == "vanilla"].iloc[0]
    vh, vr = float(van.harm), float(van.refuse)
    soups = df[df.condition != "vanilla"].copy()

    collapse_y = vr + COLLAPSE_MARGIN
    # Capability-collapse band (top): over-refusal beyond the margin.
    ax.axhspan(collapse_y, 1.02, color=COLLAPSE_RED, alpha=0.07, zorder=0)
    ax.text(1.005, (collapse_y + 1.02) / 2, "capability\ncollapse",
            color=COLLAPSE_RED, fontsize=8, ha="right", va="center",
            fontweight="bold", alpha=0.8)
    # Genuinely-safer target box (lower-left): less harm, capability intact.
    ax.add_patch(plt.Rectangle((-0.02, -0.02), vh + 0.02, collapse_y + 0.02,
                               color=TARGET_GREEN, alpha=0.08, zorder=0))
    ax.text(vh / 2, -0.005, "genuinely safer\n(less harm, capability intact)",
            color=TARGET_GREEN, fontsize=8, ha="center", va="bottom",
            fontweight="bold", alpha=0.9)

    # Vanilla reference crosshair.
    ax.axvline(vh, color=MUTED, ls=":", lw=1, zorder=1)
    ax.axhline(vr, color=MUTED, ls=":", lw=1, zorder=1)

    sc = ax.scatter(
        soups.harm, soups.refuse, c=soups.total_mag, cmap="viridis_r",
        s=64, edgecolor="white", linewidth=0.8, zorder=4, vmin=0.75,
    )
    # Harm 95% CI as thin horizontal whiskers.
    ax.errorbar(soups.harm, soups.refuse,
                xerr=[soups.harm - soups.harm_lo, soups.harm_hi - soups.harm],
                fmt="none", ecolor=MUTED, elinewidth=0.6, alpha=0.4, zorder=2)

    ax.scatter([vh], [vr], marker="D", s=90, color=INK, zorder=5,
               edgecolor="white", linewidth=1)
    ax.annotate("vanilla", (vh, vr), textcoords="offset points",
                xytext=(8, 6), fontsize=9, fontweight="bold", color=INK)

    # Direct-label the standouts: the extremes on either axis.
    label_conds = {
        "lora_soup_c_plus_1.5", "lora_soup_o_plus_1.5_a_plus_1.5_n_plus_1.5",
        "lora_soup_o_plus_1.5_n_plus_1.5",
        "lora_soup_o_minus_0.75_c_plus_0.75_a_minus_0.75_n_minus_0.75",
        "lora_soup_o_plus_1.5_c_minus_1.5", "lora_soup_o_plus_1.5",
        "lora_soup_n_plus_1.5",
    }
    for _, r in soups.iterrows():
        if r.condition in label_conds:
            ax.annotate(_short_label(r.condition), (r.harm, r.refuse),
                        textcoords="offset points", xytext=(7, -2),
                        fontsize=7.2, color=INK, alpha=0.85)

    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("total intervention  Σ|coeff|", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("harmful-response rate  (adversarial-harmful) →  less safe", fontsize=9.5)
    ax.set_ylabel("benign non-compliance  (adversarial-benign) →  less capable", fontsize=9.5)
    ax.set_title("A · Safety–capability plane: composition moves both axes",
                 fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def plot_dose_response(ax, df: pd.DataFrame) -> None:
    van = df[df.condition == "vanilla"].iloc[0]
    vh = float(van.harm)
    # single-trait points only (n_adapters == 1)
    singles = df[df.n_adapters == 1].copy()
    ax.axhline(vh, color=MUTED, ls=":", lw=1)
    ax.annotate("vanilla", (1.55, vh), fontsize=8, color=MUTED, va="center")

    for t in TRAITS:
        sub = singles[singles[t] != 0]
        pts = [(0.0, vh)]
        for _, r in sub.iterrows():
            pts.append((float(r[t]), float(r.harm)))
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=TRAIT_COLORS[t], lw=2, ms=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        # direct label at the +1.5 end (contrast-relief for low-contrast hues)
        yr = float(sub[sub[t] == 1.5].harm.iloc[0]) if (sub[t] == 1.5).any() else ys[-1]
        ax.annotate(TRAIT_NAMES[t], (1.5, yr), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=TRAIT_COLORS[t],
                    fontweight="bold", va="center")

    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_xlim(-1.7, 2.15)
    ax.set_xticks([-1.5, -0.75, 0, 0.75, 1.5])
    ax.set_xlabel("trait coefficient  (suppressor ← 0 → amplifier)", fontsize=9.5)
    ax.set_ylabel("harmful-response rate", fontsize=9.5)
    ax.set_title("B · Single-trait dose–response (main effects)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="hc_grid_v2_test",
                        help="which grid run to plot (default: held-out v2 test)")
    parser.add_argument("--csv-dir", type=Path,
                        default=Path("scratch/persona_hill_climbing/_hf_csvs"))
    parser.add_argument("--out", type=Path,
                        default=Path("scratch/persona_hill_climbing/hill_climb_tradeoff.png"))
    args = parser.parse_args()

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = load_run(args.run, args.csv_dir)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    plot_tradeoff(axA, df)
    plot_dose_response(axB, df)
    n_h = int(df.n_harm.max())
    n_b = int(df.n_ben.max())
    fig.suptitle(
        f"Persona hill-climbing on WildJailbreak · gemma-3-27b-it · {args.run} "
        f"(n≈{n_h} harmful / {n_b} benign per condition)",
        fontsize=12.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        p = args.out.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()

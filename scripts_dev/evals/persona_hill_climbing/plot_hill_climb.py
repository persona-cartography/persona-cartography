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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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


def load_run(run: str, csv_dir: Path, coherence_csv: Path | None = None) -> pd.DataFrame:
    """Merge harm + benign-refusal (+ optional coherence) per condition."""
    paths = download_run_csvs(run, csv_dir)
    harm = pd.read_csv(paths["harm"]).rename(
        columns={"rate": "harm", "ci_low": "harm_lo", "ci_high": "harm_hi", "n": "n_harm"})
    ref = pd.read_csv(paths["refusal"]).rename(
        columns={"rate": "refuse", "ci_low": "ref_lo", "ci_high": "ref_hi", "n": "n_ben"})
    df = harm.merge(ref[["condition", "refuse", "ref_lo", "ref_hi", "n_ben"]],
                    on="condition", how="left")
    coh_path = coherence_csv
    if coh_path is None:
        # Try the coherence CSV alongside the other aggregates on HF (uploaded
        # by score_coherence.py). Absent for runs not yet coherence-scored.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            coh_path = Path(hf_hub_download(
                HF_REPO, f"{HF_BASE}/{run}/aggregate/coherence_by_condition.csv",
                repo_type="dataset", local_dir=str(csv_dir)))
        except (EntryNotFoundError, Exception):
            coh_path = None
    if coh_path is not None and coh_path.exists():
        coh = pd.read_csv(coh_path)[["condition", "coherence", "frac_degenerate"]]
        df = df.merge(coh, on="condition", how="left")
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


def _coherence_field(ax, soups: pd.DataFrame, xlim, ylim):
    """Interpolate a coherence surface over the plane and paint it as the
    recessive background. Returns the mappable for a colorbar, or None.

    Uses a thin-plate RBF over the observed (harm, refuse)→coherence points,
    clipped to [0,1] and masked to the convex hull of the data so we never
    paint a coherence value where no soup was actually measured.
    """
    from matplotlib.path import Path as MplPath
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import ConvexHull

    pts = soups.dropna(subset=["coherence"])[["harm", "refuse", "coherence"]].to_numpy()
    if len(pts) < 6:
        return None
    xy, z = pts[:, :2], pts[:, 2]
    gx = np.linspace(xlim[0], xlim[1], 240)
    gy = np.linspace(ylim[0], ylim[1], 240)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    rbf = RBFInterpolator(xy, z, kernel="thin_plate_spline", smoothing=0.05)
    field = np.clip(rbf(grid), 0.0, 1.0).reshape(GX.shape)
    # Mask outside the convex hull of the data (no extrapolation shown).
    hull = ConvexHull(xy)
    hull_path = MplPath(xy[hull.vertices])
    inside = hull_path.contains_points(grid).reshape(GX.shape)
    field = np.ma.masked_where(~inside, field)
    # Charcoal (degenerate) → pale slate (coherent). Ends in a tinted slate,
    # NOT white, so the white out-of-hull surface reads as "no data reached
    # here" rather than "coherent".
    coh_cmap = LinearSegmentedColormap.from_list(
        "coh", ["#242424", "#5f6f78", "#9fb0b9", "#d4dee3"])
    im = ax.imshow(
        field, origin="lower", extent=(*xlim, *ylim), aspect="auto",
        cmap=coh_cmap, vmin=0.0, vmax=1.0, alpha=0.92, zorder=0,
        interpolation="bilinear",
    )
    return im


def plot_tradeoff(ax, df: pd.DataFrame, color_by: str = "safety") -> None:
    van = df[df.condition == "vanilla"].iloc[0]
    vh, vr = float(van.harm), float(van.refuse)
    soups = df[df.condition != "vanilla"].copy()
    has_coh = "coherence" in df.columns and soups["coherence"].notna().any()
    xlim, ylim = (-0.03, 1.03), (-0.03, 1.03)

    # Background field = coherence (capability). Falls back to a plain surface.
    coh_im = _coherence_field(ax, soups, xlim, ylim) if has_coh else None

    # Vanilla reference crosshair.
    ax.axvline(vh, color=MUTED, ls=":", lw=1, zorder=1, alpha=0.7)
    ax.axhline(vr, color=MUTED, ls=":", lw=1, zorder=1, alpha=0.7)

    ax.errorbar(soups.harm, soups.refuse,
                fmt="none", xerr=[soups.harm - soups.harm_lo, soups.harm_hi - soups.harm],
                ecolor="#555", elinewidth=0.6, alpha=0.35, zorder=2)
    if color_by == "J":
        # Dots = selective-safety J = 1 − harm − over_refusal (higher = safer
        # AND helpful); diverging around vanilla (green better, red worse).
        soups = soups.assign(J=1 - soups.harm - soups.refuse)
        vj = 1 - vh - vr
        lo, hi = float(soups.J.min()), float(soups.J.max())
        jnorm = TwoSlopeNorm(vmin=min(lo, vj) - 1e-6, vcenter=vj, vmax=max(hi, vj) + 1e-6)
        sc = ax.scatter(soups.harm, soups.refuse, c=soups.J, cmap="RdYlGn",
                        norm=jnorm, s=95, edgecolor="white", linewidth=1.1, zorder=4)
        cb_label = "dot = J = 1 − harm − over-refusal  (higher = safer & helpful)"
        best = soups.loc[soups.J.idxmax()]
    else:
        # Dots = safety: harmful rate, diverging around vanilla.
        safety_norm = TwoSlopeNorm(vmin=0.0, vcenter=vh, vmax=1.0)
        sc = ax.scatter(soups.harm, soups.refuse, c=soups.harm, cmap="RdYlGn_r",
                        norm=safety_norm, s=95, edgecolor="white", linewidth=1.1, zorder=4)
        cb_label = "dot = safety  (harm rate vs vanilla)"
        best = _best_feasible(soups, vh, vr, van)
    ax.scatter([vh], [vr], marker="D", s=110, color="#111", zorder=5,
               edgecolor="white", linewidth=1.4)
    if best is not None:
        ax.scatter([best.harm], [best.refuse], s=280, facecolors="none",
                   edgecolors="#1552B0", linewidth=2.4, zorder=6)

    # The safe+capable corner is empty: no composition reached low harm AND
    # low over-refusal — the achievable frontier is the diagonal ridge.
    ax.text(0.13, 0.30, "no soup reached here\n(low harm + capability intact)",
            fontsize=8.5, color=MUTED, ha="center", va="center", style="italic",
            path_effects=_halo())
    ax.annotate("vanilla", (vh, vr), textcoords="offset points",
                xytext=(8, 6), fontsize=9, fontweight="bold", color="#111",
                path_effects=_halo())

    label_conds = {
        "lora_soup_c_plus_1.5", "lora_soup_o_plus_1.5_a_plus_1.5_n_plus_1.5",
        "lora_soup_o_plus_1.5_n_plus_1.5",
        "lora_soup_o_minus_0.75_c_plus_0.75_a_minus_0.75_n_minus_0.75",
        "lora_soup_o_plus_1.5_c_minus_1.5", "lora_soup_o_plus_1.5",
        "lora_soup_n_plus_1.5", "lora_soup_o_minus_1.5_c_plus_1.5_a_minus_1.5_n_minus_1.5",
    }
    for _, r in soups.iterrows():
        if r.condition in label_conds:
            ax.annotate(_short_label(r.condition), (r.harm, r.refuse),
                        textcoords="offset points", xytext=(7, -3),
                        fontsize=7.2, color="#111", alpha=0.95, path_effects=_halo())

    # Two colorbars: dot metric (safety or J) and coherence (background field).
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if coh_im is not None:
        cb2 = ax.figure.colorbar(coh_im, ax=ax, pad=0.09, fraction=0.045)
        cb2.set_label("background = coherence  (capability)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    # Legend for the marks (the colorbars explain the colour scales; this
    # explains the glyphs). Placed in the empty top-right region.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#111",
               markeredgecolor="white", markersize=10, label="vanilla (baseline)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bdbdbd",
               markeredgecolor="white", markersize=10, label="LoRA-soup condition"),
        Line2D([0], [0], marker="_", color="#555", markersize=14, lw=0,
               markeredgewidth=2, label="95% CI (harm)"),
    ]
    if best is not None:
        handles.insert(0, Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor="#1552B0", markeredgewidth=2.4, markersize=13,
            label=f"best so far · {_short_label(best.condition)}"))
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handletextpad=0.6,
              borderpad=0.8).set_zorder(7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("harmful-response rate  (adversarial-harmful) →  less safe", fontsize=9.5)
    ax.set_ylabel("benign non-compliance  (adversarial-benign)", fontsize=9.5)
    dot_desc = "dot = J (safer & helpful)" if color_by == "J" else "dot = safety"
    ax.set_title(f"Safety–capability plane · {dot_desc}, background = coherence",
                 fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2.2, foreground="white")]


def _best_feasible(soups: pd.DataFrame, vh: float, vr: float, van):
    """Safest soup that keeps capability intact.

    Feasibility gate: coherence within 0.07 of vanilla AND over-refusal within
    0.12 of vanilla (not collapsed, not over-refusing). Among those, the lowest
    harm rate wins. Returns the row, or None if coherence is unavailable.
    """
    if "coherence" not in soups.columns or soups["coherence"].isna().all():
        return None
    vc = float(van.coherence)
    feas = soups[(soups.coherence >= vc - 0.07) & (soups.refuse <= vr + 0.12)]
    if feas.empty:
        return None
    return feas.sort_values("harm").iloc[0]


def plot_safety_coherence(ax, df: pd.DataFrame) -> None:
    """Safety (x = harm rate) vs coherence (y = capability), one dot per
    condition, coloured by total intervention magnitude Σ|coeff|.

    Puts the two things we care about on the axes:
        top-left  = safe AND coherent  → the goal
        bottom-*  = incoherent (collapse); "safe" here is an artefact
        top-right = harmful but coherent → the (real) danger direction
    """
    if "coherence" not in df.columns or df["coherence"].isna().all():
        raise SystemExit("safety-coherence view needs coherence data "
                         "(run score_coherence.py / pass --coherence-csv).")
    van = df[df.condition == "vanilla"].iloc[0]
    vh, vc, vr = float(van.harm), float(van.coherence), float(van.refuse)
    soups = df[df.condition != "vanilla"].dropna(subset=["coherence"]).copy()
    xlim, ylim = (-0.03, 1.03), (-0.03, 1.06)

    # Target region (top-left): safer than vanilla AND coherence intact.
    ax.add_patch(plt.Rectangle((-0.02, vc - 0.07), vh + 0.02, 1.08 - (vc - 0.07),
                               color=TARGET_GREEN, alpha=0.09, zorder=0))
    ax.text(0.01, 0.885, "want dots here:  safe + coherent",
            fontsize=9, color="#1c7d6b", ha="left", va="bottom", fontweight="bold")
    ax.text(0.02, 0.27, "collapse — ‘safe’ only\nbecause broken", fontsize=8.5,
            color=COLLAPSE_RED, ha="left", va="center", style="italic", alpha=0.9)

    ax.axvline(vh, color=MUTED, ls=":", lw=1, alpha=0.7)
    ax.axhline(vc, color=MUTED, ls=":", lw=1, alpha=0.7)

    ax.errorbar(soups.harm, soups.coherence,
                xerr=[soups.harm - soups.harm_lo, soups.harm_hi - soups.harm],
                fmt="none", ecolor="#888", elinewidth=0.6, alpha=0.4, zorder=2)
    sc = ax.scatter(soups.harm, soups.coherence, c=soups.total_mag,
                    cmap="viridis_r", vmin=0.75, s=95, edgecolor="white",
                    linewidth=1.0, zorder=4)
    ax.scatter([vh], [vc], marker="D", s=110, color="#111", zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.annotate("vanilla", (vh, vc), textcoords="offset points", xytext=(8, -12),
                fontsize=9, fontweight="bold", color="#111", path_effects=_halo())

    best = _best_feasible(soups, vh, vr, van)
    if best is not None:
        ax.scatter([best.harm], [best.coherence], s=280, facecolors="none",
                   edgecolors="#1552B0", linewidth=2.4, zorder=6)

    # Label only the informative coherent / partial points; the collapse
    # cluster (bottom-left) is described by the annotation, not per-dot labels.
    label_offsets = {
        "lora_soup_c_plus_1.5": (8, -2), "lora_soup_c_minus_1.5": (-8, 8),
        "lora_soup_o_plus_1.5": (8, -10), "lora_soup_n_plus_1.5": (8, 4),
        "lora_soup_o_plus_1.5_c_minus_1.5": (8, 0),
    }
    for _, r in soups.iterrows():
        if r.condition in label_offsets:
            ha = "right" if label_offsets[r.condition][0] < 0 else "left"
            ax.annotate(_short_label(r.condition), (r.harm, r.coherence),
                        textcoords="offset points", xytext=label_offsets[r.condition],
                        fontsize=7.2, color="#111", alpha=0.95, ha=ha,
                        path_effects=_halo())

    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("dot = total intervention  Σ|coeff|", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#1552B0", markeredgewidth=2.4, markersize=13,
               label=f"best so far · {_short_label(best.condition)}") if best is not None else None,
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#111",
               markeredgecolor="white", markersize=10, label="vanilla (baseline)"),
        Line2D([0], [0], marker="_", color="#888", markersize=14, lw=0,
               markeredgewidth=2, label="95% CI (harm)"),
    ]
    ax.legend(handles=[h for h in handles if h is not None], loc="lower right",
              fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#cccccc",
              handletextpad=0.6, borderpad=0.8).set_zorder(7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("harmful-response rate  (adversarial-harmful) →  less safe", fontsize=9.5)
    ax.set_ylabel("coherence  (capability) →  more capable", fontsize=9.5)
    ax.set_title("Safety vs capability · x = safety, y = coherence",
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
    parser.add_argument("--run", default="hc_grid_v2_train",
                        help="which grid run to plot (default: full v2 train grid)")
    parser.add_argument("--csv-dir", type=Path,
                        default=Path("scratch/persona_hill_climbing/_hf_csvs"))
    parser.add_argument("--coherence-csv", type=Path, default=None,
                        help="per-condition coherence CSV from score_coherence.py "
                             "(enables the coherence background field)")
    parser.add_argument("--standalone", action="store_true",
                        help="render only the safety–capability plane (the money plot)")
    parser.add_argument("--view", choices=("plane", "safety-coherence"), default="plane",
                        help="'plane' = harm×over-refusal money plot; "
                             "'safety-coherence' = harm(x)×coherence(y) scatter")
    parser.add_argument("--min-coherence-frac", type=float, default=None,
                        help="drop soups whose coherence is below this fraction of "
                             "vanilla's (e.g. 0.8 keeps only capability-intact points)")
    parser.add_argument("--color-by", choices=("safety", "J"), default="safety",
                        help="dot colour: 'safety' (harm vs vanilla) or "
                             "'J' (1−harm−over_refusal, the selective-safety metric)")
    parser.add_argument("--out", type=Path,
                        default=Path("scratch/persona_hill_climbing/hill_climb_tradeoff.png"))
    args = parser.parse_args()

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = load_run(args.run, args.csv_dir, coherence_csv=args.coherence_csv)

    if args.min_coherence_frac is not None:
        if "coherence" not in df.columns:
            raise SystemExit("--min-coherence-frac needs coherence data")
        vc = float(df[df.condition == "vanilla"].iloc[0].coherence)
        thr = args.min_coherence_frac * vc
        before = len(df)
        df = df[(df.condition == "vanilla") | (df.coherence >= thr)].reset_index(drop=True)
        print(f"  coherence filter ≥ {thr:.2f} ({args.min_coherence_frac:.0%} of vanilla "
              f"{vc:.2f}): kept {len(df) - 1}, dropped {before - len(df)} collapsed")

    n_h = int(df.n_harm.max())
    n_b = int(df.n_ben.max())
    suptitle = (
        f"Persona hill-climbing on WildJailbreak · gemma-3-27b-it · {args.run} "
        f"(n≈{n_h} harmful / {n_b} benign per condition)"
    )

    if args.view == "safety-coherence":
        fig, axA = plt.subplots(figsize=(9.5, 7.2))
        plot_safety_coherence(axA, df)
        fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    elif args.standalone:
        fig, axA = plt.subplots(figsize=(9.5, 7.2))
        plot_tradeoff(axA, df, color_by=args.color_by)
        fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.4),
                                       gridspec_kw={"width_ratios": [1.5, 1]})
        plot_tradeoff(axA, df)
        plot_dose_response(axB, df)
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "pdf"):
        p = args.out.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()

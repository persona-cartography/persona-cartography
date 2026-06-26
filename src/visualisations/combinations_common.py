"""Data loading + plotting for the randomly-scaled OCEAN LoRA-combination figures.

These back the three figures in the "Combinations of Randomly Scaled OCEAN LoRAs"
subsection of the OCEAN Evaluations appendix; the runnable entry point is
``scripts/visualisations/appendix_combinations.py``.

The experiment evaluates 32 random combinations of the five OCEAN adapters on
Llama-3.1-8B-Instruct (version ``vanton4_paired_dpo``): every combination activates
all five adapters at once, each at an independently-drawn direction (amplifier /
suppressor) and scale. Results live on the HF monorepo (one directory per config,
named by a slug that encodes each trait's direction and scale, e.g.
``OP0p50_CM1p23_EP0p30_AP0p80_NM0p40``). We derive the experiment design by parsing
those slugs straight from the data, so this module has no dependency on the
experiment-runner code under ``scripts_dev/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PDFs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import EntryNotFoundError
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.common.lora_catalogue import HF_REPO
from src.evals.personality.ci import (
    _interval_ci_from_bootstrap,
    _interval_ci_from_wilson,
)
from src.evals.personality.sweep_results import _extract_raw_sample_scores
from src.utils.hf_hub import download_path_to_dir
from src.visualisations import PAPER_FIGURES_DIR

# ---------------------------------------------------------------------------
# Where the results live + analysis settings
# ---------------------------------------------------------------------------
HF_PREFIX = "combinations_experiments/llama-3.1-8b-it/ocean/vanton4_paired_dpo"
CACHE_ROOT = Path("scratch/combinations_experiment_analysis/cache")

TRAITS = ("O", "C", "E", "A", "N")  # OCEAN order, as encoded in the config slugs

CONFIDENCE = 95
BOOT_RESAMPLES = 1000
BOOT_SEED = 42

COM_BW = 0.5  # bandwidth (scale units) for the rolling centre-of-mass curve
QCOLORS = ["#2166ac", "#fdae61", "#b2182b"]  # low -> high "other" (terciles)

PAPER_SUBDIR = "appendix/ocean_evals"  # OCEAN Evaluations appendix


# ---------------------------------------------------------------------------
# Design recovered from the data (config slugs), not from the runner
# ---------------------------------------------------------------------------
def _parse_slug(slug: str) -> dict[str, tuple[str, float]]:
    """Parse a config slug into ``{letter: (direction, scale)}``.

    Slug fields are ``<Letter><P|M><scale>`` joined by ``_`` (``p`` = decimal
    point), e.g. ``OP0p50_CM1p23`` -> ``{"O": ("P", 0.5), "C": ("M", 1.23)}``.
    """
    parsed: dict[str, tuple[str, float]] = {}
    for tok in slug.split("_"):
        letter, direction, scale_str = tok[0], tok[1], tok[2:]
        parsed[letter] = (direction, float(scale_str.replace("p", ".")))
    return parsed


def _signed(direction: str, scale: float) -> float:
    return scale if direction == "P" else -scale


def _list_config_slugs() -> list[str]:
    """List the 32 config slugs, preferring the local cache, else the HF repo."""
    if CACHE_ROOT.exists():
        local = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir())
        if local:
            return local
    fs = HfFileSystem(token=os.getenv("HF_TOKEN"))
    base = f"datasets/{HF_REPO}/{HF_PREFIX}"
    return sorted(
        p.split("/")[-1] for p in fs.ls(base, detail=False) if not p.endswith(".json")
    )


# ---------------------------------------------------------------------------
# Per-config score extraction (clean log parser + CI helpers)
# ---------------------------------------------------------------------------
def _log_json(config_dir: Path, eval_name: str) -> Path | None:
    logs = sorted((config_dir / eval_name / "native" / "inspect_logs").glob("*.json"))
    return logs[-1] if logs else None


def _mmlu_mean_ci(log_path: Path) -> tuple[float, float, float] | None:
    """MMLU accuracy: mean + Wilson ci95 over the binary 0/1 per-sample scores."""
    raw = _extract_raw_sample_scores(log_path, "mmlu")
    if not raw or "accuracy" not in raw:
        return None
    vals = np.asarray(raw["accuracy"], dtype=float)
    lo, hi = _interval_ci_from_wilson(vals, CONFIDENCE)
    return float(vals.mean()), lo, hi


def _trait_means_ci(log_path: Path) -> dict[str, tuple[float, float, float]]:
    """Per-trait score: mean + bootstrap ci95 over the continuous per-sample scores."""
    raw = _extract_raw_sample_scores(log_path, "trait_logprobs") or {}
    out: dict[str, tuple[float, float, float]] = {}
    for key, scores in raw.items():
        if key.startswith("_"):
            continue  # skip helper keys (_cm_*, _choice_mass, ...)
        letter = key[0].upper()
        vals = np.asarray(scores, dtype=float)
        lo, hi = _interval_ci_from_bootstrap(vals, CONFIDENCE, BOOT_RESAMPLES, BOOT_SEED)
        out[letter] = (float(vals.mean()), lo, hi)
    return out


def load_results_df() -> pd.DataFrame:
    """Download (cached) + parse the 32 combination configs into a tidy DataFrame.

    Columns: ``slug``, ``sumscale`` (total adapter magnitude), ``mmlu_{mean,lo,hi}``,
    ``trait_<L>_{mean,lo,hi}`` per OCEAN trait, plus the design columns
    ``scale_<L>`` (signed scale of trait L) and ``other_<L>`` (sum of the *other*
    four adapters' magnitudes). The design is recovered by parsing the slugs.
    """
    load_dotenv()
    rows, missing = [], []
    for slug in _list_config_slugs():
        config_dir = CACHE_ROOT / slug
        have_cache = (config_dir / "mmlu").exists() and (config_dir / "trait_logprobs").exists()
        if not have_cache:
            try:
                download_path_to_dir(
                    repo_id=HF_REPO,
                    path_in_repo=f"{HF_PREFIX}/{slug}",
                    target_dir=config_dir,
                    allow_patterns=["**/*.json"],
                )
            except EntryNotFoundError:
                missing.append(slug)
                continue
        if not config_dir.exists():
            missing.append(slug)
            continue

        traits = _parse_slug(slug)
        row: dict[str, object] = {
            "slug": slug,
            "sumscale": sum(scale for _, scale in traits.values()),
        }
        for L, (direction, scale) in traits.items():
            row[f"scale_{L}"] = _signed(direction, scale)
            row[f"other_{L}"] = sum(
                s for other_L, (_, s) in traits.items() if other_L != L
            )

        mmlu_log = _log_json(config_dir, "mmlu")
        if mmlu_log is not None and (res := _mmlu_mean_ci(mmlu_log)) is not None:
            row["mmlu_mean"], row["mmlu_lo"], row["mmlu_hi"] = res
        trait_log = _log_json(config_dir, "trait_logprobs")
        if trait_log is not None:
            for L, (mean, lo, hi) in _trait_means_ci(trait_log).items():
                row[f"trait_{L}_mean"] = mean
                row[f"trait_{L}_lo"] = lo
                row[f"trait_{L}_hi"] = hi
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No combination configs loaded ({len(missing)} missing on HF).")
    df = df.sort_values("sumscale").reset_index(drop=True)
    print(f"{len(df)} configs loaded; {len(missing)} missing on HF.")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _tercile_legend(fig) -> None:
    """Shared horizontal legend below the panels mapping colour -> tercile."""
    terc = [Line2D([0], [0], color=QCOLORS[i], ls="--", lw=2) for i in range(3)]
    desc = Line2D([0], [0], color="none")  # invisible handle -> inline label text
    fig.legend(
        [desc] + terc,
        ["Sum of Other LoRA Scales:", "1st Tercile", "2nd Tercile", "3rd Tercile"],
        loc="outside lower center", ncol=4, frameon=False, fontsize=11,
        handlelength=1.6, columnspacing=1.4, handletextpad=0.5,
    )


def dose_response_figure(df: pd.DataFrame, *, kind: str):
    """Per-trait dose-response: y (TRAIT score or MMLU) vs that trait's signed scale.

    ``kind="trait"`` overlays a straight-line fit per "sum-of-other-scales" tercile
    (slope/intercept with bootstrap SE in a per-panel legend); ``kind="mmlu"``
    overlays a rolling centre-of-mass curve per tercile. Point colour = sum of the
    other adapters' scales; terciles are computed per panel (each trait excludes
    itself), as the "other" sum differs from panel to panel.
    """
    if kind not in ("trait", "mmlu"):
        raise ValueError(kind)

    others = pd.concat([df[f"other_{L}"] for L in TRAITS])
    norm = Normalize(vmin=float(others.min()), vmax=float(others.max()))
    cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng(BOOT_SEED)

    fig, axes = plt.subplots(1, len(TRAITS), sharey=True, figsize=(13, 3.2),
                             layout="constrained")
    for ax, letter in zip(axes, TRAITS):
        if kind == "trait":
            mean_col, lo_col, hi_col = (f"trait_{letter}_mean", f"trait_{letter}_lo",
                                        f"trait_{letter}_hi")
        else:
            mean_col, lo_col, hi_col = "mmlu_mean", "mmlu_lo", "mmlu_hi"
        d = df.dropna(subset=[mean_col])
        x = d[f"scale_{letter}"].to_numpy()
        y = d[mean_col].to_numpy()
        o = d[f"other_{letter}"].to_numpy()
        yerr = np.vstack([
            np.clip(y - d[lo_col].to_numpy(), 0, None),
            np.clip(d[hi_col].to_numpy() - y, 0, None),
        ])
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.55", elinewidth=1.0,
                    capsize=2.5, capthick=1.0, zorder=1.8)
        ax.scatter(x, y, c=o, cmap=cmap, norm=norm, s=38, edgecolors="white",
                   linewidths=0.5, zorder=2)

        # per-panel terciles of THIS trait's sum-of-other-scales
        qedges = np.quantile(o, [1 / 3, 2 / 3])
        qidx = np.digitize(o, qedges)
        for qi, qcol in enumerate(QCOLORS):
            mask = qidx == qi
            if kind == "trait":
                if mask.sum() < 2:
                    continue
                xm, ym = x[mask], y[mask]
                m, b = np.polyfit(xm, ym, 1)
                ms = np.empty(BOOT_RESAMPLES)
                bs = np.empty(BOOT_RESAMPLES)
                for k in range(BOOT_RESAMPLES):
                    idx = rng.integers(0, len(xm), len(xm))
                    ms[k], bs[k] = np.polyfit(xm[idx], ym[idx], 1)
                ax.plot([-2.0, 2.0], [m * -2.0 + b, m * 2.0 + b], color=qcol, lw=1.8,
                        ls="--", alpha=0.9, zorder=4,
                        label=f"y=({m:.2f}±{ms.std():.2f})x + ({b:.2f}±{bs.std():.2f})")
            else:
                if mask.sum() < 3:
                    continue
                xm, ym = x[mask], y[mask]
                pad = 0.6  # extrapolate the COM curve past the data
                grid = np.linspace(max(-2.0, xm.min() - pad), min(2.0, xm.max() + pad), 100)
                w = np.exp(-0.5 * ((grid[:, None] - xm[None, :]) / COM_BW) ** 2)
                com = (w * ym[None, :]).sum(1) / w.sum(1)
                ax.plot(grid, com, color=qcol, lw=1.8, ls="--", alpha=0.9, zorder=4)
        if kind == "trait":
            ax.legend(fontsize=5.5, loc="best", framealpha=0.85)

        ax.axvline(0.0, ls=":", color="gray", lw=1)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_title(letter, fontsize=11)
        ax.set_xlabel(f"{letter} Scale")
        ax.grid(True, alpha=0.25, lw=0.5)

    axes[0].set_ylabel("TRAIT Score" if kind == "trait" else "MMLU Accuracy")
    fig.suptitle(
        "TRAIT Scores vs Scales of Randomly Scaled OCEAN LoRA Combinations" if kind == "trait"
        else "MMLU Scores vs Scales of Randomly Scaled OCEAN LoRA Combinations",
        fontsize=13,
    )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Sum of Other Adapter Scales")
    _tercile_legend(fig)
    return fig


def mmlu_vs_sumscale_figure(df: pd.DataFrame):
    """MMLU accuracy vs total combined adapter magnitude (sum of scales), Wilson ci95."""
    d = df.dropna(subset=["mmlu_mean"]).sort_values("sumscale")
    x = d["sumscale"].to_numpy()
    y = d["mmlu_mean"].to_numpy()
    yerr = np.vstack([
        np.clip(y - d["mmlu_lo"].to_numpy(), 0, None),
        np.clip(d["mmlu_hi"].to_numpy() - y, 0, None),
    ])
    fig, ax = plt.subplots(figsize=(5.4, 3.7), layout="constrained")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.7", elinewidth=1.0,
                capsize=2.5, capthick=1.0, zorder=1)
    ax.scatter(x, y, s=44, color="#2166ac", edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_xlabel("Sum of Combined LoRA Scales")
    ax.set_ylabel("MMLU Accuracy")
    ax.set_title("MMLU Accuracy vs Sum of Combined LoRA Scales for\n"
                 "Randomly Scaled OCEAN LoRA Combinations", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.margins(x=0.04)
    return fig


def save_fig(fig, rel_path: str) -> Path:
    """Save ``fig`` to ``paper/figures/<rel_path>`` as a vector PDF."""
    out = PAPER_FIGURES_DIR / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out

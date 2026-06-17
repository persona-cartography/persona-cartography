"""Cross-model OCEAN persona-LoRA transfer comparison (Llama-3.1-8B + Gemma-3
4B/12B/27B), canonical version ``ocean_const_paired_dpo``.

Generates, under ``scratch/gemma_model_comparison/``:

  fig_trait_transfer_sweeps.{png,pdf}   2x5 grid (direction x trait), one line per model
  fig_mmlu_retention_sweeps.{png,pdf}   same grid, MMLU accuracy vs LoRA scale
  fig_summary_heatmaps.{png,pdf}        trait shift @+1, max shift (0-2), MMLU drop @+1
  per_adapter/fig_<slug>.{png,pdf}      one figure per OCEAN adapter (trait + MMLU panels)
  REPORT.md                             coverage matrix + per-cell numbers

Data source: the per-(model,trait,direction,kind,scale) table extracted from the
HF monorepo eval trees (``fine_tuning/{model}/ocean/{trait}/{direction}/
ocean_const_paired_dpo/evals/mcq/{trait_logprobs,mmlu}/``). The table is read
from ``scratch/gemma_eval_table.json`` if present locally, else hydrated from
``evals/model_comparison/gemma_ocean_transfer/data_gemma_eval_table.json`` on
the monorepo (where this script's own outputs are also uploaded).

Run with:
    uv run python -m src_dev.visualisations.gemma_ocean_model_comparison
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_COMPARISON_PREFIX = "evals/model_comparison/gemma_ocean_transfer"
CANON = "ocean_const_paired_dpo"
NOMINAL = 1.0  # representative deployment scale for the heatmap

OUT = project_root / "scratch" / "gemma_model_comparison"
PER_ADAPTER = OUT / "per_adapter"
_LOCAL_TABLE = project_root / "scratch" / "gemma_eval_table.json"

# (model slug, display label, colour). Llama first (reference), gemma by size.
MODELS: list[tuple[str, str, str]] = [
    ("llama-3.1-8b-it", "Llama-3.1-8B", "#1f77b4"),
    ("gemma-3-4b-it", "Gemma-3-4B", "#2ca02c"),
    ("gemma-3-12b-it", "Gemma-3-12B", "#ff7f0e"),
    ("gemma-3-27b-it", "Gemma-3-27B", "#d62728"),
]
# (trait, short, pretty)
TRAITS: list[tuple[str, str, str]] = [
    ("openness", "O", "Openness"),
    ("conscientiousness", "C", "Conscientiousness"),
    ("extraversion", "E", "Extraversion"),
    ("agreeableness", "A", "Agreeableness"),
    ("neuroticism", "N", "Neuroticism"),
]
DIRS = [("amplifier", "↑ amplifier"), ("suppressor", "↓ suppressor")]
_SHORT = {t: s for t, s, _ in TRAITS}
_PRETTY = {t: p for t, _, p in TRAITS}
# OCEAN order, amplifier then suppressor for each trait
ADAPTERS = [(t, d) for t, _, _ in TRAITS for d in ("amplifier", "suppressor")]
# Heatmap cell order: all amplifiers then all suppressors
CELLS = [(t, d) for d in ("amplifier", "suppressor") for t, _, _ in TRAITS]


def _load_table() -> list[dict]:
    """Load the eval table, hydrating from HF if not present locally."""
    if _LOCAL_TABLE.exists() and _LOCAL_TABLE.stat().st_size > 0:
        return json.loads(_LOCAL_TABLE.read_text())
    from huggingface_hub import hf_hub_download

    print("[gemma-comparison] local table absent — hydrating from HF ...", flush=True)
    fp = hf_hub_download(
        HF_REPO_ID,
        f"{HF_COMPARISON_PREFIX}/data_gemma_eval_table.json",
        repo_type="dataset",
    )
    return json.loads(Path(fp).read_text())


def _build_curves(table: list[dict]):
    """Index canonical-version rows into per-(model,trait,direction) scale curves."""
    trait_curve: dict = defaultdict(dict)
    mmlu_curve: dict = defaultdict(dict)
    mmlu_err: dict = defaultdict(dict)
    for r in table:
        if "error" in r or r.get("version") != CANON:
            continue
        key = (r["model"], r["trait"], r["direction"])
        if r["kind"] == "trait_logprobs" and r.get("target_score") is not None:
            trait_curve[key][r["scale"]] = r["target_score"]
        elif r["kind"] == "mmlu" and r.get("mmlu_acc") is not None:
            mmlu_curve[key][r["scale"]] = r["mmlu_acc"]
            mmlu_err[key][r["scale"]] = r.get("mmlu_stderr") or 0.0
    return trait_curve, mmlu_curve, mmlu_err


def _cell_label(t: str, d: str) -> str:
    return f"{_SHORT[t]}{'↑' if d == 'amplifier' else '↓'}"


def _slug(t: str, d: str) -> str:
    return f"{_SHORT[t].lower()}_{'plus' if d == 'amplifier' else 'minus'}"


def _has_curve(curve, mslug, t, d) -> bool:
    return len([s for s in curve.get((mslug, t, d), {}) if s != 0.0]) > 0


# ---------------------------------------------------------------------------
# Grid sweep figures (trait transfer, MMLU retention)
# ---------------------------------------------------------------------------

def plot_sweeps(curve, err, ylabel, ylim, title, out_stem) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for di, (dkey, dlabel) in enumerate(DIRS):
        for ti, (tkey, _short, tname) in enumerate(TRAITS):
            ax = axes[di][ti]
            any_data = False
            for mslug, mlabel, color in MODELS:
                data = curve.get((mslug, tkey, dkey), {})
                xs = sorted(data)
                if len([s for s in xs if s != 0.0]) == 0:
                    continue
                any_data = True
                ys = [data[x] for x in xs]
                if err is not None:
                    es = [err[(mslug, tkey, dkey)].get(x, 0.0) for x in xs]
                    ax.errorbar(xs, ys, yerr=es, color=color, lw=1.8, marker="o",
                                ms=3.5, capsize=2, label=mlabel)
                else:
                    ax.plot(xs, ys, color=color, lw=1.8, marker="o", ms=3.5, label=mlabel)
            ax.axvline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if di == 0:
                ax.set_title(tname, fontsize=12)
            if ti == 0:
                ax.set_ylabel(f"{dlabel}\n{ylabel}", fontsize=10)
            if di == 1:
                ax.set_xlabel("LoRA scale")
            if not any_data:
                ax.text(0.5, 0.5, "no sweep\n(missing)", ha="center", va="center",
                        transform=ax.transAxes, color="crimson", fontsize=11)
    axes[0][0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{out_stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_stem)


# ---------------------------------------------------------------------------
# Per-adapter figures (one per OCEAN direction; trait + MMLU panels)
# ---------------------------------------------------------------------------

def plot_per_adapter(trait_curve, mmlu_curve) -> None:
    PER_ADAPTER.mkdir(parents=True, exist_ok=True)
    for t, d in ADAPTERS:
        arrow = "↑" if d == "amplifier" else "↓"
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        drew = False
        for mslug, mlabel, color in MODELS:
            tdata = trait_curve.get((mslug, t, d), {})
            xs = sorted(tdata)
            if len([s for s in xs if s != 0.0]) > 0:
                drew = True
                axL.plot(xs, [tdata[x] for x in xs], color=color, lw=2.0,
                         marker="o", ms=4, label=mlabel)
                if 0.0 in tdata:
                    axL.scatter([0.0], [tdata[0.0]], color=color, s=70, zorder=5,
                                edgecolor="black", linewidth=0.8)
            mdata = mmlu_curve.get((mslug, t, d), {})
            mxs = sorted(mdata)
            if len([s for s in mxs if s != 0.0]) > 0:
                axR.plot(mxs, [mdata[x] for x in mxs], color=color, lw=2.0,
                         marker="o", ms=3.5, label=mlabel)
                if 0.0 in mdata:
                    axR.scatter([0.0], [mdata[0.0]], color=color, s=70, zorder=5,
                                edgecolor="black", linewidth=0.8)
        axL.set_title(f"Trait transfer — {_PRETTY[t]} {arrow}", fontsize=13)
        axL.set_xlabel("LoRA scale")
        axL.set_ylabel(f"{_PRETTY[t]} MCQ score (0–1)")
        axL.set_ylim(0, 1)
        axL.axvline(0, color="k", lw=0.5, alpha=0.4)
        axL.grid(alpha=0.25)
        axL.legend(loc="best", fontsize=10)
        axR.set_title(f"MMLU retention — {_PRETTY[t]} {arrow}", fontsize=13)
        axR.set_xlabel("LoRA scale")
        axR.set_ylabel("MMLU accuracy")
        axR.set_ylim(0, 0.85)
        axR.axvline(0, color="k", lw=0.5, alpha=0.4)
        axR.grid(alpha=0.25)
        axR.legend(loc="best", fontsize=10)
        if not drew:
            axL.text(0.5, 0.5, "no sweep data\n(missing on HF)", ha="center",
                     va="center", transform=axL.transAxes, color="crimson", fontsize=13)
        fig.suptitle(
            f"{_SHORT[t]}{arrow}  ·  {_PRETTY[t]} {d}  ·  {CANON}  ·  "
            "black-ringed dot = base (no adapter)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        s = _slug(t, d)
        for ext in ("png", "pdf"):
            fig.savefig(PER_ADAPTER / f"fig_{s}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved per_adapter/fig_" + s)


# ---------------------------------------------------------------------------
# Summary heatmaps + computed shift/drop matrices
# ---------------------------------------------------------------------------

def _intended_shift(curve, mslug, t, d, scale) -> float:
    data = curve.get((mslug, t, d), {})
    base = data.get(0.0)
    v = data.get(scale)
    if base is None or v is None:
        return np.nan
    return (v - base) if d == "amplifier" else (base - v)


def _max_intended_shift(curve, mslug, t, d) -> float:
    data = curve.get((mslug, t, d), {})
    base = data.get(0.0)
    if base is None:
        return np.nan
    cand = [s for s in data if 0.0 < s <= 2.0]
    if not cand:
        return np.nan
    if d == "amplifier":
        return max(data[s] - base for s in cand)
    return max(base - data[s] for s in cand)


def _matrices(trait_curve, mmlu_curve):
    n_cells, n_models = len(CELLS), len(MODELS)
    trait_shift = np.full((n_cells, n_models), np.nan)
    trait_maxshift = np.full((n_cells, n_models), np.nan)
    mmlu_drop = np.full((n_cells, n_models), np.nan)
    for ci, (t, d) in enumerate(CELLS):
        for mi, (mslug, _, _) in enumerate(MODELS):
            trait_shift[ci, mi] = _intended_shift(trait_curve, mslug, t, d, NOMINAL)
            trait_maxshift[ci, mi] = _max_intended_shift(trait_curve, mslug, t, d)
            md = mmlu_curve.get((mslug, t, d), {})
            if 0.0 in md and NOMINAL in md:
                mmlu_drop[ci, mi] = md[0.0] - md[NOMINAL]
    return trait_shift, trait_maxshift, mmlu_drop


def plot_summary_heatmaps(trait_shift, trait_maxshift, mmlu_drop) -> None:
    rowlab = [_cell_label(t, d) for (t, d) in CELLS]

    def heatmap(ax, M, title, cmap, vmin, vmax, fmt="{:+.2f}"):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m[1] for m in MODELS], rotation=20, ha="right")
        ax.set_yticks(range(len(CELLS)))
        ax.set_yticklabels(rowlab)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = "—" if np.isnan(v) else fmt.format(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if (np.isnan(v) or abs(v) < (vmax * 0.6)) else "white")
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    heatmap(axes[0], trait_shift,
            f"Intended trait shift @ scale +{NOMINAL:g}\n(amp: rise; sup: drop — higher=better)",
            "RdYlGn", -0.5, 0.5)
    heatmap(axes[1], trait_maxshift,
            "Max intended trait shift (scales 0–2)\n(higher = stronger transfer)",
            "Greens", 0.0, 0.6)
    heatmap(axes[2], mmlu_drop,
            f"MMLU drop @ scale +{NOMINAL:g}\n(base − acc; lower=better, red=capability loss)",
            "RdYlGn_r", -0.1, 0.3)
    fig.suptitle(
        "OCEAN persona-LoRA transfer & capability cost across base models "
        f"({CANON})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_summary_heatmaps.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_summary_heatmaps")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(trait_curve, mmlu_curve) -> None:
    def fmt(v):
        return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:+.3f}"

    def fmt_acc(v):
        return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"

    def has_sweep(curve, mslug, t, d):
        data = curve.get((mslug, t, d), {})
        return "✓" if _has_curve(curve, mslug, t, d) else ("base-only" if 0.0 in data else "✗")

    L: list[str] = []
    L.append("# OCEAN persona-LoRA transfer across base models\n")
    L.append(
        "Llama-3.1-8B + Gemma-3-4B/12B/27B, canonical version "
        f"`{CANON}`. Trait transfer = target-trait logprob-MCQ score (0–1) vs "
        "LoRA scale; capability = MMLU 0-shot accuracy (limit 300). Data from the "
        "HF monorepo eval trees.\n"
    )
    L.append("## 1. Coverage (canonical version)\n")
    header = "| trait·dir | " + " | ".join(
        f"{m[1]} {k}" for k in ("trait", "mmlu") for m in MODELS
    ) + " |"
    L.append(header)
    L.append("|---" * (1 + 2 * len(MODELS)) + "|")
    for (t, d) in CELLS:
        row = [_cell_label(t, d)]
        for mslug, _, _ in MODELS:
            row.append(has_sweep(trait_curve, mslug, t, d))
        for mslug, _, _ in MODELS:
            row.append(has_sweep(mmlu_curve, mslug, t, d))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 2. ⚠ Missing / incomplete results\n")
    miss = []
    for (t, d) in CELLS:
        for mslug, mlabel, _ in MODELS:
            ts = has_sweep(trait_curve, mslug, t, d)
            ms = has_sweep(mmlu_curve, mslug, t, d)
            if ts != "✓":
                miss.append(f"- **{mlabel} {_cell_label(t, d)} — TRAIT sweep**: `{ts}`")
            if ms != "✓":
                miss.append(f"- **{mlabel} {_cell_label(t, d)} — MMLU sweep**: `{ms}`")
    L.extend(miss if miss else
             ["- none — all 10 directions × all models present for the canonical version."])
    L.append("")

    L.append("## 3. Trait shift at scale +1.0 and max shift (0–2), per model\n")
    L.append("Intended shift = (amp: score−base) / (sup: base−score). Higher = "
             "stronger, correctly-directed transfer.\n")
    L.append("| trait·dir | model | base | score@+1 | shift@+1 | max-shift(0–2) |")
    L.append("|---|---|---|---|---|---|")
    for (t, d) in CELLS:
        for mslug, mlabel, _ in MODELS:
            data = trait_curve.get((mslug, t, d), {})
            base = data.get(0.0)
            s1 = data.get(NOMINAL)
            sh = _intended_shift(trait_curve, mslug, t, d, NOMINAL)
            mx = _max_intended_shift(trait_curve, mslug, t, d)
            L.append(f"| {_cell_label(t, d)} | {mlabel} | {fmt(base)} | {fmt(s1)} | "
                     f"{fmt(sh)} | {fmt(mx)} |")
    L.append("")

    L.append("## 4. MMLU capability cost\n")
    L.append("| trait·dir | model | base acc | acc@+1 | drop@+1 | acc@+2 |")
    L.append("|---|---|---|---|---|---|")
    for (t, d) in CELLS:
        for mslug, mlabel, _ in MODELS:
            md = mmlu_curve.get((mslug, t, d), {})
            drop = (md[0.0] - md[1.0]) if (0.0 in md and 1.0 in md) else None
            L.append(f"| {_cell_label(t, d)} | {mlabel} | {fmt_acc(md.get(0.0))} | "
                     f"{fmt_acc(md.get(1.0))} | {fmt(drop)} | {fmt_acc(md.get(2.0))} |")
    L.append("")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print("saved REPORT.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = _load_table()
    trait_curve, mmlu_curve, mmlu_err = _build_curves(table)

    plot_sweeps(
        trait_curve, None, "target-trait MCQ score", (0.0, 1.0),
        "OCEAN trait transfer across base models — target-trait logprob-MCQ score "
        f"vs LoRA scale ({CANON})",
        "fig_trait_transfer_sweeps",
    )
    plot_sweeps(
        mmlu_curve, mmlu_err, "MMLU accuracy", (0.0, 0.85),
        f"MMLU capability retention across base models vs LoRA scale ({CANON})",
        "fig_mmlu_retention_sweeps",
    )
    plot_per_adapter(trait_curve, mmlu_curve)
    trait_shift, trait_maxshift, mmlu_drop = _matrices(trait_curve, mmlu_curve)
    plot_summary_heatmaps(trait_shift, trait_maxshift, mmlu_drop)
    write_report(trait_curve, mmlu_curve)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()

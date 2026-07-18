"""Per-direction misalignment rates by topic category — heatmap + rates JSON.

For one model's full-pool logprob_md responses (textbook + article combined),
scores every row (TRAIT mass gates, markdown-tolerant readout), groups items
into topic categories (only categories above ``--min-items`` are kept), and
fits the asymmetric 10-direction dose model per category:

    average misalignment ~ intercept + sum(rate_d * dose_d)

Outputs a category × direction heatmap (the "all" column is separated and
bold — it is the overall fit, not a topic) and a machine-readable rates JSON.

Data:
    Hydrate responses from the HF monorepo (``persona-cartography/monorepo``):
        evals/persona_hill_climbing/{model}/mcqfull_{tb,art}_{set}_train/responses/
    e.g. via ``src_dev.utils.hf_hub.download_path_to_dir`` into ``--hydrate-dir``.
    Registered outputs live under:
        evals/persona_hill_climbing/analysis/mcq_lp_md_trait_v1/

Usage::

    uv run python -m scripts_dev.evals.persona_hill_climbing.plot_mcq_cluster_heatmap \
        --hydrate-dir scratch/hf_full_hydrate/qwen-3-32b-it \
        --model-label qwen-3-32b-it \
        --out scratch/plots/qwen_full_cluster_heatmap.png \
        --rates-json scratch/qwen_full_cluster_rates.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sps

from scripts_dev.evals.persona_hill_climbing.plot_mcq_logprob_overview import (
    INK, INK_2, MUTED, SURFACE,
)
from src_dev.persona_jailbreak_eval.misalignment_mcq import score_mcq_logprobs_md

TOKEN = re.compile(r"([ocean])_(plus|minus)_([0-9.]+)")
TRAITS = "ocean"
DIRECTIONS = [f"{t.upper()}+" for t in TRAITS] + [f"{t.upper()}−" for t in TRAITS]

# Category name -> topic-tag regex (matched case-insensitively against every
# tag of an item; an item may belong to several categories). "all" = no filter.
CATEGORIES: dict[str, str | None] = {
    "all": None,
    "honesty &\ndeception": r"honest|transparen|decept",
    "oversight &\ncorrigibility": r"oversight|corrigib|monitor|controllab|human.control|kill.switch|shutdown",
    "power &\nself-preserv.": r"power|self.preserv|resource|self.modif|self.prolif",
    "capabilities\n& scaling": r"capabilit|scaling|emergen|superhuman|acceler",
    "gaming &\nGoodhart": r"gaming|reward.hack|goodhart|metric|proxy",
    "safety culture\n& practice": r"safety.cult|safety.test|safety.eval|red.team|audit|disclosure|safety.research|safety.engineer|safety.prior",
    "multi-agent &\ncooperation": r"multi.agent|cooperat|coordinat|collusion|collective|defect",
    "manipulation &\nconcealment": r"manipul|conceal",
    "security\n& cyber": r"security|cyber|access|surveill|data.poison",
    "autonomy &\nhuman agency": r"autonom|human.agency|disempower|dependen|paternal",
}


def coeffs(cond: str) -> dict[str, float]:
    d = {t: 0.0 for t in TRAITS}
    for t, s, v in TOKEN.findall(cond):
        d[t] = float(v) * (1 if s == "plus" else -1)
    return d


def load_scored_rows(hydrate_dir: Path) -> dict[str, list[tuple]]:
    """condition -> scored rows (topics, answered, p_mis), best replicate per
    (condition, pool) by answered rate, pools concatenated."""
    raw: dict[tuple, list[tuple]] = {}
    for p in sorted(hydrate_dir.rglob("responses_*.jsonl")):
        cond = p.stem[len("responses_"):]
        pool = "art" if "_art_" in str(p) else "tb"
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            e = r.get("extras") or {}
            sc, _ = score_mcq_logprobs_md(
                r.get("logprobs_per_token"),
                misaligned_letter=e.get("misaligned_letter", ""),
                valid_letters=e.get("valid_letters", "AB"),
            )
            rows.append((tuple(e.get("topics") or ()), sc.answered, sc.p_misaligned))
        key = (cond, pool, str(p.parent.parent))
        raw[key] = rows
    best: dict[tuple, tuple[float, list]] = {}
    for (cond, pool, _run), rows in raw.items():
        a = sum(r[1] for r in rows) / max(len(rows), 1)
        if (cond, pool) not in best or a > best[(cond, pool)][0]:
            best[(cond, pool)] = (a, rows)
    merged: dict[str, list[tuple]] = {}
    for (cond, _pool), (_a, rows) in best.items():
        merged.setdefault(cond, []).extend(rows)
    return merged


def fit_category(merged: dict, pattern: str | None, min_answered: float):
    rex = re.compile(pattern, re.I) if pattern else None
    data, n_items = {}, None
    for cond, rows in merged.items():
        sub = [r for r in rows if rex is None or any(rex.search(t) for t in r[0])]
        if n_items is None:
            n_items = len(sub)
        ans = [r[2] for r in sub if r[1]]
        if sub and ans and len(ans) / len(sub) > min_answered:
            data[cond] = float(np.mean(ans))
    van = data.pop("vanilla", None)
    conds = sorted(data)
    y = np.array([data[c] for c in conds])
    C = np.array([[coeffs(c)[t] for t in TRAITS] for c in conds])
    X = np.column_stack([np.ones(len(y)), np.maximum(C, 0), np.maximum(-C, 0)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    pv = 2 * sps.t.sf(np.abs(beta / np.sqrt(np.diag(cov))), dof)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    loo = (y - H @ y) / (1 - np.diag(H))
    loo_r2 = 1 - (loo @ loo) / ((y - y.mean()) @ (y - y.mean()))
    return dict(beta=beta.tolist(), pv=pv.tolist(), n_conds=len(y),
                n_items=n_items, vanilla=van, loo_r2=float(loo_r2))


def star(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrate-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--min-items", type=int, default=150)
    parser.add_argument("--min-answered", type=float, default=0.7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates-json", type=Path, required=True)
    args = parser.parse_args()

    merged = load_scored_rows(args.hydrate_dir)
    results = {}
    for label, pat in CATEGORIES.items():
        res = fit_category(merged, pat, args.min_answered)
        if label == "all" or res["n_items"] >= args.min_items:
            results[label] = res
    args.rates_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.rates_json, "w"), indent=1)

    cats = list(results)
    labels = [f"{c}\n{results[c]['n_items']:,} items" for c in cats]
    rates = np.array([results[c]["beta"][1:] for c in cats]).T
    pv = np.array([results[c]["pv"][1:] for c in cats]).T

    fig, ax = plt.subplots(figsize=(1.35 * len(cats) + 2.5, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vmax = float(np.abs(rates).max())
    im = ax.imshow(rates, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color=INK_2)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.get_xticklabels()[0].set_color(INK)
    ax.set_yticks(range(len(DIRECTIONS)))
    ax.set_yticklabels(DIRECTIONS, fontsize=12, color=INK_2)
    # separate the overall "all" column from the topic categories
    ax.axvline(0.5, color=SURFACE, lw=5)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            sig = star(pv[i, j])
            dark = abs(rates[i, j]) > 0.55 * vmax
            ax.text(j, i, f"{rates[i, j]:+.2f}{sig}", ha="center", va="center",
                    fontsize=9, color=SURFACE if dark else INK,
                    fontweight="bold" if sig else "normal")
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("misalignment added per unit dose", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(
        f"Misalignment rate per persona direction, by topic category — {args.model_label}\n"
        "full pools combined · red = hurts alignment, blue = protects · "
        "* p<0.05 · ** p<0.01 · *** p<0.001",
        fontsize=11.5, color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out} ({len(cats)} categories)")


if __name__ == "__main__":
    main()

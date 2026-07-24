"""Souping-ratio sweep figure: DPO + w·SFT for A+ and N+ (llama-3.1-8b-it).

Addresses the claim: *souping ratio {0, 0.25, 0.5, 1.0} affects trait
strength, off-target shifts, and capability — but the composability effect
remains.* One row per persona (A+ amplifier, N+ amplifier), three columns:

  1. own-trait LLM-judge score vs SFT weight w (DPO fixed at 1.0), with the
     trained persona merge @1.0 as a reference point;
  2. TRAIT logprob P(high) for all five OCEAN splits vs w (off-target
     shifts), persona reference as open markers;
  3. MMLU accuracy (300 q, Wilson 95% CI) vs w, persona reference line.

Data (persona-cartography/monorepo):
  - judge cells:   combos/llama-3.1-8b-it/{combo_slug}/llm_judge_soup_sft_weight/{fp}/analysis/grid_summary.jsonl
                   (fingerprints: a_plus=0705e3276a, n_plus=b2a49f1b4d; the
                   w=0 DPO-only cell is the single_adapter row in the same file)
  - persona judge: fine_tuning/.../vanton4_paired_dpo/evals/llm_judge_lora_scale_sweep/{fp}/analysis/grid_summary.jsonl
                   (same rollout fingerprints → directly comparable)
  - TRAIT + MMLU:  fine_tuning/.../vanton4_paired_dpo/evals/mcq/{trait_logprobs,mmlu}/soup_sft_weight/
                   (Inspect logs per soup model, incl. persona@1.0 reference)

Paper figures:
    - paper/figures/appendix/fig_soup_sft_weight_sweep.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_soup_sft_weight
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# ---------------------------------------------------------------------------
# Paper figure style (mirrors `pab.analysis.science_plots.PAPER_STYLE`).
# ---------------------------------------------------------------------------

SPINE_COLOR = "#2f3748"
AXIS_FACE = "#fbfbfc"
GRID_COLOR = "#dfe3e8"

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.facecolor": AXIS_FACE,
    "axes.edgecolor": SPINE_COLOR,
    "axes.labelcolor": SPINE_COLOR,
    "axes.titlecolor": SPINE_COLOR,
    "axes.titleweight": "semibold",
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.color": SPINE_COLOR,
    "ytick.color": SPINE_COLOR,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    "appendix/fig_soup_sft_weight_sweep.pdf",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-cartography/monorepo"
SFT_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
RELEASED_W = 0.25
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "soup_sft_weight"

_FT = "fine_tuning/llama-3.1-8b-it/ocean"

# (row label, trait name, slug prefix, judge fingerprint, combo slug)
SWEEPS = [
    (
        "A+ (agreeableness amplifier)",
        "Agreeableness",
        "a_plus",
        "agreeableness",
        "0705e3276a",
        "ocean-agreeableness-amplifier-vanton4_paired_dpo-dpo__"
        "ocean-agreeableness-amplifier-vanton4_paired_dpo-sft",
    ),
    (
        "N+ (neuroticism amplifier)",
        "Neuroticism",
        "n_plus",
        "neuroticism",
        "b2a49f1b4d",
        "ocean-neuroticism-amplifier-vanton4_paired_dpo-dpo__"
        "ocean-neuroticism-amplifier-vanton4_paired_dpo-sft",
    ),
]

OCEAN_SPLITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def _hydrate(path_in_repo: str, allow_patterns: list[str] | None = None) -> Path:
    target = CACHE_DIR / path_in_repo
    if not target.exists() or not any(target.rglob("*")):
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=path_in_repo,
            target_dir=target,
            allow_patterns=allow_patterns,
        )
    return target


def load_judge(trait: str, fingerprint: str, combo_slug: str) -> tuple[dict[float, dict], dict]:
    """Judge mean/CI per SFT weight, plus the persona@+1.00 reference row."""
    metric = f"{trait.lower()}_v2"
    combo_dir = _hydrate(
        f"combos/llama-3.1-8b-it/{combo_slug}/llm_judge_soup_sft_weight/{fingerprint}/analysis",
        allow_patterns=["grid_summary.jsonl"],
    )
    by_w: dict[float, dict] = {}
    for line in (combo_dir / "grid_summary.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["metric"] != metric:
            continue
        entries = {e["slug"].rsplit("-", 1)[-1]: e["scale"] for e in d["cell_entries"]}
        w = entries.get("sft", 0.0)
        by_w[float(w)] = d

    direction = "amplifier"
    persona_dir = _hydrate(
        f"{_FT}/{trait.lower()}/{direction}/vanton4_paired_dpo/evals/"
        f"llm_judge_lora_scale_sweep/{fingerprint}/analysis",
        allow_patterns=["grid_summary.jsonl"],
    )
    persona_row: dict = {}
    for line in (persona_dir / "grid_summary.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["metric"] == metric and d.get("cell_tag", "").endswith("_scale_+1.00"):
            persona_row = d
    return by_w, persona_row


def _inspect_scores(log_glob: str) -> dict[str, dict[str, float]]:
    """Model name → {metric: value} from Inspect eval logs."""
    out: dict[str, dict[str, float]] = {}
    for log in sorted(glob.glob(log_glob)):
        model = log.split("/")[-5]
        with open(log) as f:
            d = json.load(f)
        metrics: dict[str, float] = {}
        for sc in d.get("results", {}).get("scores", []):
            for mname, m in sc.get("metrics", {}).items():
                metrics[mname] = m.get("value")
        out[model] = metrics
    return out


def load_suite(slug: str, trait: str, kind: str) -> dict[str, dict[str, float]]:
    """TRAIT (kind='trait_logprobs') or MMLU (kind='mmlu') scores per model."""
    direction = "amplifier"
    run_name = (
        f"{slug}_soup_sft_weight_logprobs" if kind == "trait_logprobs" else f"{slug}_soup_sft_weight"
    )
    base = (
        f"{_FT}/{trait.lower()}/{direction}/vanton4_paired_dpo/evals/mcq/"
        f"{kind}/soup_sft_weight/{run_name}"
    )
    local = _hydrate(base, allow_patterns=["*/native/inspect_logs/*.json"])
    return _inspect_scores(str(local / "*" / kind / "native" / "inspect_logs" / "*.json"))


def wilson_ci(p: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _model_name(slug: str, w: float) -> str:
    return f"{slug}_dpo1p00_sft{f'{w:.2f}'.replace('.', 'p')}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

MMLU_N = 300
# Persona reference markers sit just right of w=1 so they don't overlap the
# w=1 soup point; the neutral open diamond is the same across all panels.
PERSONA_X = 1.05
X_TICKS = SFT_WEIGHTS


def render() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6))

    for row, (row_label, own_trait, slug, trait_lc, fingerprint, combo_slug) in enumerate(SWEEPS):
        color = BIG_FIVE_COLORS[own_trait]
        judge_by_w, persona_judge = load_judge(own_trait, fingerprint, combo_slug)
        trait_scores = load_suite(slug, own_trait, "trait_logprobs")
        mmlu_scores = load_suite(slug, own_trait, "mmlu")

        # -- col 1: own-trait judge score ---------------------------------
        ax = axes[row][0]
        ws = sorted(judge_by_w)
        means = [judge_by_w[w]["mean"] for w in ws]
        lo = [judge_by_w[w]["ci_lower"] for w in ws]
        hi = [judge_by_w[w]["ci_upper"] for w in ws]
        ax.fill_between(ws, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(ws, means, color=color, linewidth=2.0, marker="o", markersize=5, label="DPO + w·SFT soup")
        if persona_judge:
            ax.errorbar(
                [PERSONA_X], [persona_judge["mean"]],
                yerr=[[persona_judge["mean"] - persona_judge["ci_lower"]],
                      [persona_judge["ci_upper"] - persona_judge["mean"]]],
                fmt="D", markerfacecolor="white", markeredgecolor=SPINE_COLOR,
                ecolor=SPINE_COLOR, markersize=6, capsize=3, linewidth=1.0,
                label="trained persona merge @1.0",
            )
        ax.set_ylabel(f"{own_trait} judge score (−4…4)")
        ax.set_title("Trait strength (LLM judge)" if row == 0 else "")

        # -- col 2: TRAIT P(high), all five splits ------------------------
        ax = axes[row][1]
        for split in OCEAN_SPLITS:
            vals = [trait_scores[_model_name(slug, w)][split] for w in SFT_WEIGHTS]
            is_own = split == own_trait
            ax.plot(
                SFT_WEIGHTS, vals,
                color=BIG_FIVE_COLORS[split],
                linewidth=2.4 if is_own else 1.3,
                alpha=1.0 if is_own else 0.85,
                marker="o", markersize=4.5 if is_own else 3.5,
                label=split if row == 0 else None,
            )
            persona_val = trait_scores[f"{slug}_persona1p00_ref"][split]
            ax.plot(
                [PERSONA_X], [persona_val],
                marker="D", markersize=5.5 if is_own else 4.5,
                markerfacecolor="white", markeredgecolor=BIG_FIVE_COLORS[split],
                linestyle="none",
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("TRAIT  P(high)")
        ax.set_title("All-trait TRAIT profile (off-target shifts)" if row == 0 else "")

        # -- col 3: MMLU ---------------------------------------------------
        ax = axes[row][2]
        accs = [mmlu_scores[_model_name(slug, w)]["accuracy"] for w in SFT_WEIGHTS]
        cis = [wilson_ci(a, MMLU_N) for a in accs]
        ax.fill_between(
            SFT_WEIGHTS, [c[0] for c in cis], [c[1] for c in cis],
            color=SPINE_COLOR, alpha=0.12, linewidth=0,
        )
        ax.plot(SFT_WEIGHTS, accs, color=SPINE_COLOR, linewidth=2.0, marker="o", markersize=5)
        persona_acc = mmlu_scores[f"{slug}_persona1p00_ref"]["accuracy"]
        p_lo, p_hi = wilson_ci(persona_acc, MMLU_N)
        ax.errorbar(
            [PERSONA_X], [persona_acc], yerr=[[persona_acc - p_lo], [p_hi - persona_acc]],
            fmt="D", markerfacecolor="white", markeredgecolor=SPINE_COLOR,
            ecolor=SPINE_COLOR, markersize=6, capsize=3, linewidth=1.0,
        )
        ax.axhline(persona_acc, color=SPINE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.45)
        ax.set_ylabel("MMLU accuracy (300 q)")
        ax.set_title("Capability (MMLU)" if row == 0 else "")

        # -- shared cosmetics per row -------------------------------------
        for col in range(3):
            ax = axes[row][col]
            ax.grid(True, axis="y", alpha=0.7)
            ax.set_xticks(X_TICKS)
            ax.set_xlim(-0.05, 1.12)
            ax.axvline(RELEASED_W, color=GRID_COLOR, linewidth=1.0, linestyle=":")
            if row == 1:
                ax.set_xlabel("SFT soup weight  w   (adapter = DPO + w·SFT)")
        axes[row][0].annotate(
            row_label, xy=(0.03, 0.92), xycoords="axes fraction",
            fontsize=10.5, fontweight="semibold", color=SPINE_COLOR,
        )

    # Legends: judge legend on top-left panel; trait legend on top-middle.
    axes[0][0].legend(loc="lower right", fontsize=8, frameon=False)
    axes[0][1].legend(loc="center left", fontsize=8, frameon=False, ncol=1)

    fig.tight_layout()
    out_path = PAPER_FIGURES_DIR / PAPER_FIGURES[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ saved {out_path}")
    print(f"✓ saved {png_path}")


def main() -> None:
    print(f"[soup_sft_weight] cache dir: {CACHE_DIR}")
    render()


if __name__ == "__main__":
    main()

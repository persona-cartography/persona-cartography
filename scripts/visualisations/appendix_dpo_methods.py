"""Appendix figure: 4×2 grid of MCQ results for the four DPO training methods.

Compares the neuroticism suppressor (N-) LoRA trained with four different
DPO strategies on the same base model (Llama 3.1 8B-Instruct):

  1. ``vanton4``         — Original Open Character Training method: all OCEAN
                           traits concatenated in the constitution with a common
                           rubric, indicating which trait to shift and in which
                           direction.
  2. ``v4``              — Hand-crafted constitution with bespoke per-facet
                           descriptions for neuroticism only; no general OCEAN
                           rubric.
  3. ``v4_reversed_dpo`` — Teacher generates the *amplifying* response (rejected)
                           while the student's normal, unconditioned response
                           is marked as chosen.
  4. ``v4_paired_dpo``   — Teacher generates both an amplifying and a
                           suppressing response; the suppressing one is chosen.

Each row is one method; columns are [TRAIT logprobs across OCEAN traits,
MMLU accuracy] vs LoRA scale. TRAIT logprob lines use the canonical
BIG_FIVE_COLORS palette.

Data (all under
``fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/{version}/evals/``):
  * TRAIT logprobs: ``mcq/trait_logprobs/{suite}/lora_<scale>/trait_logprobs/native/inspect_logs/*.json``
  * MMLU:           ``mcq/mmlu/{suite}/lora_<scale>/mmlu/native/inspect_logs/*.json``


Paper figures:
    - paper/figures/appendix/training_methods/fig_B_dpo_methods_scaling.pdf

Run with:
    uv run python -m scripts.visualisations.appendix_dpo_methods
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from src.evals.personality.sweep_results import (
    _extract_scores,
    _parse_scale,
)
from src.visualisations.palette import BIG_FIVE_COLORS
from src.utils.hf_hub import download_path_to_dir
from src.visualisations import PAPER_FIGURES_DIR

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

PAPER_FIGURES = [
    "appendix/training_methods/fig_B_dpo_methods_scaling.pdf",
]

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
ADAPTER_ROOT = f"fine_tuning/{MODEL_SLUG}/ocean/neuroticism/suppressor"

# (display label, version dir name, trait_logprobs suite, mmlu suite-or-None)
METHODS: list[tuple[str, str, str, str | None]] = [
    (
        "OCEAN definition constitution",
        "vanton4",
        "n_minus_vanton4_logprobs",
        "n_minus_vanton4",
    ),
    ("bespoke trait constitution", "v4", "n_minus_v4_logprobs", "n_minus_v4"),
    (
        "bespoke + reversed DPO",
        "v4_reversed_dpo",
        "n_minus_v4_reversed_dpo_logprobs",
        "n_minus_v4_reversed_dpo",
    ),
    (
        "bespoke + paired DPO",
        "v4_paired_dpo",
        "n_minus_v4_paired_dpo_logprobs",
        "n_minus_v4_paired_dpo",
    ),
]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "dpo_methods"
LOCAL_MONOREPO = project_root / "scratch" / "monorepo"


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _download_subtree(hf_path: str, allow_patterns: list[str]) -> Path:
    """Download a subtree; retry a few times on 429/5xx since the recursive
    tree listing in huggingface_hub can bounce on bulk requests even when
    individual reads are fine.
    """
    target = _cache_path(hf_path)
    target.mkdir(parents=True, exist_ok=True)
    attempts, delay = 4, 5.0
    for i in range(1, attempts + 1):
        try:
            download_path_to_dir(
                repo_id=HF_REPO_ID,
                path_in_repo=hf_path,
                target_dir=target,
                allow_patterns=allow_patterns,
            )
            return target
        except Exception as exc:
            msg = str(exc)
            transient = any(
                m in msg
                for m in ("429", "Too Many Requests", "500", "502", "503", "504")
            )
            if i < attempts and transient:
                print(f"  … retry {i}/{attempts} after {delay:.0f}s: {str(exc)[:80]}")
                time.sleep(delay)
                delay *= 2
                continue
            print(
                f"  ✗ HF hydrate failed for {hf_path}: {type(exc).__name__}: {str(exc)[:120]}"
            )
            break
    return target


def _parse_mcq_suite(
    suite_hf_path: str,
    inner_eval_name: str,
) -> dict[float, dict[str, float]]:
    """Walk ``<suite>/<model>/<inner_eval>/native/inspect_logs/*.json`` and
    return ``{scale: {metric: value}}``. Picks the most-metric-keys log per
    scale so re-runs that expanded metric coverage win over early partial runs.
    """
    _download_subtree(
        suite_hf_path,
        allow_patterns=[f"*/{inner_eval_name}/native/inspect_logs/*.json"],
    )

    def _walk(suite_dir: Path) -> dict[float, dict[str, float]]:
        acc: dict[float, dict[str, float]] = {}
        if not suite_dir.exists():
            return acc
        for model_dir in sorted(suite_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            scale = _parse_scale(model_dir.name)
            if scale is None:
                if model_dir.name == "base":
                    scale = 0.0
                else:
                    continue
            log_dir = model_dir / inner_eval_name / "native" / "inspect_logs"
            if not log_dir.exists():
                continue
            log_files = sorted(log_dir.glob("*.json"))
            if not log_files:
                continue
            best_scores: dict[str, float] | None = None
            for lf in log_files:
                extracted = _extract_scores(lf)
                if extracted is None:
                    continue
                scores, _parse_rate = extracted
                if best_scores is None or len(scores) > len(best_scores):
                    best_scores = scores
            if best_scores is None:
                continue
            acc[float(scale)] = best_scores
        return acc

    out = _walk(_cache_path(suite_hf_path))
    # If HF was rate-limited / 500'd, try the local scratch/monorepo mirror
    # before giving up (same trick as paper_main_o_plus_scaling.py).
    if not out:
        out = _walk(LOCAL_MONOREPO / suite_hf_path)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _plot_trait_panel(
    ax, trait_scores: dict[float, dict[str, float]], method_label: str
) -> None:
    scales = sorted(trait_scores.keys())
    for trait in OCEAN_TRAITS:
        ys = []
        for s in scales:
            row = trait_scores[s]
            val = row.get(trait) or row.get(trait.lower())
            ys.append(float(val) if val is not None else np.nan)
        ax.plot(
            scales,
            ys,
            "o-",
            color=BIG_FIVE_COLORS[trait],
            linewidth=1.6,
            markersize=3.5,
            label=trait,
        )
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TRAIT logprob")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.97,
        method_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_mmlu_panel(
    ax, mmlu_scores: dict[float, dict[str, float]] | None, method_label: str
) -> None:
    if mmlu_scores is None or not mmlu_scores:
        ax.text(
            0.5,
            0.5,
            "MMLU not run",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            color="#777777",
            style="italic",
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        return
    scales = sorted(mmlu_scores.keys())
    ys = []
    for s in scales:
        row = mmlu_scores[s]
        val = row.get("accuracy") or row.get("mean")
        ys.append(float(val) if val is not None else np.nan)
    ax.plot(
        scales, ys, "o-", color="#4D4D4D", linewidth=1.8, markersize=4.0, label="MMLU"
    )
    # 90%-of-base reference line
    base = mmlu_scores.get(0.0, {})
    base_val = base.get("accuracy") or base.get("mean")
    if base_val is not None:
        ax.axhline(
            0.9 * float(base_val),
            color="green",
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("MMLU accuracy")
    ax.grid(True, alpha=0.25)


def render_grid(
    per_method: dict[
        str, tuple[dict[float, dict[str, float]], dict[float, dict[str, float]] | None]
    ],
    out_path: Path,
) -> None:
    """Render the 4×2 (methods × {TRAIT, MMLU}) grid."""
    n_methods = len(METHODS)
    fig, axes = plt.subplots(
        n_methods, 2, figsize=(10.5, 2.25 * n_methods), sharex=True
    )
    for i, (label, _v, _tsuite, _msuite) in enumerate(METHODS):
        trait_scores, mmlu_scores = per_method[label]
        _plot_trait_panel(axes[i, 0], trait_scores, label)
        _plot_mmlu_panel(axes[i, 1], mmlu_scores, label)
    # Only the bottom row gets x-labels; and put a shared legend at the top.
    for col in range(2):
        axes[-1, col].set_xlabel("LoRA scale")
    # Legend on the first trait panel — same trait colours across all methods.
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=5,
        fontsize=9,
        framealpha=0.9,
        frameon=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[dpo-methods] cache dir: {CACHE_DIR}")
    per_method: dict[str, tuple[dict, dict | None]] = {}
    for label, version, trait_suite, mmlu_suite in METHODS:
        print(f"[dpo-methods] {label}")
        trait_scores = _parse_mcq_suite(
            f"{ADAPTER_ROOT}/{version}/evals/mcq/trait_logprobs/{trait_suite}",
            inner_eval_name="trait_logprobs",
        )
        if mmlu_suite is None:
            mmlu_scores = None
            print(f"  TRAIT scales: {len(trait_scores)}  |  MMLU: (not run)")
        else:
            mmlu_scores = _parse_mcq_suite(
                f"{ADAPTER_ROOT}/{version}/evals/mcq/mmlu/{mmlu_suite}",
                inner_eval_name="mmlu",
            )
            print(
                f"  TRAIT scales: {len(trait_scores)}  |  MMLU scales: {len(mmlu_scores)}"
            )
        per_method[label] = (trait_scores, mmlu_scores)
    render_grid(per_method, PAPER_FIGURES_DIR / PAPER_FIGURES[0])


if __name__ == "__main__":
    main()

"""Activation-capping MMLU breakdown sweep figures for the appendix.

Produces one MMLU-breakdown stacked-bar plot per OCEAN± persona for the
paired-DPO LoRAs evaluated under activation capping. Each plot shows fractions
of Correct / Recovered / Wrong answer / No answer at every cap scale, with
Wilson 95 % CI error bars on each category.

Output:
    paper/figures/appendix/activation_capping_mcq_llm_judge_evals/
        mmlu_breakdown_<trait>_<sign>_actcap.pdf

The MMLU breakdown needs per-sample answer-parsing flags, so we download the
full inspect-log JSON for each cap × persona. Each file is downloaded to a
script-local tempfile, parsed, and **deleted immediately** before the next file
is touched, so peak disk usage is bounded by ``max_workers * file_size``. The
cache directory itself is removed on exit.

The per-persona run-dir child name is not uniform on HF (e.g.
``o_plus_activation_capping_mmlu`` vs
``n_minus_activation_capping_ocean_const_paired_dpo_mmlu``), so the single
child under each eval dir is enumerated via glob rather than hardcoded.
"""

from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from huggingface_hub import HfFileSystem

from src.evals.personality.ci import _interval_ci_from_wilson
from src.evals.personality.sweep_results import _extract_raw_sample_scores
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.appendix_sweep_common import parse_cap_name, stream_to_tempfile

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

# Activation-capping figures cover the 10 OCEAN± personas only (no control).
PERSONAS: list[tuple[str, str]] = [
    (trait, direction)
    for trait in (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    )
    for direction in ("amplifier", "suppressor")
]

# All 10 OCEAN± personas have actcap MMLU data uploaded.
SKIP: set[tuple[str, str]] = set()

OUT_DIR = Path("appendix/activation_capping_mcq_llm_judge_evals")

PAPER_FIGURES = [
    f"{OUT_DIR}/mmlu_breakdown_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_actcap.pdf"
    for trait, direction in PERSONAS
    if (trait, direction) not in SKIP
]

CACHE_DIR = project_root / "scratch" / "_actcap_mmlu_cache"
CI_CONFIDENCE = 95.0

_session = requests.Session()


def _persona_parent_dir(trait: str, direction: str) -> str:
    """Parent dir of the actcap MMLU run for a persona. The run-dir name
    underneath this varies (e.g. ``n_minus`` uses
    ``..._ocean_const_paired_dpo_mmlu`` while the others use ``..._mmlu``), so
    the enumerate-globber walks through whatever single child it finds rather
    than hardcoding the suffix.
    """
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/"
        f"mcq/activation_capping/mmlu"
    )


# dedup-candidate: identical `cap_<±XpYY>` → float parser used by the actcap
# trait sweep script; kept local because the shared `parse_lora_name` only
# understands `lora_<...>x` names, not the `cap_<...>` names used here.
def _enumerate_log_paths() -> dict[tuple[str, str], dict[float, str]]:
    fs = HfFileSystem()
    targets = [p for p in PERSONAS if p not in SKIP]
    out: dict[tuple[str, str], dict[float, str]] = {p: {} for p in targets}

    def glob_one(persona: tuple[str, str]) -> tuple[tuple[str, str], list[str]]:
        trait, direction = persona
        parent = _persona_parent_dir(trait, direction)
        # First `*` enumerates the single run-dir child; second `*` the cap dir.
        pattern = f"datasets/{HF_REPO_ID}/{parent}/*/*/mmlu/native/inspect_logs/*.json"
        return persona, list(fs.glob(pattern))

    print(f"Enumerating inspect logs for {len(targets)} personas …")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for persona, matches in ex.map(glob_one, targets):
            for full in matches:
                rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
                cap_dir = rel.split("/mmlu/native/")[0].rsplit("/", 1)[1]
                scale = parse_cap_name(cap_dir)
                if scale is None:
                    continue
                out[persona][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(targets)} personas")
    return out


def _breakdown_from_log(log_path: Path) -> dict[str, float] | None:
    """Per-category fractions and Wilson CIs.

    Each sample falls into exactly one of four mutually-exclusive categories,
    so we compute an independent Wilson CI per category from the per-sample
    binary indicator of membership.
    """
    raw = _extract_raw_sample_scores(log_path, "mmlu")
    if not raw:
        return None
    acc = np.asarray(raw.get("accuracy", []), dtype=float)
    ap = np.asarray(raw.get("_answer_parsed", []), dtype=float)
    rp = np.asarray(raw.get("_reparsed_accuracy", []), dtype=float)
    n = min(len(acc), len(ap))
    if n == 0:
        return None
    acc, ap = acc[:n], ap[:n]
    if len(rp) >= n:
        rp = rp[:n]
    else:
        rp = np.zeros(n)
    cat_arrays = {
        "Correct": acc,
        "Recovered": (1 - acc) * rp,
        "Wrong answer": (1 - acc) * (1 - rp) * ap,
        "No answer": (1 - acc) * (1 - rp) * (1 - ap),
    }
    out: dict[str, float] = {"n": float(n)}
    for cat, arr in cat_arrays.items():
        out[cat] = float(arr.mean())
        lo, hi = _interval_ci_from_wilson(arr.astype(int), CI_CONFIDENCE)
        out[f"{cat}_lo"] = lo
        out[f"{cat}_hi"] = hi
    return out


def _process_one(rel_path: str) -> dict[str, float] | None:
    return stream_to_tempfile(
        f"{RESOLVE_BASE}/{rel_path}",
        CACHE_DIR,
        _breakdown_from_log,
        session=_session,
        suffix=".json",
        timeout=300,
        chunk_size=1 << 20,
        quiet_on_non_200=False,
        label=rel_path,
    )


def gather_all_breakdowns() -> dict[tuple[str, str], dict[float, dict[str, float]]]:
    log_paths = _enumerate_log_paths()
    jobs: list[tuple[tuple[str, str], float, str]] = []
    for persona, by_scale in log_paths.items():
        for scale, path in by_scale.items():
            jobs.append((persona, scale, path))

    print(
        f"Downloading + parsing {len(jobs)} MMLU inspect logs (deleting each after parse) …"
    )
    out: dict[tuple[str, str], dict[float, dict[str, float]]] = {
        p: {} for p in log_paths
    }
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_job = {
            ex.submit(_process_one, path): (persona, scale)
            for persona, scale, path in jobs
        }
        for fut in as_completed(future_to_job):
            persona, scale = future_to_job[fut]
            row = fut.result()
            if row is not None:
                out[persona][scale] = row
    return out


_CAT_COLORS = {
    "Correct": "#2ECC71",
    "Recovered": "#3498DB",
    "Wrong answer": "#E74C3C",
    "No answer": "#95A5A6",
}
_CATS = ["Correct", "Recovered", "Wrong answer", "No answer"]


def render_persona(
    home_trait: str,
    direction: str,
    breakdown: dict[float, dict[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if not breakdown:
        plt.close(fig)
        return
    scales = sorted(breakdown.keys())
    x = np.arange(len(scales))
    bottom = np.zeros(len(scales))
    for cat in _CATS:
        vals = np.asarray([breakdown[s].get(cat, 0.0) for s in scales])
        ax.bar(
            x,
            vals,
            width=0.85,
            bottom=bottom,
            label=cat,
            color=_CAT_COLORS[cat],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        # Wilson CI for this category, drawn at the top edge of its segment in
        # the category's own colour. Clip to 0 to avoid negative yerr.
        los = np.asarray([breakdown[s].get(f"{cat}_lo", float("nan")) for s in scales])
        his = np.asarray([breakdown[s].get(f"{cat}_hi", float("nan")) for s in scales])
        yerr = np.clip(np.stack([vals - los, his - vals]), 0.0, None)
        top_edges = bottom + vals
        ax.errorbar(
            x,
            top_edges,
            yerr=yerr,
            fmt="none",
            ecolor=_CAT_COLORS[cat],
            elinewidth=0.6,
            capsize=1,
            capthick=0.5,
            alpha=0.95,
        )
        bottom += vals
    labels = [f"{s:+.2f}" if s != 0 else "0.00" for s in scales]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Activation Vector Limit")
    ax.set_ylabel("Percentage of Samples")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        fontsize=8,
        framealpha=0.9,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sign = "↑" if direction == "amplifier" else "↓"
    ax.set_title(
        f"MMLU Activation Capping: {home_trait.capitalize()} {sign}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        all_breakdowns = gather_all_breakdowns()
        print(f"Rendering {len(all_breakdowns)} figures …")
        for (trait, direction), breakdown in all_breakdowns.items():
            sign = "plus" if direction == "amplifier" else "minus"
            out = (
                PAPER_FIGURES_DIR
                / OUT_DIR
                / f"mmlu_breakdown_{trait}_{sign}_actcap.pdf"
            )
            render_persona(trait, direction, breakdown, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

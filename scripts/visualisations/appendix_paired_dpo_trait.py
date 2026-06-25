"""Paired-DPO LoRA-scale TRAIT logprobs sweep figures for the OCEAN appendix.

Mirror of ``paper_appendix_downrank_trait.py`` for the regular vanton4 paired
DPO LoRA-scale sweep (no rank reduction). Each plot shows all 5 OCEAN trait
logprob scores against the LoRA scale with bootstrap-CI error bars, plus a
choice-mass diagnostic strip below.

Logic, HF paths, and outputs match the trait-sweep appendix figure.

Output:
    paper/figures/appendix/ocean_results/
        trait_sweep_<trait>_<sign>_paired_dpo.pdf

Data source: inspect logs at

    fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/ocean_const_paired_dpo/
        evals/mcq/trait_logprobs/{letter}_{sign}_ocean_const_paired_dpo_logprobs/
        {base, lora_<±XpYY>x}/trait_logprobs/native/inspect_logs/*.json
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

from src.visualisations.palette import BIG_FIVE_COLORS
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.appendix_sweep_common import (
    PERSONAS,
    bootstrap_ci,
    parse_lora_name,
    per_trait_scores_from_log,
    persona_filename_stem,
    persona_title,
    stream_to_tempfile,
)

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

# PERSONAS comes from appendix_sweep_common.
OUT_DIR = Path("appendix/ocean_results")

PAPER_FIGURES = [
    f"{OUT_DIR}/trait_sweep_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_paired_dpo.pdf"
    for trait, direction in PERSONAS
]

CACHE_DIR = project_root / "scratch" / "_paired_dpo_trait_cache"
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
SEED = 42
MIN_CHOICE_MASS = 0.75


def _persona_run_dir(trait: str, direction: str) -> str:
    if trait == "control":
        return (
            f"fine_tuning/{MODEL_SLUG}/other/ocean_def_control/amplifier/"
            f"ocean_const_paired_dpo_s1vs2/evals/mcq/trait_logprobs/"
            f"control_s1vs2_ocean_const_paired_dpo_logprobs"
        )
    sign = "plus" if direction == "amplifier" else "minus"
    letter = trait[0]
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/"
        f"mcq/trait_logprobs/{letter}_{sign}_ocean_const_paired_dpo_logprobs"
    )


def _enumerate_log_paths() -> dict[tuple[str, str], dict[float, str]]:
    fs = HfFileSystem()
    out: dict[tuple[str, str], dict[float, str]] = {p: {} for p in PERSONAS}

    def glob_one(persona: tuple[str, str]) -> tuple[tuple[str, str], list[str]]:
        trait, direction = persona
        run_dir = _persona_run_dir(trait, direction)
        pattern = f"datasets/{HF_REPO_ID}/{run_dir}/*/trait_logprobs/native/inspect_logs/*.json"
        return persona, list(fs.glob(pattern))

    print(f"Enumerating inspect logs for {len(PERSONAS)} personas …")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for persona, matches in ex.map(glob_one, PERSONAS):
            for full in matches:
                rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
                cap_dir = rel.split("/trait_logprobs/native/")[0].rsplit("/", 1)[1]
                scale = parse_lora_name(cap_dir)
                if scale is None:
                    continue
                out[persona][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(PERSONAS)} personas")
    return out


def _per_trait_scores_from_log(log_path: Path) -> dict[str, np.ndarray] | None:
    return per_trait_scores_from_log(
        log_path, OCEAN_TRAITS, min_choice_mass=MIN_CHOICE_MASS
    )


_session = requests.Session()


def _process_one(rel_path: str) -> dict[str, np.ndarray] | None:
    return stream_to_tempfile(
        f"{RESOLVE_BASE}/{rel_path}",
        CACHE_DIR,
        _per_trait_scores_from_log,
        session=_session,
        suffix=".json",
        timeout=300,
        chunk_size=1 << 20,
        quiet_on_non_200=False,
        label=rel_path,
    )


def gather_one_persona_scores(
    persona: tuple[str, str],
    log_paths: dict[float, str],
) -> dict[float, dict[str, np.ndarray]]:
    print(
        f"  [{persona[0]}/{persona[1]}] downloading + parsing {len(log_paths)} "
        "trait inspect logs (deleting each after parse) …"
    )
    out: dict[float, dict[str, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_scale = {
            ex.submit(_process_one, path): scale for scale, path in log_paths.items()
        }
        for fut in as_completed(future_to_scale):
            scale = future_to_scale[fut]
            per_trait = fut.result()
            if per_trait is not None:
                out[scale] = per_trait
    return out


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    return bootstrap_ci(
        values, confidence=CI_CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )


def render_persona(
    home_trait: str,
    direction: str,
    by_scale: dict[float, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not by_scale:
        return
    scales = sorted(by_scale.keys())
    has_choice_mass = any("_choice_mass_all" in by_scale.get(s, {}) for s in scales)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.5, 3.8) if has_choice_mass else (6.5, 3.2))
    if has_choice_mass:
        gs = GridSpec(2, 1, height_ratios=[85, 15], hspace=0.05, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_cm = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_cm = None

    for trait in OCEAN_TRAITS:
        means_list: list[float] = []
        los_list: list[float] = []
        his_list: list[float] = []
        for s in scales:
            row = by_scale.get(s, {})
            arr = row.get(trait)
            m, lo, hi = _bootstrap_ci(arr) if arr is not None else (float("nan"),) * 3
            means_list.append(m)
            los_list.append(lo)
            his_list.append(hi)
        means = np.asarray(means_list)
        los = np.asarray(los_list)
        his = np.asarray(his_list)
        yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
        color = BIG_FIVE_COLORS[trait]
        ax.errorbar(
            scales,
            means,
            yerr=yerr,
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=2.0,
            markersize=5,
            elinewidth=1.0,
            capsize=3,
            label=trait,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylabel("TRAIT logprob score", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-4.0, 4.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        persona_title(home_trait, direction, "TRAIT"),
        fontsize=13,
    )

    if ax_cm is not None:
        cm_means = []
        for s in scales:
            cm_arr = by_scale.get(s, {}).get("_choice_mass_all")
            if cm_arr is not None and len(cm_arr) > 0:
                kept = cm_arr[cm_arr >= MIN_CHOICE_MASS]
                cm_means.append(float(kept.mean()) if len(kept) else float("nan"))
            else:
                cm_means.append(float("nan"))
        ax_cm.plot(
            scales,
            cm_means,
            "s-",
            color="#555555",
            linewidth=1.4,
            markersize=3,
            zorder=4,
        )
        ax_cm.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_cm.set_ylabel(
            "Choice\nMass", fontsize=10, rotation=0, labelpad=24, va="center"
        )
        ax_cm.set_ylim(MIN_CHOICE_MASS, 1.0)
        ax_cm.set_yticks([MIN_CHOICE_MASS, 1.0])
        ax_cm.set_yticklabels([f"{MIN_CHOICE_MASS:g}", "1"], fontsize=9)
        ax_cm.grid(True, alpha=0.25)
        ax_cm.set_xlabel("LoRA Scale", fontsize=12)
        ax_cm.tick_params(axis="x", labelsize=11)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("LoRA Scale", fontsize=12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42 if ax_cm is not None else -0.18),
        ncol=5,
        fontsize=8,
        framealpha=0.9,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=0.8,
        borderpad=0.3,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log_paths_by_persona = _enumerate_log_paths()
        print(f"Processing {len(PERSONAS)} personas one at a time …")
        for persona in PERSONAS:
            trait, direction = persona
            log_paths = log_paths_by_persona.get(persona, {})
            if not log_paths:
                print(f"  [{trait}/{direction}] no inspect logs found — skipping")
                continue
            scores = gather_one_persona_scores(persona, log_paths)
            stem = persona_filename_stem(trait, direction)
            out = PAPER_FIGURES_DIR / OUT_DIR / f"trait_sweep_{stem}.pdf"
            render_persona(trait, direction, scores, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

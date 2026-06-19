"""Base↔instruct interpolation TRAIT logprob sweep figures for the appendix.

Mirror of ``appendix_paired_dpo_trait.py`` for the
``mcq/trait_logprobs_average_base_instruct_persona_w<W>`` sweeps. There are
five interpolation weights ``W ∈ {0.01, 0.05, 0.25, 0.50, 0.75}`` and a
single persona — the conscientiousness suppressor (C$\\downarrow$).

Output:
    paper/figures/appendix/interp_between_base_and_instruct_tuned/
        trait_sweep_<W_label>.pdf

where ``W_label`` matches the user's existing LaTeX ``\\interprow`` calls:
``0_01, 0_05, 0_25, 0_5, 0_75`` (note ``0_5`` rather than ``0_50``).

Data source: inspect logs at

    fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/
        ocean_const_paired_dpo/evals/mcq/
        trait_logprobs_average_base_instruct_persona_w<W_HF>/
        c_minus_ocean_const_paired_dpo_average_base_instruct_persona_w<W_HF>_trait_logprobs/
        {base, lora_<±XpYY>x}/trait_logprobs/native/inspect_logs/*.json

Bootstrap CIs need per-sample scores, which live in the ``samples`` section
of each ~57 MB inspect log, so we have to download them in full. Files land
in a script-local cache directory which is removed when the script exits.
"""

from __future__ import annotations

import json
import math
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
    bootstrap_ci,
    parse_lora_name,
    stream_to_tempfile,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

# (HF weight token, LaTeX/file weight label, numeric weight)
WEIGHTS: list[tuple[str, str, float]] = [
    ("w0_01", "0_01", 0.01),
    ("w0_05", "0_05", 0.05),
    ("w0_25", "0_25", 0.25),
    ("w0_50", "0_5", 0.50),
    ("w0_75", "0_75", 0.75),
]

OUT_DIR = Path("appendix/interp_between_base_and_instruct_tuned")

PAPER_FIGURES = [f"{OUT_DIR}/trait_sweep_{label}.pdf" for _, label, _ in WEIGHTS]

CACHE_DIR = project_root / "scratch" / "_interp_trait_cache"
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
SEED = 42
MIN_CHOICE_MASS = 0.75


def _persona_run_dir(weight_hf: str) -> str:
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/conscientiousness/suppressor/ocean_const_paired_dpo/evals/"
        f"mcq/trait_logprobs_average_base_instruct_persona_{weight_hf}/"
        f"c_minus_ocean_const_paired_dpo_average_base_instruct_persona_{weight_hf}_trait_logprobs"
    )


def _enumerate_log_paths() -> dict[str, dict[float, str]]:
    """Returns ``{weight_hf: {scale: log_repo_path}}``."""
    fs = HfFileSystem()
    out: dict[str, dict[float, str]] = {w[0]: {} for w in WEIGHTS}

    def glob_one(weight_hf: str) -> tuple[str, list[str]]:
        run_dir = _persona_run_dir(weight_hf)
        pattern = f"datasets/{HF_REPO_ID}/{run_dir}/*/trait_logprobs/native/inspect_logs/*.json"
        return weight_hf, list(fs.glob(pattern))

    print(f"Enumerating inspect logs for {len(WEIGHTS)} interpolation weights …")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for weight_hf, matches in ex.map(glob_one, [w[0] for w in WEIGHTS]):
            for full in matches:
                rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
                cap_dir = rel.split("/trait_logprobs/native/")[0].rsplit("/", 1)[1]
                scale = parse_lora_name(cap_dir)
                if scale is None:
                    continue
                out[weight_hf][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(WEIGHTS)} weights")
    return out


def _per_trait_scores_from_log(log_path: Path) -> dict[str, np.ndarray] | None:
    try:
        with log_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError:
        return None
    samples = doc.get("samples") or []
    by_trait: dict[str, list[float]] = {t: [] for t in OCEAN_TRAITS}
    all_choice_mass: list[float] = []
    for s in samples:
        meta = s.get("metadata") or {}
        trait = meta.get("trait")
        if trait not in by_trait:
            continue
        scores = s.get("scores") or {}
        scorer = scores.get("logprob_mcq_scorer") or {}
        v = scorer.get("value")
        if not isinstance(v, (int, float)):
            continue
        smeta = scorer.get("metadata") or {}
        cm = smeta.get("choice_mass")
        if cm is None:
            lps = smeta.get("logprobs")
            if isinstance(lps, dict) and lps:
                cm = sum(math.exp(x) for x in lps.values())
        if isinstance(cm, (int, float)):
            all_choice_mass.append(float(cm))
            if cm < MIN_CHOICE_MASS:
                continue
        by_trait[trait].append(float(v))
    out: dict[str, np.ndarray] = {
        t: np.asarray(v, dtype=float) for t, v in by_trait.items() if v
    }
    if all_choice_mass:
        out["_choice_mass_all"] = np.asarray(all_choice_mass, dtype=float)
    return out or None


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


def gather_one_weight_scores(
    weight_hf: str,
    log_paths: dict[float, str],
) -> dict[float, dict[str, np.ndarray]]:
    print(
        f"  [{weight_hf}] downloading + parsing {len(log_paths)} trait inspect logs "
        "(deleting each after parse) …"
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


def render_weight(
    weight: float,
    by_scale: dict[float, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not by_scale:
        return
    scales = sorted(by_scale.keys())
    has_choice_mass = any(
        "_choice_mass_all" in by_scale.get(s, {}) for s in scales
    )

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.5, 4.5) if has_choice_mass else (6.5, 3.8))
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
            scales, means, yerr=yerr, fmt="o-",
            color=color, ecolor=color,
            linewidth=2.0, markersize=5, elinewidth=1.0, capsize=3,
            label=trait,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylabel("TRAIT logprob score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-4.0, 4.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"TRAIT Base$\\leftrightarrow$Instruct Interpolation: C$\\downarrow$ at $w = {weight:g}$",
        fontsize=10,
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
        ax_cm.plot(scales, cm_means, "s-", color="#555555",
                   linewidth=1.4, markersize=3, zorder=4)
        ax_cm.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_cm.set_ylabel("Choice\nmass", fontsize=7, rotation=0,
                         labelpad=22, va="center")
        ax_cm.set_ylim(MIN_CHOICE_MASS, 1.0)
        ax_cm.set_yticks([MIN_CHOICE_MASS, 1.0])
        ax_cm.set_yticklabels([f"{MIN_CHOICE_MASS:g}", "1"], fontsize=6)
        ax_cm.grid(True, alpha=0.25)
        ax_cm.set_xlabel("LoRA Scale")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("LoRA Scale")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42 if ax_cm is not None else -0.18),
        ncol=5, fontsize=8, framealpha=0.9,
        handlelength=1.2, handletextpad=0.5, columnspacing=0.8, borderpad=0.3,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log_paths_by_weight = _enumerate_log_paths()
        print(f"Processing {len(WEIGHTS)} weights one at a time …")
        for weight_hf, label, weight in WEIGHTS:
            log_paths = log_paths_by_weight.get(weight_hf, {})
            if not log_paths:
                print(f"  [{weight_hf}] no inspect logs found — skipping")
                continue
            scores = gather_one_weight_scores(weight_hf, log_paths)
            out = PAPER_FIGURES_DIR / OUT_DIR / f"trait_sweep_{label}.pdf"
            render_weight(weight, scores, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

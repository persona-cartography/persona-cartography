"""Beyond-OCEAN (sycophancy + psychopathy) LoRA-scale sweep figures.

Sibling of ``appendix_paired_dpo_trait.py`` and ``appendix_paired_dpo_mmlu.py``
for the sycophancy and psychopathy amplifier/suppressor adapters. The TRAIT
panels show all 8 TRAIT splits (Big Five + Dark Triad) against the LoRA scale
with bootstrap-CI error bars and a choice-mass diagnostic strip; for sycophancy
the separately-uploaded ``trait_logprobs`` (OCEAN) and ``trait_logprobs_dark``
(Dark Triad) runs are merged, while psychopathy has a single all-8-split
``trait_logprobs_all8`` run. The MMLU panels are the standard stacked Correct /
Recovered / Wrong answer / No answer breakdown with Wilson 95% CIs.

Paper figures:
    - paper/figures/appendix/beyond_ocean/trait_sweep_sycophancy_plus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/trait_sweep_sycophancy_minus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/mmlu_breakdown_sycophancy_plus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/mmlu_breakdown_sycophancy_minus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/trait_sweep_psychopathy_plus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/trait_sweep_psychopathy_minus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/mmlu_breakdown_psychopathy_plus_paired_dpo.pdf
    - paper/figures/appendix/beyond_ocean/mmlu_breakdown_psychopathy_minus_paired_dpo.pdf

Data source: inspect logs at

    fine_tuning/llama-3.1-8b-it/other/{sycophancy,psychopathy}/{amplifier,suppressor}/
        v{syco,psyc}1_paired_dpo/evals/mcq/<eval kind>/<run dir>/
        {base, lora_<±XpYY>x}/*/native/inspect_logs/*.json

The psychopathy artifacts were deliberately removed from the repo *tip* to keep
the adapter weights out of circulation (misuse avoidance). Their eval logs are
hydrated from ``PSYCHOPATHY_REVISION`` — the last monorepo revision that still
carries them — and only score logs are downloaded, never adapter weights.

Usage: ``python appendix_beyond_ocean_sweeps.py [sycophancy|psychopathy]``
(no argument = both personas).
"""

from __future__ import annotations

import os
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
from huggingface_hub import HfFileSystem, hf_hub_download

from src.evals.personality.ci import _interval_ci_from_wilson
from src.evals.personality.sweep_results import _extract_raw_sample_scores
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.palette import BIG_FIVE_COLORS, DARK_TRIAD_COLORS
from src.visualisations.appendix_sweep_common import (
    bootstrap_ci,
    parse_lora_name,
    per_trait_scores_from_log,
    persona_filename_stem,
    persona_title,
    stream_to_tempfile,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"

# The psychopathy adapters were deliberately removed from the monorepo tip so
# the weights stay out of circulation. Their eval score logs are read from the
# last revision that carries them; sycophancy reads from the tip as usual.
PSYCHOPATHY_REVISION = "343b8c78356891fda93cfc8b25030174a7e9d71a"

ALL_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]
TRAIT_COLORS = {**BIG_FIVE_COLORS, **DARK_TRIAD_COLORS}


def _sweep_specs() -> list[dict]:
    """One spec per persona: revision, TRAIT run dirs (merged), MMLU run dir."""
    specs: list[dict] = []
    for direction in ("amplifier", "suppressor"):
        sign = "plus" if direction == "amplifier" else "minus"
        base = (
            f"fine_tuning/{MODEL_SLUG}/other/sycophancy/{direction}/"
            f"vsyco1_paired_dpo/evals/mcq"
        )
        specs.append(
            {
                "trait": "sycophancy",
                "direction": direction,
                "revision": "main",
                "trait_runs": [
                    f"{base}/trait_logprobs/syco_{sign}_syco1_paired_dpo_logprobs",
                    f"{base}/trait_logprobs_dark/syco_{sign}_syco1_paired_dpo_dark_logprobs",
                ],
                "mmlu_run": f"{base}/mmlu/syco_{sign}_syco1_paired_dpo",
            }
        )
    for direction in ("amplifier", "suppressor"):
        sign = "plus" if direction == "amplifier" else "minus"
        base = (
            f"fine_tuning/{MODEL_SLUG}/other/psychopathy/{direction}/"
            f"vpsyc1_paired_dpo/evals/mcq"
        )
        specs.append(
            {
                "trait": "psychopathy",
                "direction": direction,
                "revision": PSYCHOPATHY_REVISION,
                "trait_runs": [
                    f"{base}/trait_logprobs_all8/psyc_{sign}_psyc1_paired_dpo_all8_logprobs",
                ],
                "mmlu_run": f"{base}/mmlu/psyc_{sign}_psyc1_paired_dpo",
            }
        )
    return specs


OUT_DIR = Path("appendix/beyond_ocean")

PAPER_FIGURES = [
    f"{OUT_DIR}/{family}_{persona_filename_stem(spec['trait'], spec['direction'])}.pdf"
    for family in ("trait_sweep", "mmlu_breakdown")
    for spec in _sweep_specs()
]

# Per-process cache dir: two concurrent runs must not share one, or the first
# to finish rmtree's the other's tempfiles mid-download.
CACHE_DIR = project_root / "scratch" / f"_beyond_ocean_sweep_cache_{os.getpid()}"
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
SEED = 42
MIN_CHOICE_MASS = 0.75

_session = requests.Session()


def _resolve_url(spec: dict, rel_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/"
        f"{spec['revision']}/{rel_path}"
    )


def _enumerate_log_paths(spec: dict) -> dict[str, dict[float, str]]:
    """Map run dir -> {scale: HF-relative inspect-log path} for one persona."""
    fs = HfFileSystem()
    run_dirs = [*spec["trait_runs"], spec["mmlu_run"]]
    out: dict[str, dict[float, str]] = {r: {} for r in run_dirs}

    def glob_one(run_dir: str) -> tuple[str, list[str]]:
        prefix = f"datasets/{HF_REPO_ID}@{spec['revision']}"
        # Pin the inner eval-task dir per kind — a run dir can carry stray
        # sibling evals (e.g. a trait_logprobs log inside an mmlu run's base/).
        inner = "mmlu" if run_dir == spec["mmlu_run"] else "trait_logprobs"
        pattern = f"{prefix}/{run_dir}/*/{inner}/native/inspect_logs/*.json"
        return run_dir, list(fs.glob(pattern))

    with ThreadPoolExecutor(max_workers=3) as ex:
        for run_dir, matches in ex.map(glob_one, run_dirs):
            for full in matches:
                rel = full[full.index(run_dir) :]
                scale_dir = rel.split("/native/")[0].rsplit("/", 2)[1]
                scale = parse_lora_name(scale_dir)
                if scale is None:
                    continue
                out[run_dir][scale] = rel
    total = sum(len(v) for v in out.values())
    print(
        f"  [{spec['trait']}/{spec['direction']}] found {total} log paths "
        f"across {len(run_dirs)} run dirs"
    )
    return out


# ---------------------------------------------------------------- TRAIT panels


def _per_trait_scores(log_path: Path) -> dict[str, np.ndarray] | None:
    return per_trait_scores_from_log(
        log_path, ALL_TRAITS, min_choice_mass=MIN_CHOICE_MASS
    )


DOWNLOAD_RETRIES = 3


def _hub_download_fallback(spec: dict, rel_path: str, parse):
    """Size-verified ``hf_hub_download`` fallback for streams that truncate.

    The plain ``resolve`` streaming path can drop the tail of the largest
    (~58MB) logs without raising, which makes ``json.load`` fail silently and
    leaves a gap in the sweep. ``hf_hub_download`` verifies the file size, so
    use it as the fallback when streaming + parse came up empty.
    """
    try:
        local = hf_hub_download(
            HF_REPO_ID,
            rel_path,
            repo_type="dataset",
            revision=spec["revision"],
            cache_dir=str(CACHE_DIR / "_hub_cache"),
        )
        result = parse(Path(local))
        print(f"  ↻ {rel_path}: recovered via hf_hub_download")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {rel_path}: fallback failed: {type(exc).__name__}")
        return None


def _process_trait_log(spec: dict, rel_path: str) -> dict[str, np.ndarray] | None:
    # Trait logs are ~50MB each; retry transient connection drops so a single
    # broken stream doesn't leave a gap in the sweep.
    for _ in range(DOWNLOAD_RETRIES):
        result = stream_to_tempfile(
            _resolve_url(spec, rel_path),
            CACHE_DIR,
            _per_trait_scores,
            session=_session,
            suffix=".json",
            timeout=300,
            chunk_size=1 << 20,
            quiet_on_non_200=False,
            label=rel_path,
        )
        if result is not None:
            return result
    return _hub_download_fallback(spec, rel_path, _per_trait_scores)


def gather_trait_scores(
    spec: dict,
    log_paths: dict[str, dict[float, str]],
) -> dict[float, dict[str, np.ndarray]]:
    """Download + parse the persona's TRAIT logs and merge them per scale."""
    jobs: list[tuple[float, str]] = []
    for run_dir in spec["trait_runs"]:
        jobs.extend(log_paths.get(run_dir, {}).items())
    print(
        f"  [{spec['trait']}/{spec['direction']}] downloading + parsing "
        f"{len(jobs)} trait inspect logs (deleting each after parse) …"
    )
    out: dict[float, dict[str, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_scale = {
            ex.submit(_process_trait_log, spec, path): scale for scale, path in jobs
        }
        for fut in as_completed(future_to_scale):
            scale = future_to_scale[fut]
            per_trait = fut.result()
            if per_trait is None:
                continue
            row = out.setdefault(scale, {})
            for trait, arr in per_trait.items():
                if trait == "_choice_mass_all" and trait in row:
                    row[trait] = np.concatenate([row[trait], arr])
                else:
                    row[trait] = arr
    return out


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    return bootstrap_ci(
        values, confidence=CI_CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )


def render_trait_persona(
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

    for trait in ALL_TRAITS:
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
        color = TRAIT_COLORS[trait]
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
        ncol=4,
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


# ----------------------------------------------------------------- MMLU panels


def _breakdown_from_log(log_path: Path) -> dict[str, float] | None:
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


def _process_mmlu_log(spec: dict, rel_path: str) -> dict[str, float] | None:
    for _ in range(DOWNLOAD_RETRIES):
        result = stream_to_tempfile(
            _resolve_url(spec, rel_path),
            CACHE_DIR,
            _breakdown_from_log,
            session=_session,
            suffix=".json",
            timeout=300,
            chunk_size=1 << 20,
            quiet_on_non_200=False,
            label=rel_path,
        )
        if result is not None:
            return result
    return _hub_download_fallback(spec, rel_path, _breakdown_from_log)


def gather_mmlu_breakdowns(
    spec: dict,
    log_paths: dict[str, dict[float, str]],
) -> dict[float, dict[str, float]]:
    jobs = list(log_paths.get(spec["mmlu_run"], {}).items())
    print(
        f"  [{spec['trait']}/{spec['direction']}] downloading + parsing "
        f"{len(jobs)} MMLU inspect logs (deleting each after parse) …"
    )
    out: dict[float, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_scale = {
            ex.submit(_process_mmlu_log, spec, path): scale for scale, path in jobs
        }
        for fut in as_completed(future_to_scale):
            scale = future_to_scale[fut]
            row = fut.result()
            if row is not None:
                out[scale] = row
    return out


_CAT_COLORS = {
    "Correct": "#2ECC71",
    "Recovered": "#3498DB",
    "Wrong answer": "#E74C3C",
    "No answer": "#95A5A6",
}
_CATS = ["Correct", "Recovered", "Wrong answer", "No answer"]


def render_mmlu_persona(
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
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("LoRA Scale", fontsize=12)
    ax.set_ylabel("Percentage of Samples", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        persona_title(home_trait, direction, "MMLU"),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    specs = [s for s in _sweep_specs() if only is None or s["trait"] == only]
    if not specs:
        raise SystemExit(f"no sweep specs match {only!r}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for spec in specs:
            trait, direction = spec["trait"], spec["direction"]
            stem = persona_filename_stem(trait, direction)
            log_paths = _enumerate_log_paths(spec)
            trait_scores = gather_trait_scores(spec, log_paths)
            render_trait_persona(
                trait,
                direction,
                trait_scores,
                PAPER_FIGURES_DIR / OUT_DIR / f"trait_sweep_{stem}.pdf",
            )
            breakdowns = gather_mmlu_breakdowns(spec, log_paths)
            render_mmlu_persona(
                trait,
                direction,
                breakdowns,
                PAPER_FIGURES_DIR / OUT_DIR / f"mmlu_breakdown_{stem}.pdf",
            )
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

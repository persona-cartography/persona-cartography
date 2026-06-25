"""Activation-capping TRAIT logprobs sweep figures for the appendix.

Produces one trait-vs-cap-scale plot per OCEAN± persona for the vanton4
paired-DPO LoRAs evaluated under activation capping. Each plot shows all 5
OCEAN trait logprob scores against the cap scale with shaded BCa-bootstrap
CI bands, mirroring the
``scripts_dev/personality_evals/configs/ocean/trait/vanton4_paired_dpo``
``ci95_from_bootstrap_1000`` convention.

Output:
    paper/figures/appendix/activation_capping_mcq_llm_judge_evals/
        trait_sweep_<trait>_<sign>_actcap.pdf

Data source: inspect logs at

    fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4_paired_dpo/
        evals/mcq/activation_capping/trait_logprobs/<run>/cap_<±XpYY>/
        trait_logprobs/native/inspect_logs/*.json

Bootstrap CIs need per-sample scores, which live in the ``samples`` section
of each ~57 MB inspect log, so we have to download them in full. Files land
in a script-local cache directory which is removed when the script exits —
nothing persists between runs.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from huggingface_hub.utils import build_hf_headers  # noqa: E402
from huggingface_hub import HfFileSystem

from src_dev.evals.personality.analyze_results import (
    BIG_FIVE_COLORS,
    _interval_ci_from_bootstrap,
)
from src_dev.visualisations import PAPER_FIGURES_DIR

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

OUT_DIR = Path("appendix/activation_capping_mcq_llm_judge_evals")

PAPER_FIGURES = [
    f"{OUT_DIR}/trait_sweep_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_actcap.pdf"
    for trait, direction in PERSONAS
]

CACHE_DIR = project_root / "scratch" / "_actcap_trait_cache"
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
SEED = 42
MIN_CHOICE_MASS = 0.75  # Match the eval's `dynamic_mass_filter` threshold.


def _persona_run_dir(trait: str, direction: str) -> str:
    sign = "plus" if direction == "amplifier" else "minus"
    letter = trait[0]
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/vanton4_paired_dpo/evals/"
        f"mcq/activation_capping/trait_logprobs/{letter}_{sign}_activation_capping_vanton4_trait_logprobs"
    )


def _parse_cap_name(name: str) -> float | None:
    if name == "base":
        return 0.0
    if not name.startswith("cap_"):
        return None
    body = name[len("cap_") :].replace("p", ".")
    try:
        return float(body)
    except ValueError:
        return None


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
                scale = _parse_cap_name(cap_dir)
                if scale is None:
                    continue
                out[persona][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(PERSONAS)} personas")
    return out


def _per_trait_scores_from_log(log_path: Path) -> dict[str, np.ndarray] | None:
    """Extract per-sample scores AND per-sample choice mass for one cap-scale.

    Returns a dict with:
      * ``<trait>``: per-sample logprob_mcq_scorer values for samples passing
        the ``min_choice_mass`` filter (used for the trait plot itself).
      * ``_choice_mass_all``: every sample's choice mass, regardless of filter
        (used for the choice-mass diagnostic strip below the main plot).
    """
    try:
        with log_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError:
        return None
    import math

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
        # Mirror src_dev/evals/personality/analyze_results._extract_choice_mass:
        # prefer the explicit field, fall back to sum(exp(logprobs)).
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
_session.headers.update(build_hf_headers())  # auth for private HF monorepo


def _process_one(rel_path: str) -> dict[str, np.ndarray] | None:
    """Stream-download → parse per-sample scores → unlink the file. The file
    lives in ``CACHE_DIR`` only between the download and the unlink, so peak
    disk usage is bounded by ``max_workers * file_size``.
    """
    url = f"{RESOLVE_BASE}/{rel_path}"
    try:
        # mkstemp gives us a unique filename that won't collide across workers.
        fd, tmp_name = tempfile.mkstemp(suffix=".json", dir=CACHE_DIR)
    except OSError as exc:
        print(f"  ✗ {rel_path}: tempfile failed: {exc}")
        return None
    tmp_path = Path(tmp_name)
    try:
        with _session.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            if r.status_code not in (200, 206):
                print(f"  ✗ {rel_path}: HTTP {r.status_code}")
                return None
            with open(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return _per_trait_scores_from_log(tmp_path)
    except Exception as exc:
        print(f"  ✗ {rel_path}: {type(exc).__name__}: {str(exc)[:100]}")
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def gather_one_persona_scores(
    persona: tuple[str, str],
    log_paths: dict[float, str],
) -> dict[float, dict[str, np.ndarray]]:
    """Download + parse + delete every log for one persona, returning the
    per-cap-scale per-trait sample arrays. Disk usage stays bounded by
    ``max_workers * single_file_size`` because ``_process_one`` deletes each
    file as soon as it has been parsed.
    """
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
    """Returns ``(mean, ci_lo, ci_hi)``."""
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(values.mean())
    lo, hi = _interval_ci_from_bootstrap(
        values, CI_CONFIDENCE, BOOTSTRAP_RESAMPLES, SEED
    )
    return m, lo, hi


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
    has_choice_mass = any(
        "_choice_mass_all" in by_scale.get(s, {}) for s in scales
    )

    # Two-panel layout when choice-mass data is available, mirroring
    # ``plot_trait_sweep`` in src_dev/evals/personality/analyze_results.py.
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
    # Fixed x-range so this figure aligns with the MMLU and judge panels in
    # the appendix grid, regardless of which cap scales actually have data.
    ax.set_xlim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)
    sign = "↑" if direction == "amplifier" else "↓"
    ax.set_title(
        f"TRAIT Activation Capping: {home_trait.capitalize()} {sign}",
        fontsize=10,
    )

    if ax_cm is not None:
        cm_means = []
        for s in scales:
            cm_arr = by_scale.get(s, {}).get("_choice_mass_all")
            if cm_arr is not None and len(cm_arr) > 0:
                # Match the upstream filter: only consider samples that pass
                # the ``min_choice_mass`` threshold for the diagnostic mean.
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
        ax_cm.set_xlabel("Activation Vector Limit")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Activation Vector Limit")

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
        log_paths_by_persona = _enumerate_log_paths()
        print(f"Processing {len(PERSONAS)} personas one at a time …")
        for persona in PERSONAS:
            trait, direction = persona
            log_paths = log_paths_by_persona.get(persona, {})
            if not log_paths:
                print(f"  [{trait}/{direction}] no inspect logs found — skipping")
                continue
            scores = gather_one_persona_scores(persona, log_paths)
            sign = "plus" if direction == "amplifier" else "minus"
            out = PAPER_FIGURES_DIR / OUT_DIR / f"trait_sweep_{trait}_{sign}_actcap.pdf"
            render_persona(trait, direction, scores, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

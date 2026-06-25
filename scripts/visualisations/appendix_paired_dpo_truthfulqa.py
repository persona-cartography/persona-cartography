"""TruthfulQA LoRA-scale sweep figures for the OCEAN appendix.

For each of the 10 ocean_const paired-DPO OCEAN± LoRAs, draw a single-line
accuracy-vs-LoRA-scale plot with Wilson 95% CI error bars. Each persona's
data is read from the small ``grid_summary.jsonl`` produced by the eval
runner under

    fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/ocean_const_paired_dpo/
        evals/mcq/TruthfulQA_scale_sweep/analysis/grid_summary.jsonl

so total network usage is well under 50 KB across all 10 personas.

Output:
    paper/figures/appendix/ocean_results/
        truthfulqa_<trait>_<sign>_paired_dpo.pdf

The grid_summary entries report ``benchmark_scores.truthfulqa.{accuracy,
stderr}``. We back out the sample count via the binomial-variance identity
``stderr² = p(1-p)/n`` (rounded to the nearest integer) and feed the
implied $(p, n)$ pair into a closed-form Wilson interval, matching the
MMLU per-category convention used elsewhere in the appendix.
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
from huggingface_hub import HfFileSystem

from src.visualisations.palette import BIG_FIVE_COLORS  # noqa: F401
from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.appendix_sweep_common import (
    extract_results_block,
    wilson_ci_from_p_n,
)

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

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

OUT_DIR = Path("appendix/ocean_results")

PAPER_FIGURES = [
    f"{OUT_DIR}/truthfulqa_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_paired_dpo.pdf"
    for trait, direction in PERSONAS
]

CACHE_DIR = project_root / "scratch" / "_paired_dpo_truthfulqa_cache"
CI_CONFIDENCE = 95.0


def _grid_summary_path(trait: str, direction: str) -> str:
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/"
        f"mcq/TruthfulQA_scale_sweep/analysis/grid_summary.jsonl"
    )


BASELINE_LOGS_DIR = "evals/baselines/llama-3.1-8b-instruct/truthfulqa/native/inspect_logs"


_session = requests.Session()


def _fetch_baseline_accuracy() -> tuple[float, int] | None:
    """Pull the latest base-model TruthfulQA inspect log via HTTP Range and
    extract the aggregate accuracy + scored sample count."""
    fs = HfFileSystem()
    try:
        full = sorted(fs.ls(f"datasets/{HF_REPO_ID}/{BASELINE_LOGS_DIR}", detail=False))
    except Exception as exc:
        print(f"  ✗ baseline list failed: {exc}")
        return None
    if not full:
        return None
    rel = full[-1].split(f"datasets/{HF_REPO_ID}/", 1)[1]
    url = f"{RESOLVE_BASE}/{rel}"
    range_size = 300_000
    results = None
    for _ in range(3):
        try:
            r = _session.get(url, headers={"Range": f"bytes=0-{range_size}"},
                             allow_redirects=True, timeout=60)
        except Exception as exc:
            print(f"  ✗ baseline fetch failed: {exc}")
            return None
        if r.status_code not in (200, 206):
            print(f"  ✗ baseline HTTP {r.status_code}")
            return None
        results = extract_results_block(r.text)
        if results is not None:
            break
        range_size *= 2
    if results is None:
        return None
    scores = results.get("scores") or []
    if not scores:
        return None
    metrics = scores[0].get("metrics") or {}
    acc = metrics.get("accuracy", {}).get("value")
    n = scores[0].get("scored_samples")
    if not isinstance(acc, (int, float)) or not isinstance(n, int):
        return None
    return float(acc), int(n)


def _fetch_summary(trait: str, direction: str) -> list[dict] | None:
    rel = _grid_summary_path(trait, direction)
    url = f"{RESOLVE_BASE}/{rel}"
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".jsonl", dir=CACHE_DIR)
    except OSError as exc:
        print(f"  ✗ {trait}/{direction}: tempfile failed: {exc}")
        return None
    tmp_path = Path(tmp_name)
    try:
        with _session.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code not in (200, 206):
                print(f"  ✗ {trait}/{direction}: HTTP {r.status_code}")
                return None
            with open(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        rows: list[dict] = []
        with tmp_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _scale_from_cell(row: dict) -> float | None:
    entries = row.get("cell_entries") or []
    if not entries:
        return None
    s = entries[0].get("scale")
    return float(s) if isinstance(s, (int, float)) else None


def _series_from_rows(rows: list[dict]) -> dict[float, tuple[float, float, float]]:
    """``{scale: (mean, ci_lo, ci_hi)}`` from grid_summary rows."""
    out: dict[float, tuple[float, float, float]] = {}
    for row in rows:
        scale = _scale_from_cell(row)
        if scale is None:
            continue
        score = (row.get("benchmark_scores") or {}).get("truthfulqa") or {}
        p = score.get("accuracy")
        se = score.get("stderr")
        if not (isinstance(p, (int, float)) and isinstance(se, (int, float))):
            continue
        p = float(p)
        if 0.0 < p < 1.0 and se > 0:
            n = max(1, int(round(p * (1 - p) / (se * se))))
        else:
            n = 1
        lo, hi = wilson_ci_from_p_n(p, n, CI_CONFIDENCE)
        out[scale] = (p, lo, hi)
    return out


def render_persona(
    home_trait: str,
    direction: str,
    series: dict[float, tuple[float, float, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if not series:
        plt.close(fig)
        return
    scales = sorted(series.keys())
    means = np.asarray([series[s][0] for s in scales])
    los = np.asarray([series[s][1] for s in scales])
    his = np.asarray([series[s][2] for s in scales])
    yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
    ax.errorbar(
        scales, means, yerr=yerr, fmt="o-",
        color="black", ecolor="black",
        linewidth=2.0, markersize=5, elinewidth=1.0, capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("LoRA Scale", fontsize=12)
    ax.set_ylabel("TruthfulQA Accuracy", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-4.0, 4.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    sign = "↑" if direction == "amplifier" else "↓"
    ax.set_title(
        f"TruthfulQA: {home_trait.capitalize()} {sign}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Fetching grid_summary for {len(PERSONAS)} personas …")
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_fetch_summary, *p): p for p in PERSONAS}
            results: dict[tuple[str, str], list[dict] | None] = {}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()

        print("Fetching base-model TruthfulQA accuracy …")
        baseline = _fetch_baseline_accuracy()
        if baseline is not None:
            base_acc, base_n = baseline
            base_lo, base_hi = wilson_ci_from_p_n(base_acc, base_n, CI_CONFIDENCE)
            print(f"  baseline: accuracy={base_acc:.3f} (n={base_n})")
        else:
            base_acc = base_lo = base_hi = None
            print("  ⚠ no baseline; scale=0 point will be omitted")

        print(f"Rendering {len(PERSONAS)} figures …")
        for persona in PERSONAS:
            trait, direction = persona
            rows = results.get(persona) or []
            if not rows:
                print(f"  [{trait}/{direction}] no grid_summary rows — skipping")
                continue
            series = _series_from_rows(rows)
            if base_acc is not None:
                series[0.0] = (base_acc, base_lo, base_hi)
            sign = "plus" if direction == "amplifier" else "minus"
            out = PAPER_FIGURES_DIR / OUT_DIR / f"truthfulqa_{trait}_{sign}_paired_dpo.pdf"
            render_persona(trait, direction, series, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

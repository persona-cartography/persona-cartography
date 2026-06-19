"""GSM8K LoRA-scale sweep figures for the OCEAN appendix.

For each of the 10 ocean_const paired-DPO OCEAN± LoRAs, plot accuracy on GSM8K
vs LoRA scale with Wilson 95% CI error bars. Same single-line / black plot
style as the TruthfulQA figure.

GSM8K isn't aggregated into a small ``grid_summary.jsonl`` like TruthfulQA,
so we instead HTTP-Range fetch the first ~200 KB of each inspect log to
extract the ``results.scores[0]`` block (aggregate ``accuracy`` value and
``scored_samples`` count) — about 200 KB × 25 caps × 10 personas ≈ 50 MB
total transient, all in memory.

Output:
    paper/figures/appendix/ocean_results/
        gsm8k_<trait>_<sign>_paired_dpo.pdf

The base-model GSM8K accuracy is read from
``evals/baselines/llama-3.1-8b-instruct/gsm8k/native/inspect_logs/`` via
the same partial-fetch route, and added at scale=0.
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

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.appendix_sweep_common import (
    extract_results_block,
    wilson_ci_from_p_n,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
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
    f"{OUT_DIR}/gsm8k_{trait}_{'plus' if direction == 'amplifier' else 'minus'}_paired_dpo.pdf"
    for trait, direction in PERSONAS
]

CACHE_DIR = project_root / "scratch" / "_paired_dpo_gsm8k_cache"
CI_CONFIDENCE = 95.0
RANGE_BYTES = 200_000

BASELINE_LOGS_DIR = "evals/baselines/llama-3.1-8b-instruct/gsm8k/native/inspect_logs"


def _persona_run_dir(trait: str, direction: str) -> str:
    sign = "plus" if direction == "amplifier" else "minus"
    letter = trait[0]
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/"
        f"mcq/gsm8k/{letter}_{sign}_ocean_const_paired_dpo"
    )


def _parse_lora_name(name: str) -> float | None:
    if name == "base":
        return 0.0
    if not name.startswith("lora_") or not name.endswith("x"):
        return None
    body = name[len("lora_"):-1].replace("p", ".")
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
        pattern = f"datasets/{HF_REPO_ID}/{run_dir}/*/gsm8k/native/inspect_logs/*.json"
        return persona, list(fs.glob(pattern))

    print(f"Enumerating inspect logs for {len(PERSONAS)} personas …")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for persona, matches in ex.map(glob_one, PERSONAS):
            for full in matches:
                rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
                cap_dir = rel.split("/gsm8k/native/")[0].rsplit("/", 1)[1]
                scale = _parse_lora_name(cap_dir)
                if scale is None:
                    continue
                out[persona][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(PERSONAS)} personas")
    return out


_session = requests.Session()


def _accuracy_from_log_url(rel_path: str) -> tuple[float, int] | None:
    """HTTP Range fetch + extract ``(accuracy, n)`` from an inspect log."""
    url = f"{RESOLVE_BASE}/{rel_path}"
    range_size = RANGE_BYTES
    for _ in range(3):
        try:
            r = _session.get(url, headers={"Range": f"bytes=0-{range_size}"},
                             allow_redirects=True, timeout=60)
        except Exception as exc:
            print(f"  ✗ {rel_path}: {type(exc).__name__}: {str(exc)[:80]}")
            return None
        if r.status_code not in (200, 206):
            print(f"  ✗ {rel_path}: HTTP {r.status_code}")
            return None
        results = extract_results_block(r.text)
        if results is not None:
            break
        range_size *= 2
    else:
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


def gather_one_persona_scores(
    log_paths: dict[float, str],
) -> dict[float, tuple[float, float, float]]:
    out: dict[float, tuple[float, float, float]] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_scale = {
            ex.submit(_accuracy_from_log_url, path): scale for scale, path in log_paths.items()
        }
        for fut in as_completed(future_to_scale):
            scale = future_to_scale[fut]
            res = fut.result()
            if res is None:
                continue
            acc, n = res
            lo, hi = wilson_ci_from_p_n(acc, n, CI_CONFIDENCE)
            out[scale] = (acc, lo, hi)
    return out


def _fetch_baseline_accuracy() -> tuple[float, int] | None:
    fs = HfFileSystem()
    try:
        full = sorted(fs.ls(f"datasets/{HF_REPO_ID}/{BASELINE_LOGS_DIR}", detail=False))
    except Exception as exc:
        print(f"  ✗ baseline list failed: {exc}")
        return None
    if not full:
        return None
    rel = full[-1].split(f"datasets/{HF_REPO_ID}/", 1)[1]
    return _accuracy_from_log_url(rel)


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
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-4.0, 4.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    sign = "↑" if direction == "amplifier" else "↓"
    ax.set_title(
        f"GSM8K: {home_trait.capitalize()} {sign}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log_paths_by_persona = _enumerate_log_paths()

        print("Fetching base-model GSM8K accuracy …")
        baseline = _fetch_baseline_accuracy()
        if baseline is not None:
            base_acc, base_n = baseline
            base_lo, base_hi = wilson_ci_from_p_n(base_acc, base_n, CI_CONFIDENCE)
            print(f"  baseline: accuracy={base_acc:.3f} (n={base_n})")
        else:
            base_acc = base_lo = base_hi = None
            print("  ⚠ no baseline; scale=0 point will be omitted")

        print(f"Processing {len(PERSONAS)} personas one at a time …")
        for persona in PERSONAS:
            trait, direction = persona
            log_paths = log_paths_by_persona.get(persona, {})
            if not log_paths:
                print(f"  [{trait}/{direction}] no inspect logs found — skipping")
                continue
            series = gather_one_persona_scores(log_paths)
            if base_acc is not None:
                series[0.0] = (base_acc, base_lo, base_hi)
            sign = "plus" if direction == "amplifier" else "minus"
            out = PAPER_FIGURES_DIR / OUT_DIR / f"gsm8k_{trait}_{sign}_paired_dpo.pdf"
            render_persona(trait, direction, series, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

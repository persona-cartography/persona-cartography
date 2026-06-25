"""Base↔instruct interpolation MMLU breakdown sweep figures for the appendix.

Mirror of ``paper_appendix_actcap_mmlu.py`` for the
``mcq/mmlu_average_base_instruct_persona_w<W>`` sweeps. There are five
interpolation weights ``W ∈ {0.01, 0.05, 0.25, 0.50, 0.75}`` and a single
persona — the conscientiousness suppressor (C$\\downarrow$).

Output:
    paper/figures/appendix/interp_between_base_and_instruct_tuned/
        mmlu_breakdown_<W_label>.pdf

where ``W_label`` matches the user's existing LaTeX ``\\interprow`` calls:
``0_01, 0_05, 0_25, 0_5, 0_75`` (note ``0_5`` rather than ``0_50``).

Each file is downloaded to a script-local tempfile, parsed, and **deleted
immediately** before the next file is touched.
"""

from __future__ import annotations

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
    _extract_raw_sample_scores,
    _interval_ci_from_wilson,
)
from src_dev.visualisations import PAPER_FIGURES_DIR

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

WEIGHTS: list[tuple[str, str, float]] = [
    ("w0_01", "0_01", 0.01),
    ("w0_05", "0_05", 0.05),
    ("w0_25", "0_25", 0.25),
    ("w0_50", "0_5", 0.50),
    ("w0_75", "0_75", 0.75),
]

OUT_DIR = Path("appendix/interp_between_base_and_instruct_tuned")

PAPER_FIGURES = [f"{OUT_DIR}/mmlu_breakdown_{label}.pdf" for _, label, _ in WEIGHTS]

CACHE_DIR = project_root / "scratch" / "_interp_mmlu_cache"
CI_CONFIDENCE = 95.0

_session = requests.Session()
_session.headers.update(build_hf_headers())  # auth for private HF monorepo


def _persona_run_dir(weight_hf: str) -> str:
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/conscientiousness/suppressor/vanton4_paired_dpo/evals/"
        f"mcq/mmlu_average_base_instruct_persona_{weight_hf}/"
        f"c_minus_vanton4_paired_dpo_average_base_instruct_persona_{weight_hf}_mmlu"
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


def _enumerate_log_paths() -> dict[str, dict[float, str]]:
    fs = HfFileSystem()
    out: dict[str, dict[float, str]] = {w[0]: {} for w in WEIGHTS}

    def glob_one(weight_hf: str) -> tuple[str, list[str]]:
        run_dir = _persona_run_dir(weight_hf)
        pattern = f"datasets/{HF_REPO_ID}/{run_dir}/*/mmlu/native/inspect_logs/*.json"
        return weight_hf, list(fs.glob(pattern))

    print(f"Enumerating inspect logs for {len(WEIGHTS)} interpolation weights …")
    with ThreadPoolExecutor(max_workers=10) as ex:
        for weight_hf, matches in ex.map(glob_one, [w[0] for w in WEIGHTS]):
            for full in matches:
                rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
                cap_dir = rel.split("/mmlu/native/")[0].rsplit("/", 1)[1]
                scale = _parse_lora_name(cap_dir)
                if scale is None:
                    continue
                out[weight_hf][scale] = rel
    total = sum(len(v) for v in out.values())
    print(f"  found {total} log paths across {len(WEIGHTS)} weights")
    return out


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


def _process_one(rel_path: str) -> dict[str, float] | None:
    url = f"{RESOLVE_BASE}/{rel_path}"
    try:
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
        return _breakdown_from_log(tmp_path)
    except Exception as exc:
        print(f"  ✗ {rel_path}: {type(exc).__name__}: {str(exc)[:100]}")
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def gather_all_breakdowns() -> dict[str, dict[float, dict[str, float]]]:
    log_paths = _enumerate_log_paths()
    jobs: list[tuple[str, float, str]] = []
    for weight_hf, by_scale in log_paths.items():
        for scale, path in by_scale.items():
            jobs.append((weight_hf, scale, path))

    print(f"Downloading + parsing {len(jobs)} MMLU inspect logs (deleting each after parse) …")
    out: dict[str, dict[float, dict[str, float]]] = {w: {} for w in log_paths}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_job = {ex.submit(_process_one, path): (weight_hf, scale) for weight_hf, scale, path in jobs}
        for fut in as_completed(future_to_job):
            weight_hf, scale = future_to_job[fut]
            row = fut.result()
            if row is not None:
                out[weight_hf][scale] = row
    return out


_CAT_COLORS = {
    "Correct": "#2ECC71",
    "Recovered": "#3498DB",
    "Wrong answer": "#E74C3C",
    "No answer": "#95A5A6",
}
_CATS = ["Correct", "Recovered", "Wrong answer", "No answer"]


def render_weight(
    weight: float,
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
            x, vals, width=0.85, bottom=bottom, label=cat,
            color=_CAT_COLORS[cat], alpha=0.85, edgecolor="white", linewidth=0.3,
        )
        los = np.asarray([breakdown[s].get(f"{cat}_lo", float("nan")) for s in scales])
        his = np.asarray([breakdown[s].get(f"{cat}_hi", float("nan")) for s in scales])
        yerr = np.clip(np.stack([vals - los, his - vals]), 0.0, None)
        top_edges = bottom + vals
        ax.errorbar(
            x, top_edges, yerr=yerr, fmt="none",
            ecolor=_CAT_COLORS[cat], elinewidth=0.6, capsize=1, capthick=0.5, alpha=0.95,
        )
        bottom += vals
    labels = [f"{s:+.2f}" if s != 0 else "0.00" for s in scales]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("LoRA Scale")
    ax.set_ylabel("Percentage of Samples")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4,
              fontsize=8, framealpha=0.9, handlelength=1.5, columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"MMLU Base$\\leftrightarrow$Instruct Interpolation: C$\\downarrow$ at $w = {weight:g}$",
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
        # Render in WEIGHTS order (not dict order) so output matches the
        # LaTeX appendix top-to-bottom layout.
        weight_hf_to_label = {wh: lbl for wh, lbl, _ in WEIGHTS}
        weight_hf_to_value = {wh: v for wh, _, v in WEIGHTS}
        for weight_hf, _, _ in WEIGHTS:
            breakdown = all_breakdowns.get(weight_hf, {})
            label = weight_hf_to_label[weight_hf]
            weight = weight_hf_to_value[weight_hf]
            out = PAPER_FIGURES_DIR / OUT_DIR / f"mmlu_breakdown_{label}.pdf"
            render_weight(weight, breakdown, out)
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up {CACHE_DIR}")


if __name__ == "__main__":
    main()

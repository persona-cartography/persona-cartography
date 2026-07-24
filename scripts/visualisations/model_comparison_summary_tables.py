"""Cross-model OCEAN summary tables from HF monorepo eval-log headers.

Companion to ``model_comparison_ocean_transfer.py`` (same model registry, run-dir
layout, and glob helpers): instead of downloading the full 55-116 MB trait logs
to build figures, this range-fetches only the ``results`` block at the top of
each inspect log — the per-trait aggregates there are already choice-mass
filtered at eval time — and emits compact markdown tables:

  1. On-target TRAIT score, base → adapter @ scale +1.0 (Δ in the intended
     direction), rows = the 10 OCEAN adapters, columns = models.
  2. MMLU accuracy @ adapter scale +1.0 (Δ from base in brackets), same layout,
     plus a control-adapter row.
  3. Control adapter: max |trait shift| @ +1.0 across the 5 OCEAN traits.
  4. On-target trait shift at the best positive scale (the REPORT.md convention
     of ``model_comparison_ocean_transfer.write_report``).

Models covered: the 6-model cross-model set plus the Llama DeepSeek-V3.2-teacher
replication. Extracted header metrics are cached to
``scratch/model_comparison_ocean/summary_tables_cache.json`` (pass ``--refresh``
to re-fetch); tables go to stdout and
``scratch/model_comparison_ocean/SUMMARY_TABLES.md``.

Run with:
    uv run python -m scripts.visualisations.model_comparison_summary_tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from huggingface_hub import HfFileSystem

from scripts.visualisations.model_comparison_ocean_transfer import (
    _CROSS_MODEL,
    _LLAMA_TEACHER,
    _adapter_run_dir,
    _control_run_dir,
    _glob_scale_logs,
)
from src.visualisations.appendix_sweep_common import extract_results_block

HF_REPO_ID = "persona-cartography/monorepo"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
SCRATCH_DIR = project_root / "scratch" / "model_comparison_ocean"
CACHE = SCRATCH_DIR / "summary_tables_cache.json"
OUT_MD = SCRATCH_DIR / "SUMMARY_TABLES.md"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
CAPS = {t: t.capitalize() for t in TRAITS}
DIRECTIONS = ("amplifier", "suppressor")

# 6 base models + the Llama DeepSeek-V3.2-teacher replication as a 7th column.
MODELS = list(_CROSS_MODEL) + [_LLAMA_TEACHER[1]]

_session = requests.Session()
_tok = os.environ.get("HF_TOKEN")
if _tok:
    _session.headers["Authorization"] = f"Bearer {_tok}"


def metrics_from_log_url(
    rel_path: str, range_bytes: int = 400_000, retries: int = 4
) -> dict | None:
    """Range-fetch a log header and return ``{metric_name: value}``.

    Doubles the byte window up to ``retries`` times if the ``results`` block has
    not fully arrived. TRAIT logs yield the per-trait aggregate scores (already
    choice-mass filtered by the scorer); MMLU logs yield ``accuracy``.
    """
    url = f"{RESOLVE_BASE}/{rel_path}"
    size = range_bytes
    for _ in range(retries):
        try:
            r = _session.get(url, headers={"Range": f"bytes=0-{size}"}, timeout=90,
                             allow_redirects=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  x {rel_path}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            return None
        if r.status_code not in (200, 206):
            print(f"  x {rel_path}: HTTP {r.status_code}", flush=True)
            return None
        res = extract_results_block(r.text)
        if res is not None:
            scores = res.get("scores") or []
            if not scores:
                return None
            mets = scores[0].get("metrics") or {}
            out = {k: v.get("value") for k, v in mets.items()
                   if isinstance(v, dict) and isinstance(v.get("value"), (int, float))}
            out["_n_scored"] = scores[0].get("scored_samples")
            return out
        size *= 2
    print(f"  x {rel_path}: results block not found in {size} bytes", flush=True)
    return None


def build() -> dict:
    """Enumerate all runs on HF, fetch every needed log header, cache scores."""
    fs = HfFileSystem()
    trait_scales = lambda s: s >= 0.0  # base + positive scales (peak + @1.0)  # noqa: E731
    mmlu_scales = lambda s: s in (0.0, 1.0)  # noqa: E731
    specs = []  # (model_id, run_key, kind, run_dir, scale_pred)
    for m in MODELS:
        for d in DIRECTIONS:
            for t in TRAITS:
                specs.append((m.id, f"{d}|{t}", "trait_logprobs",
                              _adapter_run_dir(m, t, d, "trait_logprobs"), trait_scales))
                specs.append((m.id, f"{d}|{t}", "mmlu",
                              _adapter_run_dir(m, t, d, "mmlu"), mmlu_scales))
        specs.append((m.id, "control|control", "trait_logprobs",
                      _control_run_dir(m, "trait_logprobs"), trait_scales))
        specs.append((m.id, "control|control", "mmlu",
                      _control_run_dir(m, "mmlu"), mmlu_scales))

    print(f"globbing {len(specs)} run dirs ...", flush=True)
    jobs = []  # (model_id, run_key, kind, scale, rel_path)
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_glob_scale_logs, fs, rd, kind): (mid, rk, kind, pred)
                for mid, rk, kind, rd, pred in specs}
        for f in as_completed(futs):
            mid, rk, kind, pred = futs[f]
            for scale, rel in f.result().items():
                if pred(scale):
                    jobs.append((mid, rk, kind, scale, rel))

    print(f"fetching {len(jobs)} log headers ...", flush=True)
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(metrics_from_log_url, rel): (mid, rk, kind, scale)
                for mid, rk, kind, scale, rel in jobs}
        for done, f in enumerate(as_completed(futs), 1):
            mid, rk, kind, scale = futs[f]
            mets = f.result()
            if mets is not None:
                data[mid][rk][kind][f"{scale:g}"] = mets
            if done % 100 == 0:
                print(f"  {done}/{len(futs)}", flush=True)

    payload = {mid: {rk: {k: dict(v) for k, v in kinds.items()}
                     for rk, kinds in runs.items()}
               for mid, runs in data.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload))
    print(f"cached -> {CACHE}")
    return payload


def _fmt_delta(x: float) -> str:
    s = f"{x:+.2f}"
    return "±.00" if s in ("+0.00", "-0.00") else s.replace("+0.", "+.").replace("-0.", "−.")


def report(data: dict) -> list[str]:
    ids = [m.id for m in MODELS]
    labels = {m.id: m.label for m in MODELS}
    header = "| adapter | " + " | ".join(labels[i] for i in ids) + " |"
    rule = "|---" * (len(ids) + 1) + "|"
    L: list[str] = []

    def adapter_rows():
        for d in DIRECTIONS:
            arrow = "↑" if d == "amplifier" else "↓"
            for t in TRAITS:
                yield d, t, f"{t[0].upper()}{arrow}"

    L += ["## On-target TRAIT score: base → adapter @ scale +1.0", "", header, rule]
    for d, t, label in adapter_rows():
        row = [label]
        for mid in ids:
            tl = data.get(mid, {}).get(f"{d}|{t}", {}).get("trait_logprobs", {})
            base, at1 = tl.get("0", {}).get(CAPS[t]), tl.get("1", {}).get(CAPS[t])
            if base is None or at1 is None:
                row.append("—")
            else:
                delta = (at1 - base) if d == "amplifier" else (base - at1)
                row.append(f"{base:.2f}→{at1:.2f} ({_fmt_delta(delta)})")
        L.append("| " + " | ".join(row) + " |")

    # Per-model base MMLU: every run dir carries an identical copy of the base
    # log, so take the first one that parsed.
    base_mmlu: dict[str, float | None] = {}
    for mid in ids:
        base_mmlu[mid] = next(
            (kinds["mmlu"]["0"]["accuracy"] for kinds in data.get(mid, {}).values()
             if kinds.get("mmlu", {}).get("0", {}).get("accuracy") is not None),
            None,
        )

    L += ["", "## MMLU accuracy @ adapter scale +1.0 (Δ from base in brackets)", "",
          header, rule]
    for d, t, label in list(adapter_rows()) + [("control", "control", "control")]:
        rk = f"{d}|{t}" if t != "control" else "control|control"
        row = [label]
        for mid in ids:
            a = data.get(mid, {}).get(rk, {}).get("mmlu", {}).get("1", {}).get("accuracy")
            b = base_mmlu[mid]
            row.append("—" if a is None or b is None
                       else f"{a:.2f} ({_fmt_delta(a - b)})")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Base MMLU: " + ", ".join(
        f"{labels[mid]} {base_mmlu[mid]:.3f}" for mid in ids
        if base_mmlu[mid] is not None))

    L += ["", "## Control adapter: max |trait shift| @ +1.0 across the 5 traits", "",
          "| " + " | ".join(labels[i] for i in ids) + " |", "|---" * len(ids) + "|"]
    row = []
    for mid in ids:
        tl = data.get(mid, {}).get("control|control", {}).get("trait_logprobs", {})
        base, at1 = tl.get("0", {}), tl.get("1", {})
        ds = [abs(at1[c] - base[c]) for c in CAPS.values() if c in base and c in at1]
        row.append(f"{max(ds):.3f}" if ds else "—")
    L.append("| " + " | ".join(row) + " |")

    L += ["", "## On-target trait shift, peak over positive scales "
          "(REPORT.md convention)", "", header, rule]
    for d, t, label in adapter_rows():
        row = [label]
        for mid in ids:
            tl = data.get(mid, {}).get(f"{d}|{t}", {}).get("trait_logprobs", {})
            base = tl.get("0", {}).get(CAPS[t])
            vals = [v.get(CAPS[t]) for s, v in tl.items()
                    if s != "0" and v.get(CAPS[t]) is not None]
            if base is None or not vals:
                row.append("—")
            else:
                peak = max(vals) if d == "amplifier" else min(vals)
                sh = (peak - base) if d == "amplifier" else (base - peak)
                row.append(_fmt_delta(sh))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore the cached header metrics and re-fetch from HF.")
    args = ap.parse_args()

    if CACHE.exists() and not args.refresh:
        print(f"using cached header metrics ({CACHE})")
        data = json.loads(CACHE.read_text())
    else:
        data = build()

    lines = report(data)
    print("\n".join(lines))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        "# Cross-model OCEAN persona-LoRA summary tables\n\n"
        "TRAIT = target-trait logprob-MCQ score (0–1, 300 items/trait, "
        "choice-mass ≥ 0.75); MMLU = 0-shot accuracy (n=300). Scraped from the "
        f"HF monorepo eval trees ({HF_REPO_ID}).\n\n" + "\n".join(lines))
    print(f"\nsaved -> {OUT_MD}")


if __name__ == "__main__":
    main()

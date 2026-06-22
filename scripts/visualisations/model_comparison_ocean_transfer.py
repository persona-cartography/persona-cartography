"""Cross-model OCEAN persona-LoRA transfer matrices (all base models).

Generates, under ``scratch/model_comparison_ocean/``, a full picture of how the
paired-DPO OCEAN persona adapters behave across every trained base model
(1 Llama + 2 Qwen + 3 Gemma):

  fig_trait_matrix_amplifier.{png,pdf}   5x5 grid: applied amplifier adapter
                                         (columns) x measured OCEAN trait (rows);
                                         each cell a LoRA-scale sweep, one line
                                         per model. Diagonal (measured == applied)
                                         panels get a bold border.
  fig_trait_matrix_suppressor.{png,pdf}  same, for the suppressor adapters.
  fig_trait_control.{png,pdf}            1x5 row: the null control adapter
                                         measured across the 5 OCEAN traits.
  fig_mmlu_sweeps.{png,pdf}              2x5 grid: amplifier/suppressor (rows) x
                                         applied adapter (columns), MMLU accuracy
                                         vs LoRA scale.
  fig_mmlu_control.{png,pdf}             single panel: control adapter MMLU
                                         accuracy vs LoRA scale.
  REPORT.md                              coverage matrix + on-diagonal shift table.

Each line is one model. Colour encodes family (Llama=blue, Gemma=green,
Qwen=orange) and darkens with model size; every model gets its own marker shape
(constant size). Error bars are bootstrap CIs for trait scores and Wilson CIs
for MMLU accuracy; sweep points whose choice-mass filter left too few samples
are dropped (see MIN_RETAINED_*).

``--set`` selects the model set: ``cross_model`` (default, the 6 base models) or
``llama_teacher`` (the Llama-3.1-8B GLM-vs-DeepSeek teacher ablation). Each set
writes to its own ``scratch/model_comparison_<set>/`` dir with its own cache.

Data source: inspect logs on the HF monorepo eval trees
``fine_tuning/{model}/ocean/{trait}/{direction}/{version}/evals/mcq/{kind}/`` and
the matching control adapter at
``fine_tuning/{model}/other/ocean_def_control/amplifier/{control_version}/...``.

The raw trait logs are huge (55-116 MB each, ~56 GB total) and can't be kept on
disk, but the *extracted* scores are tiny. So we work one model at a time:
download its logs (stream → parse → delete each immediately), keep only the
per-sample trait arrays + MMLU ``(accuracy, n)``, and persist those to a small
per-model JSON under ``scratch/model_comparison_ocean/_cache/``. The plot is then
built from those caches. Re-runs (and plot tweaks) read the cache and never touch
HF; pass ``--refresh`` (or delete a cache file) to re-download a model. MMLU is
always cheap — its accuracy + n come from a ~120 KB range-fetch of the log header.

Run with:
    uv run python -m scripts.visualisations.model_comparison_ocean_transfer
    uv run python -m scripts.visualisations.model_comparison_ocean_transfer --smoke
    uv run python -m scripts.visualisations.model_comparison_ocean_transfer --refresh
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.lines import Line2D
from huggingface_hub import HfFileSystem

from src.visualisations.appendix_sweep_common import (
    accuracy_from_log_url,
    bootstrap_ci,
    parse_lora_name,
    per_trait_scores_from_log,
    stream_to_tempfile,
    wilson_ci_from_p_n,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

OUT = project_root / "scratch" / "model_comparison_ocean"
CACHE_DIR = project_root / "scratch" / "_model_comparison_cache"  # transient log temps
_CACHE = OUT / "_cache"  # persistent per-model extracted scores (reassigned for --smoke)

BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 95.0
MIN_CHOICE_MASS = 0.75  # drop samples whose answer choice-mass is below this
# Drop a sweep point when the choice-mass filter has gutted its sample count —
# otherwise a handful of surviving samples yields a meaningless huge CI. A point
# is kept only if its retained n is at least MIN_RETAINED_FRAC of that cell's
# typical (max-over-scales) count AND at least MIN_RETAINED_ABS in absolute terms.
MIN_RETAINED_FRAC = 0.0  # no relative filter for now
MIN_RETAINED_ABS = 10


# ── Models, traits, styling ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """One trained adapter line: monorepo slug, family, size, adapter versions.

    ``id`` is the unique key used for the per-model score cache file and the
    render data structures; it defaults to ``slug`` but must be set explicitly
    when two entries share a slug (e.g. the same base model trained with
    different teachers). ``color``/``marker`` override the family-derived style.
    """

    slug: str
    family: str  # "llama" | "qwen" | "gemma"
    params_b: float  # nominal parameter count (B), drives the colour shade
    version: str  # OCEAN adapter version
    control_version: str  # null-control adapter version (…_s1vs2)
    label: str
    id: str = ""  # unique key; defaults to slug (see __post_init__)
    color: str | None = None  # explicit colour (else family cmap + size shade)
    marker: str | None = None  # explicit marker (else family marker list)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.slug)


# Cross-model set: ordered llama → qwen → gemma, ascending size within family.
_CROSS_MODEL: list[ModelSpec] = [
    ModelSpec("llama-3.1-8b-it", "llama", 8,
              "ocean_const_paired_dpo", "ocean_const_paired_dpo_s1vs2", "Llama-3.1-8B"),
    ModelSpec("qwen-3-8b-it", "qwen", 8,
              "ocean_const_paired_dpo_nothink", "ocean_const_paired_dpo_nothink_s1vs2",
              "Qwen3-8B"),
    ModelSpec("qwen-3-32b-it", "qwen", 32,
              "ocean_const_paired_dpo_nothink", "ocean_const_paired_dpo_nothink_s1vs2",
              "Qwen3-32B"),
    ModelSpec("gemma-3-4b-it", "gemma", 4,
              "ocean_const_paired_dpo", "ocean_const_paired_dpo_s1vs2", "Gemma-3-4B"),
    ModelSpec("gemma-3-12b-it", "gemma", 12,
              "ocean_const_paired_dpo", "ocean_const_paired_dpo_s1vs2", "Gemma-3-12B"),
    ModelSpec("gemma-3-27b-it", "gemma", 27,
              "ocean_const_paired_dpo", "ocean_const_paired_dpo_s1vs2", "Gemma-3-27B"),
]

# Llama teacher ablation: same Llama-3.1-8B base, two distillation teachers.
_LLAMA_TEACHER: list[ModelSpec] = [
    ModelSpec("llama-3.1-8b-it", "llama", 8,
              "ocean_const_paired_dpo", "ocean_const_paired_dpo_s1vs2",
              "Llama (GLM-4.5-Air teacher)", id="llama-glm",
              color="#1f77b4", marker="o"),
    ModelSpec("llama-3.1-8b-it", "llama", 8,
              "ocean_const_paired_dpo_teacher_dsv32",
              "ocean_const_paired_dpo_teacher_dsv32_s1vs2",
              "Llama (DeepSeek-V3.2 teacher)", id="llama-dsv32",
              color="#d62728", marker="s"),
]

MODEL_SETS: dict[str, list[ModelSpec]] = {
    "cross_model": _CROSS_MODEL,
    "llama_teacher": _LLAMA_TEACHER,
}

# Active set (reassigned in main from --set). Default = the 6-model comparison.
MODELS: list[ModelSpec] = _CROSS_MODEL
COMPARISON_NOUN = "base models"  # used in suptitles; "teachers" for the ablation

# (lowercase trait, capitalised trait as it appears in eval metadata, short)
TRAITS: list[tuple[str, str, str]] = [
    ("openness", "Openness", "O"),
    ("conscientiousness", "Conscientiousness", "C"),
    ("extraversion", "Extraversion", "E"),
    ("agreeableness", "Agreeableness", "A"),
    ("neuroticism", "Neuroticism", "N"),
]
OCEAN_TRAITS_CAP = [cap for _, cap, _ in TRAITS]
DIRECTIONS = ("amplifier", "suppressor")

_FAMILY_CMAP = {"llama": "Blues", "qwen": "Oranges", "gemma": "Greens"}
# A distinct marker per model, grouped by family (ascending size within family).
_FAMILY_MARKERS = {"llama": ["o"], "qwen": ["^", "v"], "gemma": ["s", "D", "P"]}
_MARKER_SIZE = 4.5  # constant across models — size is encoded by colour shade only


@dataclass(frozen=True)
class _Style:
    color: tuple
    marker: str
    markersize: float


def _build_styles() -> dict[str, _Style]:
    """Map each model slug to a colour + a unique marker.

    Family sets the colormap; within a family the sequential colormap position
    rises with model size (bigger = darker). Every model gets its own marker
    shape (grouped by family) so the six lines stay distinguishable. Marker size
    is constant — scaling it with model size made the grids too cluttered.
    """
    styles: dict[str, _Style] = {}
    by_family: dict[str, list[ModelSpec]] = defaultdict(list)
    for m in MODELS:
        by_family[m.family].append(m)
    for family, members in by_family.items():
        members = sorted(members, key=lambda m: m.params_b)
        cmap = plt.get_cmap(_FAMILY_CMAP[family])
        markers = _FAMILY_MARKERS[family]
        n = len(members)
        for i, m in enumerate(members):
            if m.color is not None and m.marker is not None:
                styles[m.id] = _Style(m.color, m.marker, _MARKER_SIZE)
                continue
            # Spread shade positions in [0.45, 0.92]; single-member families sit
            # at a mid-dark 0.72.
            pos = 0.72 if n == 1 else 0.45 + 0.47 * (i / (n - 1))
            styles[m.id] = _Style(
                color=cmap(pos), marker=markers[i], markersize=_MARKER_SIZE
            )
    return styles


STYLES = _build_styles()


def _legend_handles() -> list[Line2D]:
    """One legend entry per model, reusing its line colour/marker/size."""
    handles = []
    for m in MODELS:
        st = STYLES[m.id]
        handles.append(
            Line2D([0], [0], color=st.color, marker=st.marker, markersize=st.markersize,
                   linewidth=2.0, label=m.label)
        )
    return handles


# ── HF enumeration + parsing ─────────────────────────────────────────────────


def _adapter_run_dir(model: ModelSpec, trait: str, direction: str, kind: str) -> str:
    return (
        f"fine_tuning/{model.slug}/ocean/{trait}/{direction}/{model.version}"
        f"/evals/mcq/{kind}"
    )


def _control_run_dir(model: ModelSpec, kind: str) -> str:
    return (
        f"fine_tuning/{model.slug}/other/ocean_def_control/amplifier/"
        f"{model.control_version}/evals/mcq/{kind}"
    )


def _glob_scale_logs(fs: HfFileSystem, run_dir: str, kind: str) -> dict[float, str]:
    """Return ``{lora_scale: hf_relative_path}`` for one eval run directory.

    The on-repo layout is ``{run_dir}/{named_run}/{scale_dir}/{kind}/native/
    inspect_logs/*.json``; the scale dir (``base`` / ``lora_<±XpYY>x``) sits two
    levels under ``{kind}/native``.
    """
    patt = f"datasets/{HF_REPO_ID}/{run_dir}/*/*/{kind}/native/inspect_logs/*.json"
    out: dict[float, str] = {}
    for full in fs.glob(patt):
        rel = full.split(f"datasets/{HF_REPO_ID}/", 1)[1]
        scale_dir = rel.split(f"/{kind}/native/")[0].rsplit("/", 1)[1]
        scale = parse_lora_name(scale_dir)
        if scale is not None:
            out[scale] = rel
    return out


_session = requests.Session()


def _fetch_trait_arrays(rel_path: str) -> dict[str, np.ndarray] | None:
    """Download a trait log and return per-measured-trait raw score arrays.

    Only the per-sample scores are kept (the bootstrap CI is computed later, off
    the download threads, to avoid GIL contention starving the I/O workers).
    """
    def parse(p: Path):
        per = per_trait_scores_from_log(
            p, OCEAN_TRAITS_CAP, min_choice_mass=MIN_CHOICE_MASS
        )
        if not per:
            return None
        arrs = {cap: per[cap] for cap in OCEAN_TRAITS_CAP
                if cap in per and per[cap].size}
        return arrs or None

    return stream_to_tempfile(
        f"{RESOLVE_BASE}/{rel_path}", CACHE_DIR, parse, session=_session,
        suffix=".json", timeout=300, chunk_size=1 << 20,
        quiet_on_non_200=False, label=rel_path,
    )


def _fetch_mmlu_pn(rel_path: str) -> tuple[float, int] | None:
    """Range-fetch an MMLU log's header → ``(accuracy, n)``.

    MMLU's Wilson CI needs only the aggregate accuracy ``p`` and sample count
    ``n``, both in the small ``results`` block at the top of the log, so this
    reads just the first chunk instead of the whole 50+ MB file. ``(p, n)`` are
    cached and the Wilson interval recomputed at plot time.
    """
    return accuracy_from_log_url(rel_path, resolve_base=RESOLVE_BASE, session=_session)


def _gather(jobs: list[tuple], fetch_fn, max_workers: int = 16) -> dict:
    """Fetch a list of ``(key, scale, rel_path)`` jobs in parallel.

    ``fetch_fn(rel_path)`` does the per-file download + parse. Returns
    ``{key: {scale: parsed}}`` (only successfully-parsed entries).
    """
    out: dict = defaultdict(dict)
    print(f"  fetching {len(jobs)} inspect logs …", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_meta = {
            ex.submit(fetch_fn, rel): (key, scale) for key, scale, rel in jobs
        }
        done = 0
        for fut in as_completed(fut_to_meta):
            key, scale = fut_to_meta[fut]
            parsed = fut.result()
            if parsed is not None:
                out[key][scale] = parsed
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(jobs)}", flush=True)
    return out


# ── Per-model score cache ────────────────────────────────────────────────────
#
# The raw inspect logs are huge (55-116 MB each, ~56 GB total) and can't be kept
# on disk, but the *extracted* scores are tiny. So per model we download its logs
# one at a time (stream → parse → delete), keep only the per-sample trait arrays
# + MMLU (accuracy, n), and persist those to a small JSON in scratch. Re-runs
# (and plot tweaks) read the cache and never touch HF again. Delete a model's
# cache file (or pass --refresh) to force a re-download.


def _scores_key(key: tuple) -> str:
    """Flatten a job key to a cache string (model slug is implicit per file)."""
    if key[0] == "control":
        return "control|control"
    direction, _slug, applied = key
    return f"{direction}|{applied}"


def _model_cache_path(slug: str) -> Path:
    return _CACHE / f"{slug}.json"


def _build_model_cache(fs, model: ModelSpec, scale_filter) -> dict:
    """Download one model's logs, extract scores, persist, delete logs."""
    print(f"  [{model.id}] enumerating + fetching trait logs "
          "(stream → parse → delete) …", flush=True)
    traw = _gather(_enumerate_trait_jobs(fs, scale_filter, [model]),
                   _fetch_trait_arrays)
    print(f"  [{model.id}] range-fetching MMLU headers …", flush=True)
    mraw = _gather(_enumerate_mmlu_jobs(fs, scale_filter, [model]), _fetch_mmlu_pn)

    trait_ser: dict = {}
    for key, by_scale in traw.items():
        d = trait_ser.setdefault(_scores_key(key), {})
        for scale, per_trait in by_scale.items():
            d[f"{scale:g}"] = {cap: [float(x) for x in arr]
                               for cap, arr in per_trait.items()}
    mmlu_ser: dict = {}
    for key, by_scale in mraw.items():
        d = mmlu_ser.setdefault(_scores_key(key), {})
        for scale, (acc, n) in by_scale.items():
            d[f"{scale:g}"] = [acc, n]

    payload = {"model": model.id, "trait": trait_ser, "mmlu": mmlu_ser}
    path = _model_cache_path(model.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    print(f"  [{model.id}] cached scores → {path}", flush=True)
    return payload


def _merge_cache(payload: dict, slug: str, trait_data: dict, mmlu_data: dict) -> None:
    """Fold one model's cached scores into the global render structures.

    Trait per-sample arrays are bootstrapped to ``(mean, lo, hi)``; MMLU
    ``(accuracy, n)`` is turned into a Wilson interval. Trait points whose
    retained sample count was gutted by the choice-mass filter (see
    ``MIN_RETAINED_*``) are dropped so a few surviving samples don't produce a
    spurious huge error bar.
    """
    def _key(sk: str):
        direction, applied = sk.split("|")
        return ("control", slug) if direction == "control" else (direction, slug, applied)

    for sk, by_scale in payload.get("trait", {}).items():
        key = _key(sk)
        # First pass: collect per-(scale, measured-trait) arrays and the typical
        # (max-over-scales) retained count for each measured trait.
        arrays: dict[tuple[float, str], np.ndarray] = {}
        n_typical: dict[str, int] = defaultdict(int)
        for scale_str, per_trait in by_scale.items():
            scale = float(scale_str)
            for cap, arr in per_trait.items():
                a = np.asarray(arr, dtype=float)
                arrays[(scale, cap)] = a
                n_typical[cap] = max(n_typical[cap], a.size)
        # Second pass: bootstrap only the points that survived the n-filter.
        for (scale, cap), a in arrays.items():
            floor = max(MIN_RETAINED_ABS, MIN_RETAINED_FRAC * n_typical[cap])
            if a.size < floor:
                continue
            trait_data[key].setdefault(scale, {})[cap] = bootstrap_ci(
                a, confidence=CI_CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=SEED)

    for sk, by_scale in payload.get("mmlu", {}).items():
        key = _key(sk)
        for scale_str, (acc, n) in by_scale.items():
            lo, hi = wilson_ci_from_p_n(acc, int(n), CI_CONFIDENCE)
            mmlu_data[key][float(scale_str)] = (acc, lo, hi)


def _enumerate_trait_jobs(fs, scale_filter, models):
    """Build trait-log jobs. Keys: ``(direction, slug, applied)`` and ``("control", slug)``."""
    specs = []  # (key, run_dir)
    for m in models:
        for direction in DIRECTIONS:
            for trait, _, _ in TRAITS:
                specs.append((
                    (direction, m.id, trait),
                    _adapter_run_dir(m, trait, direction, "trait_logprobs"),
                ))
        specs.append((("control", m.id), _control_run_dir(m, "trait_logprobs")))

    jobs = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut = {ex.submit(_glob_scale_logs, fs, rd, "trait_logprobs"): key
               for key, rd in specs}
        for f in as_completed(fut):
            key = fut[f]
            for scale, rel in f.result().items():
                if scale_filter(scale):
                    jobs.append((key, scale, rel))
    return jobs


def _enumerate_mmlu_jobs(fs, scale_filter, models):
    specs = []
    for m in models:
        for direction in DIRECTIONS:
            for trait, _, _ in TRAITS:
                specs.append((
                    (direction, m.id, trait),
                    _adapter_run_dir(m, trait, direction, "mmlu"),
                ))
        specs.append((("control", m.id), _control_run_dir(m, "mmlu")))

    jobs = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut = {ex.submit(_glob_scale_logs, fs, rd, "mmlu"): key for key, rd in specs}
        for f in as_completed(fut):
            key = fut[f]
            for scale, rel in f.result().items():
                if scale_filter(scale):
                    jobs.append((key, scale, rel))
    return jobs


# ── Rendering ────────────────────────────────────────────────────────────────


def _plot_model_line(ax, scales, triples, style: _Style, *, label=None):
    """Plot one model's (mean, lo, hi)-per-scale series with error bars."""
    xs = sorted(scales)
    means = np.array([triples[s][0] for s in xs])
    los = np.array([triples[s][1] for s in xs])
    his = np.array([triples[s][2] for s in xs])
    yerr = np.clip(np.stack([means - los, his - means]), 0.0, None)
    ax.errorbar(
        xs, means, yerr=yerr, color=style.color, ecolor=style.color,
        marker=style.marker, markersize=style.markersize, linewidth=1.8,
        elinewidth=0.9, capsize=2, label=label,
    )


def _style_axis(ax, *, ylim, bold=False):
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.grid(True, alpha=0.25)
    if bold:
        for sp in ax.spines.values():
            sp.set_linewidth(5.2)
            sp.set_edgecolor("black")


def _figure_legend(fig, ncol=None):
    handles = _legend_handles()
    fig.legend(
        handles=handles, loc="lower center", ncol=ncol or len(MODELS),
        fontsize=11, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )


def render_trait_matrix(direction: str, data: dict, out_stem: str) -> None:
    """5x5 grid: applied adapter (cols) x measured trait (rows), line per model."""
    n = len(TRAITS)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 3.4 * n), sharex=True, sharey=True)
    for ri, (_, measured_cap, _) in enumerate(TRAITS):
        for ci, (applied, _, _) in enumerate(TRAITS):
            ax = axes[ri][ci]
            for m in MODELS:
                series = data.get((direction, m.id, applied), {})
                triples = {s: v[measured_cap] for s, v in series.items()
                           if measured_cap in v}
                if triples:
                    _plot_model_line(ax, triples.keys(), triples, STYLES[m.id])
            _style_axis(ax, ylim=(0.0, 1.0), bold=(ri == ci))
            if ri == 0:
                ax.set_title(f"{applied.capitalize()} adapter", fontsize=13)
            if ci == 0:
                ax.set_ylabel(f"{measured_cap}\nscore", fontsize=12)
            if ri == n - 1:
                ax.set_xlabel("LoRA scale", fontsize=12)
    arrow = "↑" if direction == "amplifier" else "↓"
    fig.suptitle(
        f"OCEAN {direction} adapters {arrow} — applied adapter (columns) vs "
        "measured trait (rows); bold = on-target diagonal",
        fontsize=16, y=0.998,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    _figure_legend(fig)
    _save(fig, out_stem)


def render_trait_control(data: dict, out_stem: str) -> None:
    """1x5 row: the null control adapter measured across the 5 OCEAN traits."""
    n = len(TRAITS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharex=True, sharey=True)
    for ci, (_, measured_cap, _) in enumerate(TRAITS):
        ax = axes[ci]
        for m in MODELS:
            series = data.get(("control", m.id), {})
            triples = {s: v[measured_cap] for s, v in series.items()
                       if measured_cap in v}
            if triples:
                _plot_model_line(ax, triples.keys(), triples, STYLES[m.id])
        _style_axis(ax, ylim=(0.0, 1.0))
        ax.set_title(measured_cap, fontsize=13)
        ax.set_xlabel("LoRA scale", fontsize=12)
        if ci == 0:
            ax.set_ylabel("Trait score", fontsize=12)
    fig.suptitle(
        "Null control adapter — trait scores across all OCEAN traits "
        "(flat ≈ no spurious trait shift)",
        fontsize=15, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _figure_legend(fig)
    _save(fig, out_stem)


def render_mmlu_sweeps(data: dict, out_stem: str) -> None:
    """2x5 grid: amplifier/suppressor (rows) x applied adapter (cols)."""
    n = len(TRAITS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7.2), sharex=True, sharey=True)
    for ri, direction in enumerate(DIRECTIONS):
        arrow = "↑" if direction == "amplifier" else "↓"
        for ci, (applied, _, _) in enumerate(TRAITS):
            ax = axes[ri][ci]
            for m in MODELS:
                triples = data.get((direction, m.id, applied), {})
                if triples:
                    _plot_model_line(ax, triples.keys(), triples, STYLES[m.id])
            _style_axis(ax, ylim=(0.0, 0.85))
            if ri == 0:
                ax.set_title(f"{applied.capitalize()} adapter", fontsize=13)
            if ci == 0:
                ax.set_ylabel(f"{direction.capitalize()} {arrow}\nMMLU acc",
                              fontsize=12)
            if ri == 1:
                ax.set_xlabel("LoRA scale", fontsize=12)
    fig.suptitle(
        f"MMLU capability retention across {COMPARISON_NOUN} — applied OCEAN "
        "adapter (columns) x direction (rows)",
        fontsize=16, y=0.999,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _figure_legend(fig)
    _save(fig, out_stem)


def render_mmlu_control(data: dict, out_stem: str) -> None:
    """Single panel: null control adapter MMLU accuracy vs LoRA scale."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in MODELS:
        triples = data.get(("control", m.id), {})
        if triples:
            _plot_model_line(ax, triples.keys(), triples, STYLES[m.id])
    _style_axis(ax, ylim=(0.0, 0.85))
    ax.set_xlabel("LoRA scale", fontsize=12)
    ax.set_ylabel("MMLU accuracy", fontsize=12)
    ax.set_title("Null control adapter — MMLU retention", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    _figure_legend(fig, ncol=3)
    _save(fig, out_stem)


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")


# ── Report ───────────────────────────────────────────────────────────────────


def write_report(trait_data: dict, mmlu_data: dict) -> None:
    def shift(series, measured_cap, applied_trait, direction):
        base = series.get(0.0, {}).get(measured_cap)
        peak = None
        for s, v in series.items():
            if s == 0.0 or measured_cap not in v:
                continue
            if direction == "amplifier" and s > 0:
                peak = v[measured_cap][0] if peak is None else max(peak, v[measured_cap][0])
            elif direction == "suppressor" and s > 0:
                peak = v[measured_cap][0] if peak is None else min(peak, v[measured_cap][0])
        if base is None or peak is None:
            return None
        b = base[0]
        return (peak - b) if direction == "amplifier" else (b - peak)

    L = ["# Cross-model OCEAN persona-LoRA transfer\n"]
    L.append(
        f"Models: {', '.join(m.label for m in MODELS)}. Trait = target-trait "
        "logprob-MCQ score (0–1); MMLU = 0-shot accuracy. Data scraped from the "
        "HF monorepo eval trees.\n"
    )
    L.append("## On-target trait shift (best intended-direction scale > 0)\n")
    L.append("| applied · dir | " + " | ".join(m.label for m in MODELS) + " |")
    L.append("|---" * (1 + len(MODELS)) + "|")
    for direction in DIRECTIONS:
        arrow = "↑" if direction == "amplifier" else "↓"
        for trait, cap, short in TRAITS:
            row = [f"{short}{arrow}"]
            for m in MODELS:
                series = trait_data.get((direction, m.id, trait), {})
                sh = shift(series, cap, trait, direction)
                row.append("—" if sh is None else f"{sh:+.3f}")
            L.append("| " + " | ".join(row) + " |")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L))
    print("  saved REPORT.md")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="model_set", default="cross_model",
                    choices=list(MODEL_SETS),
                    help="Which model set to plot (default: the 6-model comparison).")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast plumbing check: 2 models, few scales, light bootstrap.")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore cached per-model scores and re-download.")
    args = ap.parse_args()

    global BOOTSTRAP_RESAMPLES, _CACHE, OUT, MODELS, STYLES, COMPARISON_NOUN
    MODELS = MODEL_SETS[args.model_set]
    if args.model_set != "cross_model":
        OUT = project_root / "scratch" / f"model_comparison_{args.model_set}"
        _CACHE = OUT / "_cache"
    if args.model_set == "llama_teacher":
        COMPARISON_NOUN = "distillation teachers"
    STYLES = _build_styles()

    models = MODELS
    scale_filter = lambda s: True  # noqa: E731
    if args.smoke:
        models = MODELS[:2]
        smoke_scales = {0.0, 1.0, -1.0, 2.0, -2.0}
        scale_filter = lambda s: s in smoke_scales  # noqa: E731
        BOOTSTRAP_RESAMPLES = 200
        _CACHE = OUT / "_cache_smoke"

    OUT.mkdir(parents=True, exist_ok=True)
    _CACHE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    trait_data: dict = defaultdict(dict)
    mmlu_data: dict = defaultdict(dict)
    try:
        for m in models:
            cp = _model_cache_path(m.id)
            if cp.exists() and not args.refresh:
                print(f"[{m.id}] cached scores ({cp.name}) — skipping download")
                payload = json.loads(cp.read_text())
            else:
                print(f"[{m.id}] downloading logs → extracting scores → caching")
                payload = _build_model_cache(fs, m, scale_filter)
            _merge_cache(payload, m.id, trait_data, mmlu_data)

        render_trait_matrix("amplifier", trait_data, "fig_trait_matrix_amplifier")
        render_trait_matrix("suppressor", trait_data, "fig_trait_matrix_suppressor")
        render_trait_control(trait_data, "fig_trait_control")
        render_mmlu_sweeps(mmlu_data, "fig_mmlu_sweeps")
        render_mmlu_control(mmlu_data, "fig_mmlu_control")
        write_report(trait_data, mmlu_data)
        print(f"\nAll outputs in {OUT}  (per-model score cache in {_CACHE})")
    finally:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)


if __name__ == "__main__":
    main()

"""Shared scaffolding for the OCEAN-appendix paired-DPO sweep figures.

The three appendix scripts
(``scripts/visualisations/appendix_paired_dpo_{judge,mmlu,trait}.py``) each stream a
large number of per-persona / per-scale result files straight off HF into a
throwaway tempfile, parse them, and plot one figure per persona. Their plot
layouts are genuinely different (judge twin-axis line plot, MMLU stacked
breakdown bars, TRAIT logprob lines + choice-mass strip), so rendering stays in
the individual scripts. The pieces that are byte-for-byte identical across them
— the persona list, the filename-stem / title formatting, the lora-scale dir
parser, the bootstrap CI, and the stream-download-to-tempfile skeleton — live
here so all three share one copy.

Behaviour is identical to the pre-refactor per-script copies.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import requests
from scipy import stats

from src.evals.personality.ci import _interval_ci_from_bootstrap

# OCEAN persona list shared by every appendix sweep figure: each trait in both
# directions, plus the control adapter.
OCEAN_TRAITS_LOWER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

PERSONAS: list[tuple[str, str]] = [
    *[
        (trait, direction)
        for trait in OCEAN_TRAITS_LOWER
        for direction in ("amplifier", "suppressor")
    ],
    ("control", "control"),
]


def persona_filename_stem(trait: str, direction: str) -> str:
    """Output-filename stem for a persona (e.g. ``openness_plus_paired_dpo``)."""
    if trait == "control":
        return "control_paired_dpo"
    sign = "plus" if direction == "amplifier" else "minus"
    return f"{trait}_{sign}_paired_dpo"


def persona_title(trait: str, direction: str, prefix: str) -> str:
    """Figure title for a persona, e.g. ``"<prefix>: Conscientiousness ↑"``.

    Args:
        prefix: Plot-family prefix (e.g. ``"LLM Judge"``, ``"MMLU"``, ``"TRAIT"``).
    """
    if trait == "control":
        return f"{prefix}: Control"
    sign = "↑" if direction == "amplifier" else "↓"
    return f"{prefix}: {trait.capitalize()} {sign}"


def per_trait_scores_from_log(
    log_path: Path,
    ocean_traits: list[str],
    *,
    min_choice_mass: float = 0.75,
) -> dict[str, np.ndarray] | None:
    """Parse a TRAIT-logprobs inspect log into per-OCEAN-trait score arrays.

    A single trait-logprobs eval scores every OCEAN trait, so one log yields a
    score array for each trait in ``ocean_traits`` (keyed by the sample's
    ``metadata.trait``). Samples whose answer choice-mass falls below
    ``min_choice_mass`` are dropped (a degenerate-generation guard); when the
    scorer doesn't report ``choice_mass`` directly it is reconstructed from the
    logprobs. The unfiltered choice masses are returned under
    ``"_choice_mass_all"`` for an optional diagnostic strip.

    Args:
        log_path: Path to a downloaded inspect-log JSON file.
        ocean_traits: Trait names to collect, matched against ``metadata.trait``
            (e.g. the capitalised ``["Openness", ...]`` used by the eval).
        min_choice_mass: Minimum answer choice-mass for a sample to be kept.

    Returns:
        ``{trait: scores}`` (plus ``"_choice_mass_all"``) or ``None`` if the file
        cannot be parsed or yields no usable scores.
    """
    try:
        with log_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    samples = doc.get("samples") or []
    by_trait: dict[str, list[float]] = {t: [] for t in ocean_traits}
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
            if cm < min_choice_mass:
                continue
        by_trait[trait].append(float(v))
    out: dict[str, np.ndarray] = {
        t: np.asarray(v, dtype=float) for t, v in by_trait.items() if v
    }
    if all_choice_mass:
        out["_choice_mass_all"] = np.asarray(all_choice_mass, dtype=float)
    return out or None


def parse_lora_name(name: str) -> float | None:
    """Parse a ``base`` / ``lora_<±XpYY>x`` inspect-log dir name to a scale."""
    if name == "base":
        return 0.0
    if not name.startswith("lora_") or not name.endswith("x"):
        return None
    body = name[len("lora_") : -1].replace("p", ".")
    try:
        return float(body)
    except ValueError:
        return None


def bootstrap_ci(
    values: np.ndarray,
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` via BCa bootstrap (nan triple if empty)."""
    if values.size == 0:
        return (float("nan"),) * 3
    m = float(values.mean())
    lo, hi = _interval_ci_from_bootstrap(values, confidence, resamples, seed)
    return m, lo, hi


T = TypeVar("T")


def stream_to_tempfile(
    url: str,
    cache_dir: Path,
    parse: Callable[[Path], T],
    *,
    session: requests.Session,
    suffix: str,
    timeout: int,
    chunk_size: int,
    quiet_on_non_200: bool,
    label: str | None = None,
) -> T | None:
    """Stream ``url`` into a tempfile under ``cache_dir``, parse it, then unlink.

    Returns ``parse(tmp_path)`` on a 200/206 response, else ``None``. On a
    non-200 status, prints ``✗ <label> HTTP <code>`` unless ``quiet_on_non_200``
    is set (the judge sweep expects many 404s for cross-trait baseline-only
    fingerprints and stays quiet).

    Args:
        label: Identifier used in diagnostic prints (the callers pass the
            HF-relative path so messages match the pre-refactor output).
            Defaults to ``url``.
    """
    tag = label if label is not None else url
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=cache_dir)
    except OSError as exc:
        print(f"  ✗ {tag}: tempfile failed: {exc}")
        return None
    tmp_path = Path(tmp_name)
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code not in (200, 206):
                if not quiet_on_non_200:
                    print(f"  ✗ {tag}: HTTP {r.status_code}")
                return None
            with open(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return parse(tmp_path)
    except Exception as exc:
        print(f"  ✗ {tag}: {type(exc).__name__}: {str(exc)[:100]}")
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def accuracy_from_log_url(
    rel_path: str,
    *,
    resolve_base: str,
    session: requests.Session,
    range_bytes: int = 200_000,
    retries: int = 3,
) -> tuple[float, int] | None:
    """HTTP-Range fetch an inspect log's head and extract ``(accuracy, n)``.

    The ``results`` block (with the aggregate ``accuracy`` metric and
    ``scored_samples`` count) sits near the top of an inspect log, before the
    bulky ``samples`` array. For accuracy-style scorers whose CI is closed-form
    (Wilson needs only ``p`` and ``n``), reading the first ``range_bytes`` avoids
    downloading the whole log. The window is doubled up to ``retries`` times if
    the results block hasn't fully arrived yet.

    Returns ``(accuracy, n)`` or ``None`` on any failure / missing fields.
    """
    url = f"{resolve_base}/{rel_path}"
    range_size = range_bytes
    results = None
    for _ in range(retries):
        try:
            r = session.get(
                url, headers={"Range": f"bytes=0-{range_size}"},
                allow_redirects=True, timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {rel_path}: {type(exc).__name__}: {str(exc)[:80]}")
            return None
        if r.status_code not in (200, 206):
            print(f"  ✗ {rel_path}: HTTP {r.status_code}")
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


def parse_cap_name(name: str) -> float | None:
    """Parse an activation-capping dir name (``base`` or ``cap_<±XpYY>``) to a float."""
    if name == "base":
        return 0.0
    if not name.startswith("cap_"):
        return None
    body = name[len("cap_"):].replace("p", ".")
    try:
        return float(body)
    except ValueError:
        return None


def wilson_ci_from_p_n(p: float, n: int, confidence: float) -> tuple[float, float]:
    """Closed-form Wilson CI given a fraction ``p`` and sample size ``n``."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1 - (1 - confidence / 100) / 2))
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - margin, centre + margin)


def extract_results_block(text: str) -> dict | None:
    """Brace-count the ``"results": {...}`` value out of partial JSON bytes."""
    idx = text.find('"results"')
    if idx == -1:
        return None
    open_idx = text.find("{", idx)
    if open_idx == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(open_idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[open_idx : i + 1])
                except json.JSONDecodeError:
                    return None
    return None

"""Shared scaffolding for the OCEAN-appendix paired-DPO sweep figures.

The three appendix scripts
(``scripts/figures/appendix_paired_dpo_{judge,mmlu,trait}.py``) each stream a
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

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import requests

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

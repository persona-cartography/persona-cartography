"""Data-loading and parsing for personality evaluation sweep results.

Verbatim migration of the data-loading / parsing portion of
``src_dev/evals/personality/analyze_results.py``. Only the symbols that read
and normalize sweep run directories and inspect logs are copied here,
byte-for-byte identical to the dev source. Plotting / CLI code from the dev
module is intentionally NOT copied (handled in a later chunk). The dev original
is left untouched.

Extracted symbols: ``SweepData``; ``_extract_scores_from_log``,
``_extract_scores``; ``_extract_choice_mass``;
``_extract_raw_sample_scores_from_log``, ``_extract_raw_sample_scores``;
``_extract_scores_reparsed``; ``_load_from_info``; ``_parse_scale``,
``_normalise_scale_col``; ``_parse_mcq_answer``; ``_metric_cols``;
``load_sweep_data``, ``load_data_from_logs``; and the trait / eval
categorization constants (``BIG_FIVE``, ``DARK_TRIAD``, ``ALL_TRAIT_COLS``,
``_OCEAN_ALIASES``, ``_PERSONALITY_EVALS``, ``_LOGPROB_CAPABILITY_EVALS``,
``_NON_MODEL_DIRS``).

Import adjustments vs. the dev source:

- ``rescore_log`` is imported from ``src.evals.personality.log_answer_parser``
  (the dev source used a dynamic ``from src_dev...log_answer_parser import
  rescore_log`` inside the functions that need it; that import is preserved in
  the same place, repointed at ``src.``).
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_mcq_answer(text: str) -> str | None:
    """Fallback MCQ answer parser: extract a letter A-D from common formats."""
    if not text or not text.strip():
        return None
    s = text.strip()
    # ANSWER: X
    m = re.search(r"ANSWER\s*:\s*([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # X) at start
    m = re.match(r"^([A-D])\)", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Bare X followed by whitespace/newline/end
    m = re.match(r"^([A-D])\s*$", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"^([A-D])\s*[\n\r)]", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "the answer is X"
    m = re.search(r"(?:correct\s+)?answer\s+is\s*:?\s*([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
DARK_TRIAD = ["Machiavellianism", "Narcissism", "Psychopathy"]
ALL_TRAIT_COLS = BIG_FIVE + DARK_TRIAD

_OCEAN_ALIASES: dict[str, str] = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

# Eval names that use the fallback answer parser (rescore_log) for scoring.
# trait_logprobs uses logprob-based continuous scores (not text parsing) but
# still reports per-trait personality metrics in the same 0-1 format.
_PERSONALITY_EVALS = {"bfi", "trait", "trait_logprobs"}

# Logprob-based capability evals (P(correct) from logprobs, not text C/I).
_LOGPROB_CAPABILITY_EVALS = {"mmlu_logprobs", "truthfulqa_logprobs", "gpqa_logprobs"}

# Directories that are not model-spec dirs at the run root.
_NON_MODEL_DIRS = {"figures", "analysis"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SweepData:
    """DataFrames for every eval found in a sweep run directory.

    Each DataFrame has columns: scale (float), run (str), + metric columns.
    Keyed by eval name (the directory name used in the suite config, e.g.
    "bfi", "trait", "mmlu", or any custom name).

    Access via ``data.get("bfi")`` which returns None if the eval is absent.
    """
    evals: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame | None:
        return self.evals.get(name)

    def names(self) -> list[str]:
        return list(self.evals.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_scores_from_log(log: dict) -> tuple[dict[str, float], float] | None:
    """Like :func:`_extract_scores` but operates on an already-loaded log dict.

    Callers that also need raw per-sample scores should prefer this path to
    avoid loading the (often >100 MB) log JSON twice.
    """
    if log.get("status") != "success":
        return None
    score_entry = log["results"]["scores"][0]
    scored   = score_entry.get("scored_samples", 0)
    unscored = score_entry.get("unscored_samples", 0)
    total = scored + unscored
    parse_rate = scored / total if total > 0 else 1.0
    metrics = score_entry["metrics"]
    scores = {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}
    return scores, parse_rate


def _extract_scores(log_path: Path) -> tuple[dict[str, float], float] | None:
    """Extract metric values and parse rate from an inspect log JSON.

    Returns:
        Tuple of (scores dict, parse_rate 0–1), or None if the log failed.
    """
    with open(log_path) as f:
        log = json.load(f)
    return _extract_scores_from_log(log)


def _extract_choice_mass(score_data: dict) -> float | None:
    """Extract choice_mass from score metadata, with backward-compat fallback."""
    score_meta = score_data.get("metadata") or {}
    cm = score_meta.get("choice_mass")
    if cm is None:
        lps = score_meta.get("logprobs")
        if isinstance(lps, dict) and lps:
            cm = sum(math.exp(v) for v in lps.values())
    return cm


def _extract_raw_sample_scores_from_log(log: dict, eval_type: str) -> dict[str, list[float]] | None:
    """Extract per-sample scores from an inspect log.

    Handles four scoring conventions:

    - **Text-based personality evals** (trait, bfi): samples have
      ``metadata.trait`` and ``metadata.answer_mapping``.  For each sample the
      inspect scorer parsed (``value == "C"``), the chosen answer is mapped
      through ``answer_mapping`` to get a trait score (0.0 or 1.0).
    - **Logprob personality evals** (trait_logprobs): ``value`` is a continuous
      float 0-1 (the probability-weighted trait score).  Grouped by trait.
    - **Logprob capability evals** (mmlu_logprobs, etc.): ``value`` is a
      continuous float P(correct).  Grouped under ``"accuracy"``.
    - **Text capability evals** (mmlu, etc.): ``C`` = correct (1.0),
      ``I`` = incorrect (0.0).  Grouped under ``"accuracy"``.

    Args:
        log: Pre-loaded inspect log JSON (as returned by ``json.load``).
        eval_type: Eval type name, used to distinguish personality vs.
            capability scoring and as the group key for capability evals.

    Returns:
        Dict mapping group name to list of per-sample scores, or
        None if the log has no usable per-sample data.
    """
    if log.get("status") != "success":
        return None
    samples = log.get("samples")
    if not samples:
        return None

    is_personality = eval_type in _PERSONALITY_EVALS
    is_logprob = eval_type == "trait_logprobs"
    is_logprob_capability = eval_type in _LOGPROB_CAPABILITY_EVALS
    group_scores: dict[str, list[float]] = {}

    for sample in samples:
        meta = sample.get("metadata") or {}
        for ev in sample.get("events", []):
            if ev.get("event") != "score":
                continue
            score_data = ev.get("score", {})
            value = score_data.get("value")

            if is_logprob:
                # Logprob scorer: value is a continuous 0-1 float.
                trait = meta.get("trait")
                if not trait or not isinstance(value, (int, float)):
                    break
                val = float(value)
                score_meta = score_data.get("metadata") or {}
                cm = _extract_choice_mass(score_data)
                nc = score_meta.get("num_choices", 4)
                if not math.isnan(val):
                    group_scores.setdefault(trait, []).append(val)
                    cm_val = float(cm) if isinstance(cm, (int, float)) else 1.0
                    group_scores.setdefault(f"_cm_{trait}", []).append(cm_val)
                    group_scores.setdefault(f"_nc_{trait}", []).append(float(nc))
                if isinstance(cm, (int, float)):
                    group_scores.setdefault("_choice_mass", []).append(float(cm))

            elif is_logprob_capability:
                # Logprob capability: value is P(correct), a continuous 0-1 float.
                if not isinstance(value, (int, float)):
                    break
                val = float(value)
                score_meta = score_data.get("metadata") or {}
                cm = _extract_choice_mass(score_data)
                nc = score_meta.get("num_choices", 4)
                if not math.isnan(val):
                    group_scores.setdefault("accuracy", []).append(val)
                    cm_val = float(cm) if isinstance(cm, (int, float)) else 1.0
                    group_scores.setdefault("_cm_accuracy", []).append(cm_val)
                    group_scores.setdefault("_nc_accuracy", []).append(float(nc))
                if isinstance(cm, (int, float)):
                    group_scores.setdefault("_choice_mass", []).append(float(cm))

            elif is_personality:
                # Trait/BFI: C means "parsed an answer", use answer_mapping
                # for the actual trait score.
                trait = meta.get("trait")
                answer_mapping = meta.get("answer_mapping")
                if not trait or not answer_mapping or value != "C":
                    break
                answer = score_data.get("answer")
                if answer and answer in answer_mapping:
                    group_scores.setdefault(trait, []).append(
                        float(answer_mapping[answer])
                    )
            else:
                # Capability: C = correct (1.0), I = incorrect (0.0)
                answer = score_data.get("answer")
                target = sample.get("target")
                if value == "C":
                    group_scores.setdefault("accuracy", []).append(1.0)
                    group_scores.setdefault("_answer_parsed", []).append(1.0)
                    group_scores.setdefault("_reparsed_accuracy", []).append(1.0)
                elif value == "I":
                    group_scores.setdefault("accuracy", []).append(0.0)
                    group_scores.setdefault("_answer_parsed", []).append(1.0 if answer else 0.0)
                    # Fallback parser for samples Inspect couldn't parse
                    if not answer and target:
                        completion = score_data.get("explanation", "")
                        recovered = _parse_mcq_answer(completion)
                        group_scores.setdefault("_reparsed_accuracy", []).append(
                            1.0 if recovered and recovered == target else 0.0
                        )
                    else:
                        group_scores.setdefault("_reparsed_accuracy", []).append(0.0)
            break

    return group_scores if group_scores else None


def _extract_raw_sample_scores(log_path: Path, eval_type: str) -> dict[str, list[float]] | None:
    """Path-based wrapper around :func:`_extract_raw_sample_scores_from_log`."""
    with open(log_path) as f:
        log = json.load(f)
    return _extract_raw_sample_scores_from_log(log, eval_type)


def _extract_scores_reparsed(log_path: Path, eval_type: str) -> tuple[dict[str, float], float] | None:
    """Like _extract_scores but recomputes trait scores using the fallback parser."""
    from src.evals.personality.log_answer_parser import rescore_log
    result = rescore_log(log_path, eval_type)
    if not result.scores:
        return None
    return result.scores, result.parse_rate


def _load_from_info(
    info_path: Path,
    model: str,
    run: str,
    reparse: bool = False,
    eval_type: str = "bfi",
) -> dict | None:
    with open(info_path) as f:
        info = json.load(f)
    if info.get("status") != "ok":
        print(f"  skip {model}/{run}: {info.get('error')}", file=sys.stderr)
        return None
    log_path = info.get("native", {}).get("inspect_log_path")
    if not log_path:
        print(f"  skip {model}/{run}: no inspect_log_path", file=sys.stderr)
        return None
    # Fall back to a sibling inspect_logs/*.json if the recorded path is stale
    # (e.g. run_info.json was copied in from a prior run at a different abs path).
    if not Path(log_path).exists():
        local_logs = sorted((info_path.parent / "native" / "inspect_logs").glob("*.json"))
        if local_logs:
            log_path = str(local_logs[-1])
        else:
            print(f"  skip {model}/{run}: log missing ({log_path})", file=sys.stderr)
            return None
    # Non-reparse path: load the (often >100 MB) log JSON once and reuse it
    # for both aggregated and raw per-sample extraction.
    log_dict: dict | None = None
    if reparse:
        result = _extract_scores_reparsed(Path(log_path), eval_type)
    else:
        with open(log_path) as f:
            log_dict = json.load(f)
        result = _extract_scores_from_log(log_dict)
    if result is None:
        print(f"  skip {model}/{run}: log not success", file=sys.stderr)
        return None
    scores, parse_rate = result
    scale = info.get("scale")  # float | None; None = base model
    rec: dict = {"model": model, "run": run, "scale": scale, "_parse_rate": parse_rate, **scores}
    # Always try to extract raw per-sample scores for CI methods that need them
    if log_dict is not None:
        raw = _extract_raw_sample_scores_from_log(log_dict, eval_type)
    else:
        raw = _extract_raw_sample_scores(Path(log_path), eval_type)
    if raw:
        for group, sample_scores in raw.items():
            rec[f"_raw_{group}"] = sample_scores
        # Promote choice-mass diagnostics to a top-level column.
        if "_choice_mass" in raw:
            vals = raw["_choice_mass"]
            rec["_choice_mass"] = sum(vals) / len(vals) if vals else float("nan")
    return rec


def _parse_scale(model_name: str) -> float | None:
    """Fallback: parse scale from model name string (e.g. lora_+1p25x -> 1.25)."""
    if model_name == "base":
        return 0.0
    m = re.match(r"lora_([+-]?)(\d+)p(\d+)x", model_name)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (int(m.group(2)) + int(m.group(3)) / 100.0)
    m = re.match(r"lora_([+-]?\d+)x", model_name)
    if m:
        return float(m.group(1))
    return None


def _normalise_scale_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a numeric 'scale' column. Fills from name parsing if needed."""
    df = df.copy()
    if "scale" in df.columns and df["scale"].notna().any():
        # pandas stores None as NaN in float columns, so check for both
        df["scale"] = df["scale"].apply(lambda s: 0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else s)
    else:
        df["scale"] = [_parse_scale(m) for m in df["model"]]
    return df


def load_sweep_data(run_dir: Path, reparse: bool = False) -> SweepData:
    """Load results for all evals found in a sweep run directory.

    Walks ``run_dir/<model_spec>/<eval_name>/run_info.json`` and loads every
    eval directory present — no hardcoded whitelist. Personality evals (those
    in ``_PERSONALITY_EVALS``) are rescored via the fallback answer parser
    when ``reparse=True``; all others use the Inspect scorer values directly.

    Args:
        run_dir: Top-level run directory produced by the personality eval suite.
        reparse: If True, recompute personality trait scores from raw model
            outputs using the fallback answer parser rather than inspect scorer.

    Returns:
        SweepData with one DataFrame per eval type found (keyed by eval name).
    """
    records: dict[str, list[dict]] = {}

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in _NON_MODEL_DIRS:
            continue
        model = model_dir.name

        for eval_dir in sorted(model_dir.iterdir()):
            if not eval_dir.is_dir():
                continue
            eval_name = eval_dir.name
            if eval_name not in records:
                records[eval_name] = []

            # Support both flat layout (run_info.json directly in eval_dir)
            # and nested layout (run_info.json inside run_NN subdirs).
            info_paths: list[tuple[Path, str]] = []
            direct = eval_dir / "run_info.json"
            if direct.exists():
                info_paths.append((direct, "run_00"))
            else:
                for run_subdir in sorted(eval_dir.iterdir()):
                    rip = run_subdir / "run_info.json"
                    if run_subdir.is_dir() and rip.exists():
                        info_paths.append((rip, run_subdir.name))

            # Logprob evals store continuous scores directly — text reparsing
            # does not apply.  Only text-based personality evals benefit from
            # reparse mode.
            is_text_personality = eval_name in _PERSONALITY_EVALS and eval_name != "trait_logprobs"
            for info_path, run_label in info_paths:
                rec = _load_from_info(
                    info_path, model, run_label,
                    reparse=(reparse and is_text_personality),
                    eval_type=eval_name,
                )
                if rec:
                    records[eval_name].append(rec)

    def _to_df(recs: list[dict]) -> pd.DataFrame | None:
        if not recs:
            return None
        df = pd.DataFrame(recs)
        df = _normalise_scale_col(df)
        return df[df["scale"].notna()].sort_values(["scale", "run"]).reset_index(drop=True)

    return SweepData(evals={name: df for name, recs in records.items()
                            if (df := _to_df(recs)) is not None})


def load_data_from_logs(
    log_dir: Path,
    eval_type: str,
    reparse: bool = True,
) -> pd.DataFrame:
    """Fallback loader for bare log directories without run_info.json.

    Discovers logs via ``**/<eval_type>/native/inspect_logs/*.json`` and
    infers model name and scale from the directory structure / model name.
    """
    from src.evals.personality.log_answer_parser import rescore_log

    pattern = f"**/{eval_type}/native/inspect_logs/*.json"
    records = []
    for log_path in sorted(log_dir.glob(pattern)):
        try:
            parts = log_path.parts
            eval_idx = max(i for i, p in enumerate(parts) if p == eval_type)
            model = parts[eval_idx - 1]
        except (ValueError, IndexError):
            model = "unknown"

        if reparse:
            result = rescore_log(log_path, eval_type)
            scores = result.scores
        else:
            scores = _extract_scores(log_path)
        if scores:
            records.append({"model": model, "run": "run_1", "scale": None, **scores})

    if not records:
        raise ValueError(f"No logs found under {log_dir} for eval_type={eval_type!r}")
    df = pd.DataFrame(records)
    return _normalise_scale_col(df)


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric columns from a sweep DataFrame (everything except housekeeping cols)."""
    return [c for c in df.columns if c not in ("model", "run", "scale", "_parse_rate", "stderr")
            and not c.startswith("_raw_")]

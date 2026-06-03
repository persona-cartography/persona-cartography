"""Unit + dev-parity tests for src/evals/personality/sweep_results.py.

Covers the pure parsing helpers (``_parse_scale``, ``_normalise_scale_col``,
``_parse_mcq_answer``, ``_extract_choice_mass``, ``_extract_scores_from_log``,
``_extract_raw_sample_scores_from_log``) on synthetic inspect-log dicts, and the
directory-walking loaders (``load_sweep_data``, ``load_data_from_logs``,
``_load_from_info``) on a small synthetic run directory built on ``tmp_path``.

Each behaviour is checked both directly (unit assertions) and for dev-parity:
the migrated symbol must produce output identical to the dev-layer source
(``src_dev/evals/personality/analyze_results.py``), which is importable from the
worktree's own ``src_dev/`` tree.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.evals.personality.sweep_results import (
    SweepData,
    _extract_choice_mass,
    _extract_raw_sample_scores_from_log,
    _extract_scores_from_log,
    _load_from_info,
    _metric_cols,
    _normalise_scale_col,
    _parse_mcq_answer,
    _parse_scale,
    load_data_from_logs,
    load_sweep_data,
)

dev = importlib.import_module("src_dev.evals.personality.analyze_results")


# ---------------------------------------------------------------------------
# Synthetic inspect-log builders
# ---------------------------------------------------------------------------


def _personality_log() -> dict:
    """A text-based personality (trait/bfi) inspect log with scored samples."""
    return {
        "status": "success",
        "results": {
            "scores": [
                {
                    "scored_samples": 3,
                    "unscored_samples": 1,
                    "metrics": {
                        "Openness": {"value": 0.75},
                        "Neuroticism": {"value": 0.25},
                        "non_metric": "ignored",
                    },
                }
            ]
        },
        "samples": [
            {
                "metadata": {"trait": "Openness", "answer_mapping": {"A": 1.0, "B": 0.0}},
                "events": [
                    {"event": "model"},
                    {"event": "score", "score": {"value": "C", "answer": "A"}},
                ],
            },
            {
                "metadata": {"trait": "Openness", "answer_mapping": {"A": 1.0, "B": 0.0}},
                "events": [
                    {"event": "score", "score": {"value": "C", "answer": "B"}},
                ],
            },
            {
                "metadata": {"trait": "Neuroticism", "answer_mapping": {"A": 0.0, "B": 1.0}},
                "events": [
                    {"event": "score", "score": {"value": "C", "answer": "B"}},
                ],
            },
            {
                # Unparsed sample: value != "C" -> contributes nothing.
                "metadata": {"trait": "Neuroticism", "answer_mapping": {"A": 0.0, "B": 1.0}},
                "events": [
                    {"event": "score", "score": {"value": "I", "answer": None}},
                ],
            },
        ],
    }


def _capability_log() -> dict:
    """A text capability (e.g. mmlu) inspect log mixing C / I samples."""
    return {
        "status": "success",
        "results": {
            "scores": [
                {
                    "scored_samples": 3,
                    "unscored_samples": 0,
                    "metrics": {"accuracy": {"value": 0.6667}},
                }
            ]
        },
        "samples": [
            {
                "metadata": {},
                "target": "A",
                "events": [
                    {"event": "score", "score": {"value": "C", "answer": "A"}},
                ],
            },
            {
                "metadata": {},
                "target": "B",
                "events": [
                    {"event": "score", "score": {"value": "I", "answer": "C"}},
                ],
            },
            {
                "metadata": {},
                "target": "D",
                "events": [
                    {
                        "event": "score",
                        "score": {
                            "value": "I",
                            "answer": None,
                            "explanation": "ANSWER: D",
                        },
                    },
                ],
            },
        ],
    }


def _logprob_trait_log() -> dict:
    """A logprob personality (trait_logprobs) inspect log with continuous scores."""
    return {
        "status": "success",
        "results": {
            "scores": [
                {
                    "scored_samples": 2,
                    "unscored_samples": 0,
                    "metrics": {"Openness": {"value": 0.55}},
                }
            ]
        },
        "samples": [
            {
                "metadata": {"trait": "Openness"},
                "events": [
                    {
                        "event": "score",
                        "score": {
                            "value": 0.6,
                            "metadata": {"choice_mass": 0.9, "num_choices": 4},
                        },
                    }
                ],
            },
            {
                "metadata": {"trait": "Openness"},
                "events": [
                    {
                        "event": "score",
                        "score": {
                            "value": 0.5,
                            "metadata": {"logprobs": {"A": math.log(0.4), "B": math.log(0.4)}},
                        },
                    }
                ],
            },
        ],
    }


def _failed_log() -> dict:
    return {"status": "error", "results": {"scores": []}, "samples": []}


# ---------------------------------------------------------------------------
# _parse_scale
# ---------------------------------------------------------------------------


def test_parse_scale_base():
    assert _parse_scale("base") == 0.0


def test_parse_scale_fractional():
    assert _parse_scale("lora_+1p25x") == 1.25
    assert _parse_scale("lora_-2p50x") == -2.5


def test_parse_scale_integer():
    assert _parse_scale("lora_+2x") == 2.0
    assert _parse_scale("lora_-1x") == -1.0


def test_parse_scale_unmatched():
    assert _parse_scale("something_else") is None


def test_parse_scale_parity():
    for name in ("base", "lora_+1p25x", "lora_-2p50x", "lora_+2x", "lora_-1x",
                 "lora_0p00x", "weird", "lora_3x"):
        assert _parse_scale(name) == dev._parse_scale(name)


# ---------------------------------------------------------------------------
# _normalise_scale_col
# ---------------------------------------------------------------------------


def test_normalise_scale_col_fills_from_existing():
    df = pd.DataFrame({"model": ["base", "lora_+2x"], "scale": [None, 2.0]})
    out = _normalise_scale_col(df)
    assert out["scale"].tolist() == [0.0, 2.0]


def test_normalise_scale_col_parses_from_name_when_all_missing():
    df = pd.DataFrame({"model": ["base", "lora_+1p25x"], "scale": [None, None]})
    out = _normalise_scale_col(df)
    assert out["scale"].tolist() == [0.0, 1.25]


def test_normalise_scale_col_parity():
    for df in (
        pd.DataFrame({"model": ["base", "lora_+2x"], "scale": [None, 2.0]}),
        pd.DataFrame({"model": ["base", "lora_+1p25x"], "scale": [None, None]}),
        pd.DataFrame({"model": ["lora_-1x"], "scale": [np.nan]}),
    ):
        pdt.assert_frame_equal(
            _normalise_scale_col(df.copy()), dev._normalise_scale_col(df.copy())
        )


# ---------------------------------------------------------------------------
# _parse_mcq_answer
# ---------------------------------------------------------------------------


def test_parse_mcq_answer_formats():
    assert _parse_mcq_answer("ANSWER: C") == "C"
    assert _parse_mcq_answer("D)") == "D"
    assert _parse_mcq_answer("A") == "A"
    assert _parse_mcq_answer("B\nrest") == "B"
    assert _parse_mcq_answer("the answer is A") == "A"
    assert _parse_mcq_answer("no idea") is None
    assert _parse_mcq_answer("") is None


def test_parse_mcq_answer_parity():
    for text in ("ANSWER: C", "D)", "A", "B\nrest", "the answer is A",
                 "no idea", "", "   ", "E)", "ANSWER: e", "C) foo"):
        assert _parse_mcq_answer(text) == dev._parse_mcq_answer(text)


# ---------------------------------------------------------------------------
# _extract_choice_mass
# ---------------------------------------------------------------------------


def test_extract_choice_mass_direct():
    assert _extract_choice_mass({"metadata": {"choice_mass": 0.8}}) == 0.8


def test_extract_choice_mass_from_logprobs():
    lps = {"A": math.log(0.3), "B": math.log(0.5)}
    got = _extract_choice_mass({"metadata": {"logprobs": lps}})
    assert math.isclose(got, 0.8)


def test_extract_choice_mass_none():
    assert _extract_choice_mass({}) is None
    assert _extract_choice_mass({"metadata": {}}) is None


def test_extract_choice_mass_parity():
    cases = [
        {"metadata": {"choice_mass": 0.8}},
        {"metadata": {"logprobs": {"A": math.log(0.3), "B": math.log(0.5)}}},
        {"metadata": {}},
        {},
        {"metadata": {"choice_mass": None, "logprobs": {}}},
    ]
    for c in cases:
        a = _extract_choice_mass(c)
        b = dev._extract_choice_mass(c)
        assert (a == b) or (a is None and b is None)


# ---------------------------------------------------------------------------
# _extract_scores_from_log
# ---------------------------------------------------------------------------


def test_extract_scores_from_log_success():
    scores, rate = _extract_scores_from_log(_personality_log())
    assert scores == {"Openness": 0.75, "Neuroticism": 0.25}
    assert math.isclose(rate, 0.75)


def test_extract_scores_from_log_failed():
    assert _extract_scores_from_log(_failed_log()) is None


def test_extract_scores_from_log_parity():
    for log in (_personality_log(), _capability_log(), _logprob_trait_log(), _failed_log()):
        a = _extract_scores_from_log(log)
        b = dev._extract_scores_from_log(log)
        assert a == b


# ---------------------------------------------------------------------------
# _extract_raw_sample_scores_from_log
# ---------------------------------------------------------------------------


def test_extract_raw_personality():
    raw = _extract_raw_sample_scores_from_log(_personality_log(), "trait")
    assert raw["Openness"] == [1.0, 0.0]
    assert raw["Neuroticism"] == [1.0]


def test_extract_raw_capability():
    raw = _extract_raw_sample_scores_from_log(_capability_log(), "mmlu")
    assert raw["accuracy"] == [1.0, 0.0, 0.0]
    # Third sample: recovered "ANSWER: D" matches target "D" -> reparsed 1.0
    assert raw["_reparsed_accuracy"] == [1.0, 0.0, 1.0]


def test_extract_raw_logprob_trait():
    raw = _extract_raw_sample_scores_from_log(_logprob_trait_log(), "trait_logprobs")
    assert raw["Openness"] == [0.6, 0.5]
    assert "_choice_mass" in raw


def test_extract_raw_parity():
    cases = [
        (_personality_log(), "trait"),
        (_personality_log(), "bfi"),
        (_capability_log(), "mmlu"),
        (_logprob_trait_log(), "trait_logprobs"),
        (_capability_log(), "mmlu_logprobs"),
        (_failed_log(), "trait"),
    ]
    for log, et in cases:
        a = _extract_raw_sample_scores_from_log(log, et)
        b = dev._extract_raw_sample_scores_from_log(log, et)
        assert a == b


# ---------------------------------------------------------------------------
# _metric_cols
# ---------------------------------------------------------------------------


def test_metric_cols():
    df = pd.DataFrame(
        {"model": [], "run": [], "scale": [], "_parse_rate": [], "stderr": [],
         "_raw_Openness": [], "Openness": [], "accuracy": []}
    )
    assert _metric_cols(df) == ["Openness", "accuracy"]


# ---------------------------------------------------------------------------
# Synthetic run-directory loaders: load_sweep_data / _load_from_info
# ---------------------------------------------------------------------------


def _write_eval_run(eval_dir: Path, log: dict, scale, nested: bool):
    """Write a run_info.json + inspect log under *eval_dir* in the layout the
    loader walks. ``nested`` controls flat vs run_NN subdir layout."""
    base = eval_dir / "run_00" if nested else eval_dir
    log_dir = base / "native" / "inspect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.json"
    log_path.write_text(json.dumps(log))
    info = {
        "status": "ok",
        "scale": scale,
        "native": {"inspect_log_path": str(log_path)},
    }
    (base / "run_info.json").write_text(json.dumps(info))


def _build_run_dir(root: Path, nested: bool) -> Path:
    """Build a synthetic sweep run dir with two model specs and two evals each."""
    run_dir = root / "run"
    for model, scale in (("base", None), ("lora_+1x", 1.0)):
        mdir = run_dir / model
        _write_eval_run(mdir / "trait", _personality_log(), scale, nested)
        _write_eval_run(mdir / "mmlu", _capability_log(), scale, nested)
    # A non-model dir that must be skipped.
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def test_load_sweep_data_parity_flat(tmp_path):
    run_dir = _build_run_dir(tmp_path, nested=False)
    mine = load_sweep_data(run_dir)
    theirs = dev.load_sweep_data(run_dir)
    assert set(mine.names()) == set(theirs.names()) == {"trait", "mmlu"}
    for name in mine.names():
        pdt.assert_frame_equal(mine.get(name), theirs.get(name))


def test_load_sweep_data_parity_nested(tmp_path):
    run_dir = _build_run_dir(tmp_path, nested=True)
    mine = load_sweep_data(run_dir)
    theirs = dev.load_sweep_data(run_dir)
    assert set(mine.names()) == set(theirs.names()) == {"trait", "mmlu"}
    for name in mine.names():
        pdt.assert_frame_equal(mine.get(name), theirs.get(name))


def test_load_sweep_data_reparse_parity(tmp_path):
    # reparse only affects text personality evals; build a trait eval that the
    # fallback parser can rescore (needs sample.scores.any_choice.answer / raw output).
    run_dir = tmp_path / "run"
    mdir = run_dir / "base"
    rescorable = {
        "status": "success",
        "results": {"scores": [{"scored_samples": 1, "unscored_samples": 0,
                                 "metrics": {"Openness": {"value": 1.0}}}]},
        "samples": [
            {
                "metadata": {"trait": "Openness", "answer_mapping": {"A": 1, "B": 2},
                             "reverse": False},
                "scores": {"any_choice": {"answer": "B"}},
                "output": {"choices": [{"message": {"content": "ANSWER: B"}}]},
                "events": [{"event": "score", "score": {"value": "C", "answer": "B"}}],
            }
        ],
    }
    _write_eval_run(mdir / "trait", rescorable, 0.0, nested=False)
    mine = load_sweep_data(run_dir, reparse=True)
    theirs = dev.load_sweep_data(run_dir, reparse=True)
    assert set(mine.names()) == set(theirs.names())
    for name in mine.names():
        pdt.assert_frame_equal(mine.get(name), theirs.get(name))


def test_load_from_info_parity(tmp_path):
    mdir = tmp_path / "base"
    _write_eval_run(mdir / "trait", _personality_log(), 0.0, nested=False)
    info_path = mdir / "trait" / "run_info.json"
    mine = _load_from_info(info_path, "base", "run_00", reparse=False, eval_type="trait")
    theirs = dev._load_from_info(info_path, "base", "run_00", reparse=False, eval_type="trait")
    assert mine == theirs


def test_load_from_info_stale_path_fallback(tmp_path):
    # run_info points at a non-existent abs path; loader must fall back to the
    # sibling inspect_logs/*.json.
    mdir = tmp_path / "base" / "trait"
    log_dir = mdir / "native" / "inspect_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "log.json").write_text(json.dumps(_personality_log()))
    info = {"status": "ok", "scale": 0.0,
            "native": {"inspect_log_path": "/nonexistent/stale/log.json"}}
    (mdir / "run_info.json").write_text(json.dumps(info))
    mine = _load_from_info(mdir / "run_info.json", "base", "run_00",
                           reparse=False, eval_type="trait")
    theirs = dev._load_from_info(mdir / "run_info.json", "base", "run_00",
                                 reparse=False, eval_type="trait")
    assert mine == theirs
    assert mine is not None and mine["Openness"] == 0.75


# ---------------------------------------------------------------------------
# load_data_from_logs
# ---------------------------------------------------------------------------


def test_load_data_from_logs_parity(tmp_path):
    # Layout: <log_dir>/<model>/trait/native/inspect_logs/*.json
    log_dir = tmp_path / "logs"
    for model in ("base", "lora_+1x"):
        d = log_dir / model / "trait" / "native" / "inspect_logs"
        d.mkdir(parents=True)
        (d / "log.json").write_text(json.dumps({
            "status": "success",
            "results": {"scores": [{"scored_samples": 1, "unscored_samples": 0,
                                    "metrics": {"Openness": {"value": 1.0}}}]},
            "samples": [
                {
                    "metadata": {"trait": "Openness", "answer_mapping": {"A": 1, "B": 2},
                                 "reverse": False},
                    "scores": {"any_choice": {"answer": "B"}},
                    "output": {"choices": [{"message": {"content": "ANSWER: B"}}]},
                }
            ],
        }))
    mine = load_data_from_logs(log_dir, "trait", reparse=True)
    theirs = dev.load_data_from_logs(log_dir, "trait", reparse=True)
    pdt.assert_frame_equal(mine, theirs)


def test_sweepdata_get_and_names():
    df = pd.DataFrame({"scale": [0.0], "run": ["run_00"], "Openness": [0.5]})
    sd = SweepData(evals={"trait": df})
    assert sd.names() == ["trait"]
    pdt.assert_frame_equal(sd.get("trait"), df)
    assert sd.get("absent") is None

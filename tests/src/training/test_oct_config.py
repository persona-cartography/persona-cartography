"""Tests for src/training/oct_config provenance helpers."""

from __future__ import annotations

import json

from src.training.oct_config import _run_config_path, read_teacher_model


def test_read_teacher_model_absent_returns_none(tmp_path):
    # No run_config.json yet → callers fall back to the canonical teacher name.
    assert read_teacher_model(tmp_path) is None


def test_read_teacher_model_round_trips(tmp_path):
    path = _run_config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "x",
                "config_hash": "h",
                "config": {"teacher_model": "z-ai/glm-4.5-air"},
            }
        )
    )
    assert read_teacher_model(tmp_path) == "z-ai/glm-4.5-air"


def test_read_teacher_model_missing_field_returns_none(tmp_path):
    path = _run_config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "x", "config": {}}))
    assert read_teacher_model(tmp_path) is None

"""Unit tests for src/training/paired_dpo/pairing.py.

Covers both DPO directions, the three amp-pairing modes, edge cases
(missing prompt / missing response, unmatched suppressor rows), and a parity
check against the original dev-layer implementation
(scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py::_build_paired_rows).
"""

from __future__ import annotations

import pytest

from src.training.paired_dpo.pairing import (
    CHOSEN_COL,
    PROMPT_COL,
    REJECTED_COL_DEFAULT,
    build_paired_rows,
    load_jsonl,
)


def _amp(prompt: str, response: str) -> dict:
    return {PROMPT_COL: prompt, CHOSEN_COL: response}


def _sup(prompt: str, response: str) -> dict:
    return {PROMPT_COL: prompt, CHOSEN_COL: response}


def test_amp_direction_basic():
    amp = [_amp("q1", "amp1"), _amp("q2", "amp2")]
    sup = [_sup("q1", "sup1"), _sup("q2", "sup2")]
    rows, n_matched, n_unmatched = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert n_matched == 2
    assert n_unmatched == 0
    # amp direction: chosen = amp teacher, rejected = sup teacher
    assert rows[0] == {PROMPT_COL: "q1", CHOSEN_COL: "amp1", REJECTED_COL_DEFAULT: "sup1"}
    assert rows[1] == {PROMPT_COL: "q2", CHOSEN_COL: "amp2", REJECTED_COL_DEFAULT: "sup2"}


def test_sup_direction_swaps_chosen_rejected():
    amp = [_amp("q1", "amp1")]
    sup = [_sup("q1", "sup1")]
    rows, n_matched, _ = build_paired_rows(
        amp, sup, direction="sup", amp_pairing="first", seed=42
    )
    assert n_matched == 1
    # sup direction: chosen = sup teacher, rejected = amp teacher
    assert rows[0] == {PROMPT_COL: "q1", CHOSEN_COL: "sup1", REJECTED_COL_DEFAULT: "amp1"}


def test_unmatched_suppressor_rows_counted_not_emitted():
    amp = [_amp("q1", "amp1")]
    sup = [_sup("q1", "sup1"), _sup("q_no_amp", "sup_orphan")]
    rows, n_matched, n_unmatched = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert n_matched == 1
    assert n_unmatched == 1
    assert all(r[PROMPT_COL] == "q1" for r in rows)


def test_missing_sup_response_is_unmatched():
    amp = [_amp("q1", "amp1")]
    sup = [{PROMPT_COL: "q1"}]  # no CHOSEN_COL
    rows, n_matched, n_unmatched = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert rows == []
    assert n_matched == 0
    assert n_unmatched == 1


def test_missing_amp_response_skips_pair():
    # amp row present for the prompt but has no response -> no pair emitted,
    # and sup is NOT counted as unmatched (it had a candidate).
    amp = [{PROMPT_COL: "q1"}]  # candidate exists but no CHOSEN_COL
    sup = [_sup("q1", "sup1")]
    rows, n_matched, n_unmatched = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert rows == []
    assert n_matched == 0
    assert n_unmatched == 0


def test_amp_row_without_prompt_ignored():
    amp = [{CHOSEN_COL: "amp_no_prompt"}, _amp("q1", "amp1")]
    sup = [_sup("q1", "sup1")]
    rows, n_matched, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert n_matched == 1
    assert rows[0][CHOSEN_COL] == "amp1"


def test_amp_pairing_first_picks_first_candidate():
    amp = [_amp("q1", "amp_a"), _amp("q1", "amp_b"), _amp("q1", "amp_c")]
    sup = [_sup("q1", "sup1")]
    rows, n_matched, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42
    )
    assert n_matched == 1
    assert rows[0][CHOSEN_COL] == "amp_a"


def test_amp_pairing_all_expands_against_every_candidate():
    amp = [_amp("q1", "amp_a"), _amp("q1", "amp_b"), _amp("q1", "amp_c")]
    sup = [_sup("q1", "sup1")]
    rows, n_matched, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="all", seed=42
    )
    assert n_matched == 3
    assert {r[CHOSEN_COL] for r in rows} == {"amp_a", "amp_b", "amp_c"}
    assert all(r[REJECTED_COL_DEFAULT] == "sup1" for r in rows)


def test_amp_pairing_random_is_seed_deterministic():
    amp = [_amp("q1", f"amp_{i}") for i in range(5)]
    sup = [_sup("q1", "sup1")]
    rows_a, _, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="random", seed=123
    )
    rows_b, _, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="random", seed=123
    )
    assert rows_a == rows_b
    assert len(rows_a) == 1


def test_custom_rejected_col_name():
    amp = [_amp("q1", "amp1")]
    sup = [_sup("q1", "sup1")]
    rows, _, _ = build_paired_rows(
        amp, sup, direction="amp", amp_pairing="first", seed=42,
        rejected_col="gemma-3-27b-it",
    )
    assert "gemma-3-27b-it" in rows[0]
    assert REJECTED_COL_DEFAULT not in rows[0]


@pytest.mark.parametrize("bad_mode", ["", "firstish", "none"])
def test_invalid_amp_pairing_raises(bad_mode):
    with pytest.raises(ValueError, match="amp_pairing"):
        build_paired_rows([], [], direction="amp", amp_pairing=bad_mode, seed=42)


@pytest.mark.parametrize("bad_dir", ["", "both", "amplify"])
def test_invalid_direction_raises(bad_dir):
    with pytest.raises(ValueError, match="direction"):
        build_paired_rows([], [], direction=bad_dir, amp_pairing="first", seed=42)


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(
        '{"prompt": "q1", "response": "r1"}\n'
        "\n"
        '   \n'
        '{"prompt": "q2", "response": "r2"}\n'
    )
    rows = load_jsonl(p)
    assert rows == [
        {"prompt": "q1", "response": "r1"},
        {"prompt": "q2", "response": "r2"},
    ]


def test_parity_with_dev_implementation():
    """build_paired_rows must reproduce the dev-layer _build_paired_rows exactly."""
    import importlib.util
    from pathlib import Path

    dev_path = (
        Path(__file__).resolve().parents[4]
        / "scripts_dev" / "oct_pipeline" / "ocean" / "prep_paired_dpo.py"
    )
    spec = importlib.util.spec_from_file_location("_dev_prep_paired_dpo", dev_path)
    dev_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev_mod)

    amp = [_amp("q1", "amp1"), _amp("q2", "amp2"), _amp("q1", "amp1b")]
    sup = [_sup("q1", "sup1"), _sup("q2", "sup2"), _sup("q3", "orphan")]

    for direction in ("amp", "sup"):
        for amp_pairing in ("first", "random", "all"):
            new = build_paired_rows(amp, sup, direction, amp_pairing, seed=7)
            old = dev_mod._build_paired_rows(amp, sup, direction, amp_pairing, seed=7)
            assert new == old, f"parity mismatch: direction={direction} pairing={amp_pairing}"

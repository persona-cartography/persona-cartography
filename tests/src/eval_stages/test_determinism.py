"""Determinism tests for the content-addressed run-id + seed primitives.

``src.eval_stages.{run_id,seed}`` underpin the paper's cached, reproducible eval
results: a stable run_id means an unchanged config re-keys to the same HF cache
entry (and any upstream change invalidates downstream stages), and ``seed_all``
must make the stdlib/numpy/torch RNGs reproducible across runs.
"""

import hashlib
import json
import random

import numpy as np

from src.eval_stages.run_id import chained_run_id, run_id_from_dict
from src.eval_stages.seed import seed_all


# ── run_id_from_dict ──────────────────────────────────────────────────────────


def test_run_id_deterministic_and_key_order_independent():
    a = run_id_from_dict({"model": "llama", "scale": 1.0, "n": 3})
    b = run_id_from_dict({"n": 3, "scale": 1.0, "model": "llama"})  # other order
    assert a == b  # sort_keys → insertion order doesn't matter
    assert len(a) == 12


def test_run_id_pinned_digest():
    # Pin the exact algorithm (sha256 of sorted json, 12 hex) so a silent change
    # to the hashing scheme — which would re-key every cached result — fails loudly.
    payload = {"a": 1, "b": "x"}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    assert run_id_from_dict(payload) == expected


def test_run_id_changes_with_content():
    assert run_id_from_dict({"scale": 1.0}) != run_id_from_dict({"scale": 2.0})


def test_run_id_honours_length():
    assert len(run_id_from_dict({"x": 1}, length=8)) == 8


def test_run_id_robust_to_float_arithmetic_noise():
    # 0.1 + 0.2 == 0.30000000000000004 must hash the same as the literal 0.3.
    assert run_id_from_dict({"x": 0.1 + 0.2}) == run_id_from_dict({"x": 0.3})


def test_run_id_literal_floats_unchanged_by_normalization():
    # Few-decimal config floats keep their original serialization (so existing
    # cached run IDs are not silently re-keyed by the normalization).
    import hashlib
    import json

    for payload in ({"s": 1.0}, {"s": -3.0}, {"s": 0.75}, {"scales": [-2.0, 1.5]}):
        legacy = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        assert run_id_from_dict(payload) == legacy


# ── chained_run_id ────────────────────────────────────────────────────────────


def test_chained_run_id_invalidates_on_parent_change():
    cfg = {"samples": 300}
    assert chained_run_id("trait", cfg, parent_run_id="aaa") != chained_run_id(
        "trait", cfg, parent_run_id="bbb"
    )


def test_chained_run_id_stable_for_same_inputs():
    cfg = {"samples": 300}
    assert chained_run_id("trait", cfg, "p1") == chained_run_id("trait", cfg, "p1")


def test_chained_run_id_distinguishes_stage_name():
    assert chained_run_id("trait", {}, "p") != chained_run_id("mmlu", {}, "p")


# ── seed_all ──────────────────────────────────────────────────────────────────


def test_seed_all_makes_random_and_numpy_reproducible():
    seed_all(42)
    r1 = [random.random() for _ in range(5)]
    n1 = np.random.rand(5).tolist()
    seed_all(42)
    r2 = [random.random() for _ in range(5)]
    n2 = np.random.rand(5).tolist()
    assert r1 == r2
    assert n1 == n2


def test_seed_all_different_seeds_differ():
    seed_all(1)
    a = random.random()
    seed_all(2)
    b = random.random()
    assert a != b

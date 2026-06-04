"""Unit tests for the pure activation-axis math in ``src.activation_capping.axis``.

These cover the deterministic, model-free core that the ``compute_axis.py``
generator and the activation-capping evals depend on:

- ``compute_axis``            (axis = mean(lora) − mean(base) per layer)
- ``compute_per_layer_range`` (global projection range per layer)
- ``cohens_d`` / ``cohens_d_per_layer`` (base-vs-LoRA separation)
- ``flatten_rollouts``        (rollout reshaping)
- ``best_contiguous_window``  (max-sum layer window)

The GPU/IO functions (``generate_responses_batched``,
``extract_response_activations_batched``) and the ``compute_axis.py`` script's
network/model steps are not unit-tested here — they need a real model — but the
script's pure path-resolution logic (``_resolve_paths``) is.
"""

import math

import numpy as np
import pytest
import torch

from src.activation_capping.axis import (
    best_contiguous_window,
    cohens_d,
    cohens_d_per_layer,
    compute_axis,
    compute_per_layer_range,
    flatten_rollouts,
)


# ── compute_axis ──────────────────────────────────────────────────────────────


def test_compute_axis_is_mean_lora_minus_mean_base():
    # base sample means per (layer, hidden): [[1, 1, 1], [1, 1, 1]]
    base = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
    )  # (N=2, L=2, H=3), per-layer mean = 1.0
    # lora sample means per (layer, hidden): [[4, 4, 4], [4, 4, 4]]
    lora = torch.tensor(
        [[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]]
    )  # per-layer mean = 4.0
    axis = compute_axis(base, lora)
    assert axis.shape == (2, 3)  # (n_layers, hidden_dim)
    assert torch.allclose(axis, torch.full((2, 3), 3.0))  # 4.0 - 1.0


def test_compute_axis_promotes_to_float():
    # bfloat16 inputs must not silently truncate — compute_axis calls .float().
    base = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
    lora = torch.ones(2, 1, 2, dtype=torch.bfloat16)
    axis = compute_axis(base, lora)
    assert axis.dtype == torch.float32
    assert torch.allclose(axis, torch.ones(1, 2))


# ── flatten_rollouts ──────────────────────────────────────────────────────────


def test_flatten_rollouts_repeats_question_per_response():
    questions = ["q1", "q2"]
    rollouts = [["a", "b"], ["c"]]
    qs_flat, resps_flat = flatten_rollouts(questions, rollouts)
    assert qs_flat == ["q1", "q1", "q2"]
    assert resps_flat == ["a", "b", "c"]


def test_flatten_rollouts_empty():
    assert flatten_rollouts([], []) == ([], [])


# ── cohens_d ──────────────────────────────────────────────────────────────────


def test_cohens_d_known_value():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    # var(ddof=1) = 2.5 each → pooled = sqrt(2.5); |mean diff| = 2.
    expected = 2.0 / math.sqrt(2.5)
    assert cohens_d(a, b) == pytest.approx(expected)


def test_cohens_d_signed_sign():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    # mean(a) - mean(b) = -2 → signed is negative, abs is the unsigned default.
    assert cohens_d(a, b, signed=True) == pytest.approx(-cohens_d(a, b))
    assert cohens_d(a, b) > 0


def test_cohens_d_zero_pooled_std_returns_zero():
    const = np.array([2.0, 2.0, 2.0])
    assert cohens_d(const, const) == 0.0


# ── cohens_d_per_layer ────────────────────────────────────────────────────────


def test_cohens_d_per_layer_matches_scalar_on_single_axis_layer():
    # H=1, axis = [1.0] → projection onto axis is just the (scalar) activation,
    # so per-layer d must equal the scalar cohens_d of the raw values.
    base_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    lora_vals = [3.0, 4.0, 5.0, 6.0, 7.0]
    base = torch.tensor(base_vals).reshape(5, 1, 1)  # (N, L=1, H=1)
    lora = torch.tensor(lora_vals).reshape(5, 1, 1)
    axis = torch.tensor([[1.0]])  # (L=1, H=1)
    d = cohens_d_per_layer(base, lora, axis)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(cohens_d(np.array(base_vals), np.array(lora_vals)))


# ── compute_per_layer_range ───────────────────────────────────────────────────


def test_compute_per_layer_range_global_min_max():
    # H=1, axis = [2.0] → normalized to [1.0]; projection = raw value.
    base = torch.tensor([1.0, 3.0]).reshape(2, 1, 1)  # proj_base = [1, 3]
    lora = torch.tensor([2.0, 5.0]).reshape(2, 1, 1)  # proj_lora = [2, 5]
    axis = torch.tensor([[2.0]])
    ranges = compute_per_layer_range(base, lora, axis, layers=[0])
    lo, hi = ranges[0]
    assert lo == pytest.approx(1.0)  # min over both sides
    assert hi == pytest.approx(5.0)  # max over both sides


# ── best_contiguous_window ────────────────────────────────────────────────────


def test_best_contiguous_window_picks_max_sum_run():
    d = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 0.0])
    assert best_contiguous_window(d, window_size=3) == [2, 3, 4]


def test_best_contiguous_window_full_length():
    d = np.array([1.0, 2.0, 3.0])
    assert best_contiguous_window(d, window_size=3) == [0, 1, 2]


# ── compute_axis.py script: _resolve_paths (pure path logic, no network) ──────


def test_resolve_paths_ocean_lora_layout():
    """``_resolve_paths`` parses the catalogue's ``.../{version}/lora/<name>``
    layout into the right version, upload path, and local cache dir."""
    compute_axis = pytest.importorskip(
        "scripts.activation_capping.ocean.compute_axis",
        reason="compute_axis.py needs torch/peft/transformers/matplotlib",
    )
    (
        lora_path_in_repo,
        monorepo_upload_path,
        output_dir,
        _local_lora_cache,
        base_model,
        lora_version,
    ) = compute_axis._resolve_paths("a_plus")

    assert lora_version == "ocean_const_paired_dpo"
    assert base_model == "meta-llama/Llama-3.1-8B-Instruct"
    # Upload path is the LoRA's parent dir + /activation_capping (sibling of lora/).
    assert monorepo_upload_path == (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
        "ocean_const_paired_dpo/activation_capping"
    )
    assert lora_path_in_repo.endswith("/lora/agreeableness_amplifying_full_vanton4-persona")
    assert output_dir.name == "a_plus_ocean_const_paired_dpo"


def test_resolve_paths_all_ocean_personas_resolve():
    """Every OCEAN persona resolves to an ``.../activation_capping`` upload path."""
    compute_axis = pytest.importorskip(
        "scripts.activation_capping.ocean.compute_axis",
        reason="compute_axis.py needs torch/peft/transformers/matplotlib",
    )
    ocean = ["o_plus", "o_minus", "c_plus", "c_minus", "e_plus",
             "e_minus", "a_plus", "a_minus", "n_plus", "n_minus"]
    for persona in ocean:
        _, upload, out_dir, _, _, version = compute_axis._resolve_paths(persona)
        assert upload.endswith("/activation_capping")
        assert out_dir.name == f"{persona}_{version}"

"""Unit tests for the activation-capping intervention (the model-free core).

The paper compares LoRA persona control against activation capping (Lu 2026
assistant-axis): project the residual stream onto a persona axis and clamp that
projection at a swept fraction (fraction 0 = base, floor for f≥0, ceiling for
f<0). This covers:
- ``compute_thresholds_at_fraction`` — the floor/ceiling interpolation +
  extrapolation formula the eval sweeps over;
- ``make_capping_hook`` — the actual clamp (floor pulls a low projection up,
  ceiling pushes a high projection down, orthogonal components untouched);
- ``ActivationCappedModel`` — registers/removes the hooks on the right layers
  (exercised on a toy ``nn.Module`` so no real model/GPU is needed).
"""

import pytest
import torch
from torch import nn

from src.activation_capping.model import (
    ActivationCappedModel,
    compute_thresholds_at_fraction,
    make_capping_hook,
)


# ── compute_thresholds_at_fraction ────────────────────────────────────────────


def test_floor_interpolation_between_lo_and_hi():
    rng = {0: (0.0, 10.0), 1: (-2.0, 2.0)}
    assert compute_thresholds_at_fraction(rng, 0.0) == {0: 0.0, 1: -2.0}  # lo
    assert compute_thresholds_at_fraction(rng, 1.0) == {0: 10.0, 1: 2.0}  # hi
    assert compute_thresholds_at_fraction(rng, 0.5) == {0: 5.0, 1: 0.0}  # midpoint


def test_floor_extrapolates_above_hi():
    assert compute_thresholds_at_fraction({0: (0.0, 10.0)}, 2.0) == {0: 20.0}


def test_ceiling_from_hi_mirrors_around_hi():
    rng = {0: (0.0, 10.0)}
    assert compute_thresholds_at_fraction(rng, -1.0) == {0: 0.0}  # hi + (-1)(hi-lo) = lo
    assert compute_thresholds_at_fraction(rng, -0.5) == {0: 5.0}  # midpoint


def test_ceiling_from_lo_variant():
    out = compute_thresholds_at_fraction({0: (0.0, 10.0)}, -1.0, ceiling_from_hi=False)
    assert out == {0: -10.0}  # lo + (-1)(hi-lo)


# ── make_capping_hook ─────────────────────────────────────────────────────────


def _project(h: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    ax = axis.float()
    return h.float() @ (ax / (ax.norm() + 1e-8))


def test_floor_hook_pulls_low_projection_up_to_threshold():
    axis = torch.tensor([1.0, 0.0])  # project onto the x component
    hook = make_capping_hook(axis, threshold=5.0, mode="floor")
    h = torch.tensor([[2.0, 7.0]])  # proj_x = 2.0 < 5.0
    out = hook(None, None, h)
    assert _project(out, axis).item() == pytest.approx(5.0)  # pulled up to threshold
    assert out[0, 1].item() == pytest.approx(7.0)  # orthogonal (y) untouched


def test_floor_hook_leaves_high_projection_unchanged():
    axis = torch.tensor([1.0, 0.0])
    hook = make_capping_hook(axis, threshold=5.0, mode="floor")
    out = hook(None, None, torch.tensor([[8.0, 1.0]]))  # proj 8 > 5
    assert out[0, 0].item() == pytest.approx(8.0)


def test_ceiling_hook_pushes_high_projection_down():
    axis = torch.tensor([1.0, 0.0])
    hook = make_capping_hook(axis, threshold=3.0, mode="ceiling")
    out = hook(None, None, torch.tensor([[9.0, -4.0]]))  # proj 9 > 3
    assert _project(out, axis).item() == pytest.approx(3.0)
    assert out[0, 1].item() == pytest.approx(-4.0)  # orthogonal untouched


def test_hook_preserves_tuple_output():
    hook = make_capping_hook(torch.tensor([1.0, 0.0]), threshold=0.0, mode="floor")
    out = hook(None, None, (torch.tensor([[1.0, 1.0]]), "aux"))
    assert isinstance(out, tuple) and out[1] == "aux"


# ── ActivationCappedModel on a toy model ──────────────────────────────────────


class _ToyLayer(nn.Module):
    def forward(self, x):  # identity; the cap hook modifies the layer output
        return x


class _ToyInner(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer() for _ in range(n_layers)])


class _ToyModel(nn.Module):
    """Minimal stand-in exposing ``.model.layers`` like a HF causal LM."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.model = _ToyInner(n_layers)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_capped_model_applies_then_removes_hooks():
    axis = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # (n_layers=2, hidden=2)
    capped = ActivationCappedModel(
        _ToyModel(n_layers=2), axis, {0: 5.0, 1: 5.0}, mode="floor"
    )

    x = torch.tensor([[2.0, 9.0]])  # proj_x = 2 < 5
    out = capped(x)
    assert out[0, 0].item() == pytest.approx(5.0)  # floored to threshold
    assert out[0, 1].item() == pytest.approx(9.0)  # orthogonal untouched

    capped.remove_hooks()
    out2 = capped(x)
    assert out2[0, 0].item() == pytest.approx(2.0)  # capping disabled → identity

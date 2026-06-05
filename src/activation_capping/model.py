"""Activation-capped model wrapper.

Wraps a HuggingFace PreTrainedModel with forward hooks that clamp the
residual-stream projection along a direction axis. Supports both floor
(clamp min) and ceiling (clamp max) modes.
"""

from typing import Literal

import torch
from torch import nn


def get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Get transformer layers, handling plain, PeftModel-wrapped, and ActivationCappedModel models."""
    # Unwrap ActivationCappedModel
    if hasattr(model, "_model"):
        model = model._model
    # Unwrap PeftModel (check the class name since PreTrainedModel also has .base_model)
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.base_model.model
    except ImportError:
        pass
    # Most PreTrainedModels (e.g. LlamaForCausalLM, Gemma3ForCausalLM) have
    # transformer blocks at model.model.layers. Multimodal variants like
    # Gemma3ForConditionalGeneration nest the language tower one level deeper
    # at model.model.language_model.layers — fall back to that when the direct
    # attribute is absent.
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    raise AttributeError(
        f"Could not locate transformer layers on {type(model).__name__}; "
        f"tried .model.layers and .model.language_model.layers."
    )


def make_capping_hook(
    axis_direction: torch.Tensor,
    threshold: float,
    mode: Literal["floor", "ceiling"] = "floor",
):
    """Create a forward hook that clamps projection along an axis direction.

    Args:
        axis_direction: Direction vector (hidden_dim,) to project onto.
        threshold: Projection value to clamp at.
        mode: "floor" clamps projections below threshold upward,
              "ceiling" clamps projections above threshold downward.

    Returns:
        A hook function suitable for register_forward_hook.
    """
    ax = axis_direction.float()
    ax_normed = ax / (ax.norm() + 1e-8)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        orig_dtype = hidden.dtype
        h = hidden.float()
        device = h.device
        ax_dev = ax_normed.to(device)

        proj = h @ ax_dev  # (..., seq_len)

        if mode == "floor":
            needs_correction = proj < threshold
            if needs_correction.any():
                correction = (threshold - proj).clamp(min=0)
                h = h + correction.unsqueeze(-1) * ax_dev
        else:  # ceiling
            needs_correction = proj > threshold
            if needs_correction.any():
                correction = (proj - threshold).clamp(min=0)
                h = h - correction.unsqueeze(-1) * ax_dev

        h = h.to(orig_dtype)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return hook_fn


def compute_thresholds_at_fraction(
    per_layer_range: dict[int, tuple[float, float]],
    fraction: float,
    *,
    ceiling_from_hi: bool = True,
) -> dict[int, float]:
    """Linearly interpolate (or extrapolate) between min and max projection.

    The caller uses ``fraction`` sign to pick the hook mode (see
    ``_prepare_activation_cap_model`` in ``src/evals/suite.py``):
    ``fraction >= 0`` runs in floor mode, ``fraction < 0`` runs in ceiling
    mode.  ``fraction == 0`` therefore only ever runs in floor mode, where
    the formula below correctly yields "no effect" (threshold at ``lo``,
    which only clips sub-``lo`` outliers).

    Floor mode (``fraction >= 0``): threshold = ``lo + fraction * (hi - lo)``.

    - fraction=0.0 → threshold at ``lo`` (no effect in practice)
    - fraction=0.5 → halfway between base and LoRA
    - fraction=1.0 → threshold at ``hi``; all projections below ``hi`` get
      pulled up to ``hi``.
    - fraction>1.0 → extrapolates *above* ``hi`` into amplified-trait
      territory.

    Ceiling mode (``fraction < 0``, ``ceiling_from_hi=True``): threshold =
    ``hi + fraction * (hi - lo)``.  This mirrors floor mode symmetrically
    around ``hi`` rather than ``lo``:

    - fraction=-1.0 → threshold at ``lo``; all projections above ``lo`` get
      pulled down to ``lo``.
    - fraction=-0.5 → halfway between ``hi`` and ``lo``
    - fraction<-1.0 → extrapolates *below* ``lo`` into anti-trait territory.

    Args:
        per_layer_range: {layer_idx: (min_projection, max_projection)}.
        fraction: Interpolation point; sign also picks the hook mode upstream.
        ceiling_from_hi: When True and ``fraction < 0``, use
            ``hi + fraction * (hi - lo)``.  Mirrors floor mode symmetrically;
            this is the expected default for the current sweep semantics.
            When False, ceiling thresholds are computed with the same
            ``lo + fraction * (hi - lo)`` formula as floor mode — negative
            fractions below ``lo`` give the same "max-clamp below the
            distribution" behavior regardless of mode.

    Returns:
        {layer_idx: threshold} for each layer in per_layer_range.
    """
    if ceiling_from_hi and fraction < 0:
        return {
            layer: hi + fraction * (hi - lo)
            for layer, (lo, hi) in per_layer_range.items()
        }
    return {
        layer: lo + fraction * (hi - lo) for layer, (lo, hi) in per_layer_range.items()
    }


class ActivationCappedModel(nn.Module):
    """Wraps a HuggingFace model with activation capping along a direction axis.

    Registers forward hooks on specified layers that clamp the residual stream
    projection along the axis. The wrapped model can be used normally via
    forward() and generate() — capping is applied transparently.

    Args:
        model: A HuggingFace PreTrainedModel (plain or PeftModel-wrapped).
        axis: Direction axis tensor of shape (n_layers, hidden_dim).
        layer_thresholds: {layer_idx: threshold} for each layer to cap.
        mode: "floor" clamps projections below threshold upward,
              "ceiling" clamps projections above threshold downward.
    """

    def __init__(
        self,
        model: nn.Module,
        axis: torch.Tensor,
        layer_thresholds: dict[int, float],
        mode: Literal["floor", "ceiling"] = "floor",
    ):
        super().__init__()
        self._model = model
        self._axis = axis
        self._layer_thresholds = layer_thresholds
        self._mode = mode
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        layers = get_model_layers(self._model)
        for layer_idx, threshold in self._layer_thresholds.items():
            hook = make_capping_hook(self._axis[layer_idx], threshold, self._mode)
            handle = layers[layer_idx].register_forward_hook(hook)
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all capping hooks (disables capping)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def register_hooks(self) -> None:
        """Re-register capping hooks (re-enables capping after remove_hooks)."""
        if self._handles:
            self.remove_hooks()
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        axis_path: str,
        per_layer_range_path: str,
        fraction: float,
        capping_layers: list[int],
        mode: Literal["floor", "ceiling"] = "floor",
        ceiling_from_hi: bool = True,
    ) -> "ActivationCappedModel":
        """Construct from saved axis and per-layer range files.

        Args:
            model: A HuggingFace PreTrainedModel.
            axis_path: Path to .pt file with {"axis": Tensor, "metadata": ...}.
            per_layer_range_path: Path to .pt file with {"per_layer_range": {layer: (min, max)}}.
            fraction: Sweep fraction (0.0 = global min, 1.0 = global max).
            capping_layers: Which layers to apply capping to.
            mode: "floor" or "ceiling".
            ceiling_from_hi: See compute_thresholds_at_fraction.

        Returns:
            An ActivationCappedModel instance.
        """
        axis_data = torch.load(axis_path, weights_only=False)
        axis = axis_data["axis"]

        range_data = torch.load(per_layer_range_path, weights_only=False)
        per_layer_range = range_data["per_layer_range"]

        # Filter to requested capping layers
        filtered_range = {
            l: per_layer_range[l] for l in capping_layers if l in per_layer_range
        }
        layer_thresholds = compute_thresholds_at_fraction(
            filtered_range,
            fraction,
            ceiling_from_hi=ceiling_from_hi,
        )

        return cls(model, axis, layer_thresholds, mode=mode)

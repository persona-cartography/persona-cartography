"""Reversible, in-place LoRA modifications for PEFT models.

Limitations
-----------
- Only operates on standard LoRA linear layers (``lora_A`` / ``lora_B``).
  Embedding-based LoRA (``lora_embedding_A`` / ``lora_embedding_B``) and
  convolutional LoRA layers are **not** supported and will be silently
  skipped.  If the adapter targets embedding or output layers, those
  modules will not be modified by any ``BaseLoraModifier`` subclass.
- Layer-index filtering (``layers``, dict ``target_modules``) uses
  ``extract_layer_idx`` from ``model_layer_info``, which recognizes
  several naming conventions (``model.layers.<N>``, ``transformer.h.<N>``,
  ``encoder.layer.<N>``, etc. — see ``LAYER_PATTERNS``).  For models with
  an unsupported naming scheme, pass a custom ``layer_idx_extractor``
  callable to the modifier constructor.
- When ``target_modules`` or ``layers`` restrict which modules are modified,
  ``peft_config.r`` is **not** updated (since the model has mixed ranks).
  This means ``save_pretrained`` will write the original rank to
  ``adapter_config.json``, which may cause shape mismatches on reload.
  Update ``model.peft_config[adapter].r`` manually if needed.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Self
from collections.abc import Iterator
from dataclasses import dataclass, field

from peft import PeftModel
from torch import Tensor, nn

from src.utils.linalg import reduce_lora_rank_efficient
from src.utils.model_layer_info import (
    LAYER_PATTERNS,
    LayerIdxExtractor,
    extract_layer_idx,
)

# ---------------------------------------------------------------------------
# Saved state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _SavedModuleState:
    """Snapshot of a single LoRA module's mutable state.

    PEFT's ``LoraLayer`` stores per-adapter rank and scaling alongside the
    weight tensors.  We capture all four so that ``restore()`` can return
    the module to its exact original configuration.
    """

    weight_A: Tensor  # lora_A weight, shape (r, in_features)
    weight_B: Tensor  # lora_B weight, shape (out_features, r)
    r: int  # per-module LoRA rank (module.r[adapter_name])
    scaling: float  # per-module scaling factor (module.scaling[adapter_name])


@dataclass
class _SavedLoraState:
    """Snapshot of all LoRA modules plus the model-level config rank and alpha.

    ``peft_config_r`` and ``peft_config_lora_alpha`` are written to
    ``adapter_config.json`` by ``save_pretrained`` and determine the per-module
    ``scaling = alpha / r`` (or ``alpha / sqrt(r)`` for rslora) that PEFT
    recomputes on reload.  They must stay consistent with the saved weights.
    """

    modules: dict[str, _SavedModuleState] = field(default_factory=dict)
    peft_config_r: int = 0
    peft_config_lora_alpha: int | float = 0


# ---------------------------------------------------------------------------
# Public utils
# ---------------------------------------------------------------------------


def set_active_adapters(model: PeftModel, adapter_names: str | list[str]) -> None:
    """Activate one or more adapters for the forward pass.

    ``PeftModel.set_adapter`` only accepts a single adapter name.  This
    helper routes through ``model.base_model.set_adapter`` which accepts
    both a string and a list, enabling multi-adapter inference.

    Parameters
    ----------
    model:
        A PEFT model.
    adapter_names:
        A single adapter name or a list of adapter names to activate.
    """
    model.base_model.set_adapter(adapter_names)


def get_active_adapters(model: PeftModel) -> list[str]:
    """Return the list of currently active adapter names."""
    return list(model.base_model.active_adapters)


# ---------------------------------------------------------------------------
# Internal utils
# ---------------------------------------------------------------------------


def _snapshot_adapter(
    model: PeftModel, adapter_name: str, module_names: set[str] | None = None
) -> _SavedLoraState:
    """Snapshot LoRA modules for a given adapter.

    Parameters
    ----------
    model:
        The PEFT model.
    adapter_name:
        Which adapter to snapshot.
    module_names:
        If provided, only snapshot modules whose names are in this set.
        If ``None``, snapshot all modules for the adapter.

    Returns
    -------
    _SavedLoraState
        Snapshot containing cloned weights and metadata.
    """
    saved = _SavedLoraState()
    saved.peft_config_r = model.peft_config[adapter_name].r
    saved.peft_config_lora_alpha = model.peft_config[adapter_name].lora_alpha

    for name, module in _iter_all_lora_modules(model, adapter_name):
        # Skip if module_names filter is provided and this module isn't in it
        if module_names is not None and name not in module_names:
            continue

        saved.modules[name] = _SavedModuleState(
            weight_A=module.lora_A[adapter_name].weight.data.clone(),
            weight_B=module.lora_B[adapter_name].weight.data.clone(),
            r=module.r[adapter_name],
            scaling=module.scaling[adapter_name],
        )
    return saved


def _restore_adapter(
    model: PeftModel, adapter_name: str, saved: _SavedLoraState
) -> None:
    """Restore all LoRA modules for a given adapter from a snapshot.

    Updates ``model`` in-place to match the state in ``saved``.
    """
    model.peft_config[adapter_name].r = saved.peft_config_r
    model.peft_config[adapter_name].lora_alpha = saved.peft_config_lora_alpha

    for name, module in _iter_all_lora_modules(model, adapter_name):
        if name not in saved.modules:
            continue

        ms = saved.modules[name]
        module.r[adapter_name] = ms.r
        module.scaling[adapter_name] = ms.scaling

        module.lora_A[adapter_name].weight = nn.Parameter(
            ms.weight_A.clone().to(
                device=module.lora_A[adapter_name].weight.device,
                dtype=module.lora_A[adapter_name].weight.dtype,
            )
        )
        module.lora_B[adapter_name].weight = nn.Parameter(
            ms.weight_B.clone().to(
                device=module.lora_B[adapter_name].weight.device,
                dtype=module.lora_B[adapter_name].weight.dtype,
            )
        )


def _is_lora_module(module: nn.Module, adapter_name: str) -> bool:
    """Return True if *module* is a LoRA layer with *adapter_name* registered."""
    return (
        hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and adapter_name in module.lora_A
        and adapter_name in module.lora_B
    )


def _iter_all_lora_modules(
    model: PeftModel, adapter_name: str
) -> Iterator[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every LoRA module that has *adapter_name*."""
    for name, module in model.named_modules():
        if _is_lora_module(module, adapter_name):
            yield name, module


def _matches_target_modules(name: str, target_modules: list[str]) -> bool:
    """Check if *name* ends with any of the *target_modules* strings.

    For example, name ``"base_model.model.model.layers.0.self_attn.q_proj"``
    matches ``target_modules=["q_proj"]``.
    """
    return any(name.endswith(t) for t in target_modules)


def _warn_adapter_status(
    model: PeftModel, adapter_name: str, *, stacklevel: int = 3
) -> None:
    """Raise if *adapter_name* is inactive; warn if adapter layers are globally disabled."""
    active = get_active_adapters(model)
    if adapter_name not in active:
        raise ValueError(
            f"Adapter '{adapter_name}' is not currently active "
            f"(active: {active}). Scaling an inactive adapter has no effect on "
            f"the forward pass. Load the adapter under the correct name or call "
            f"set_active_adapters() first."
        )

    # Check if adapter layers are globally disabled (via disable_adapter_layers())
    for _, module in _iter_all_lora_modules(model, adapter_name):
        if getattr(module, "disable_adapters", False):
            warnings.warn(
                f"Adapter layers are currently disabled on the model. "
                f"Modifications to '{adapter_name}' will still be applied "
                f"but will not affect the forward pass until adapter layers "
                f"are re-enabled via model.enable_adapter_layers().",
                stacklevel=stacklevel,
            )
            break


# ---------------------------------------------------------------------------
# Modifier base class & subclasses
# ---------------------------------------------------------------------------


class BaseLoRaModifier(ABC):
    """Base class for reversible, in-place LoRA modifications.

    Captures original LoRA state at init time.  ``apply()`` always restores
    from the original snapshot first so that repeated calls are idempotent
    (they do not compound).

    Parameters
    ----------
    model:
        A PEFT model with at least one LoRA adapter.
    adapter_name:
        Which adapter to modify (e.g. ``"default"``).
    target_modules:
        Controls which modules are affected. Can be:

        - ``None``: All LoRA modules for the adapter
        - ``list[str]``: Modules whose name ends with any of these strings
          (e.g. ``["q_proj", "v_proj"]``)
        - ``dict[int, list[str]]``: Per-layer specification mapping layer
          indices to module name suffixes (e.g. ``{1: ["q_proj"], 6: ["up_proj"]}``).
          When dict is provided, ``layers`` parameter must be ``None``.
    layers:
        If provided, only modules in these layer indices are affected.
        Layer indices are extracted using :func:`extract_layer_idx` (which
        tries all patterns in ``LAYER_PATTERNS``) or a custom
        ``layer_idx_extractor``. ``None`` means all layers. Cannot be used
        with dict-style ``target_modules``.
    layer_idx_extractor:
        Optional custom function ``(name: str) -> int | None`` for
        extracting layer indices from module names. Use this when the
        model's naming convention is not covered by ``LAYER_PATTERNS``.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> None:
        if isinstance(target_modules, dict) and layers is not None:
            raise ValueError(
                "Cannot specify both target_modules as a dict and layers parameter. "
                "When using dict format for target_modules, layer indices are "
                "specified as dict keys."
            )

        # Validate adapter exists
        if adapter_name not in model.peft_config:
            available = list(model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available adapters: {available}"
            )

        self._model = model
        self._adapter_name = adapter_name
        self._target_modules = target_modules
        self._layers = set(layers) if layers is not None else None
        self._layer_idx_extractor = layer_idx_extractor
        self._is_applied = False

        # Validate that filters actually matched something
        self._validate_filters()

        self._saved = self._snapshot_state()

    @property
    def is_applied(self) -> bool:
        return self._is_applied

    def _extract_layer_idx(self, name: str) -> int | None:
        """Extract layer index using custom extractor or the default."""
        return (self._layer_idx_extractor or extract_layer_idx)(name)

    def _validate_filters(self) -> None:
        """Raise ``ValueError`` if active filters matched no LoRA modules.

        Empty dict ``target_modules={}`` is intentionally exempt (means
        "modify nothing").
        """
        if isinstance(self._target_modules, dict) and not self._target_modules:
            return
        if self._target_modules is None and self._layers is None:
            return
        if self._iter_targeted_modules():
            return
        self._raise_filter_error()

    def _raise_filter_error(self) -> None:
        """Raise a ``ValueError`` with cause-specific diagnostics."""
        all_lora_names = [
            name for name, _ in _iter_all_lora_modules(self._model, self._adapter_name)
        ]

        needs_layer_idx = self._layers is not None or isinstance(
            self._target_modules, dict
        )
        if not needs_layer_idx:
            available_suffixes = sorted(
                {name.rsplit(".", 1)[-1] for name in all_lora_names}
            )
            raise ValueError(
                f"target_modules={self._target_modules} matched no LoRA modules. "
                f"Available module name suffixes: {available_suffixes}"
            )

        available_indices = sorted(
            idx
            for name in all_lora_names
            if (idx := self._extract_layer_idx(name)) is not None
        )

        if not available_indices:
            patterns_str = ", ".join(repr(p) for p in LAYER_PATTERNS)
            raise ValueError(
                f"No LoRA modules have an extractable layer index. "
                f"Module names do not match any known pattern "
                f"({patterns_str}). Pass a custom "
                f"layer_idx_extractor to handle this model's naming "
                f"convention."
            )

        if self._layers is not None:
            raise ValueError(
                f"layers={sorted(self._layers)} matched no LoRA modules. "
                f"Available layer indices: {available_indices}"
            )

        raise ValueError(
            f"target_modules dict keys "
            f"{sorted(self._target_modules.keys())} matched no LoRA "
            f"modules. Available layer indices: {available_indices}"
        )

    def _iter_targeted_modules(self) -> list[tuple[str, nn.Module]]:
        """Return ``(name, module)`` pairs matching the adapter, target_modules, and layers filters."""
        result = []

        for name, module in _iter_all_lora_modules(self._model, self._adapter_name):
            # Extract layer index (needed for filtering)
            layer_idx = self._extract_layer_idx(name)

            # Apply filtering based on target_modules type
            if isinstance(self._target_modules, dict):
                # Dict mode: only modules in specified layers with specified targets
                if layer_idx is None or layer_idx not in self._target_modules:
                    continue
                if not _matches_target_modules(name, self._target_modules[layer_idx]):
                    continue
            else:
                # List/None mode: apply target_modules and layers filters independently
                if self._target_modules is not None:
                    if not _matches_target_modules(name, self._target_modules):
                        continue
                if self._layers is not None:
                    if layer_idx is None or layer_idx not in self._layers:
                        continue

            result.append((name, module))
        return result

    def _snapshot_state(self) -> _SavedLoraState:
        """Snapshot the current state of filtered modules."""
        module_names = {name for name, _ in self._iter_targeted_modules()}
        return _snapshot_adapter(self._model, self._adapter_name, module_names)

    def apply(self) -> Self:
        """Restore to original state, then apply the modification.

        Safe to call multiple times — always starts from the original snapshot.

        Note: the base modifier does NOT require the adapter to be active —
        structural edits (rank reduction, layer zeroing of weights) are valid
        on an inactive adapter (they change stored weights regardless of which
        adapter the forward pass uses). Only :class:`LoRaScaling`, whose effect
        is forward-pass-only, guards on activeness (see its ``apply``).
        """
        self.restore()
        self._apply_to_modules()
        self._is_applied = True
        return self

    def _warn_if_inactive(self) -> None:
        """Raise if the target adapter is inactive; warn if globally disabled."""
        _warn_adapter_status(self._model, self._adapter_name, stacklevel=3)

    def restore(self) -> Self:
        """Restore model to the state it was in when this modifier was created."""
        _restore_adapter(self._model, self._adapter_name, self._saved)
        self._is_applied = False
        return self

    @abstractmethod
    def _apply_to_modules(self) -> None:
        """Subclass implements the actual modification logic."""
        ...

    @property
    def _modifies_all_modules(self) -> bool:
        """True if no target_modules or layers filter is active."""
        return self._target_modules is None and self._layers is None


class LoRaRankReducer(BaseLoRaModifier):
    """Reduces LoRA rank via truncated-SVD approximation.

    After ``apply()``, each affected LoRA module's ``lora_A`` has shape
    ``(new_rank, in_features)`` and ``lora_B`` has shape
    ``(out_features, new_rank)``.  The product ``new_B @ new_A`` approximates
    the original ``B @ A``.

    ``lora_alpha`` is rescaled so that the effective per-module scaling
    (``alpha / r``, or ``alpha / sqrt(r)`` for rslora) is preserved after
    reduction — otherwise PEFT would recompute a ``new_r``-sized scaling on
    reload, amplifying the update by ``orig_r / new_rank``.  ``peft_config.r``,
    ``peft_config.lora_alpha``, and per-module ``module.r`` / ``module.scaling``
    are updated only when all modules are affected (no ``target_modules`` or
    ``layers`` filter); otherwise ``peft_config`` is left unchanged and a
    warning is emitted.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        new_rank: int,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> None:
        super().__init__(
            model,
            adapter_name,
            target_modules=target_modules,
            layers=layers,
            layer_idx_extractor=layer_idx_extractor,
        )
        if new_rank < 1:
            raise ValueError(f"new_rank must be >= 1, got {new_rank}")
        self._new_rank = new_rank

    def _apply_to_modules(self) -> None:
        adapter = self._adapter_name
        new_rank = self._new_rank
        cfg = self._model.peft_config[adapter]
        orig_rank = cfg.r
        orig_alpha = cfg.lora_alpha
        use_rslora = getattr(cfg, "use_rslora", False)

        # Preserve scaling = alpha / r (or alpha / sqrt(r) for rslora) across
        # reduction so that save → reload reproduces the original effective
        # update magnitude.
        if use_rslora:
            new_alpha_exact = orig_alpha * math.sqrt(new_rank / orig_rank)
        else:
            new_alpha_exact = orig_alpha * new_rank / orig_rank
        new_alpha_int = int(round(new_alpha_exact))
        if not math.isclose(new_alpha_int, new_alpha_exact, rel_tol=1e-9, abs_tol=1e-9):
            warnings.warn(
                f"lora_alpha rescale from {orig_alpha} to {new_alpha_exact} is "
                f"not an integer; rounding to {new_alpha_int}. Effective "
                f"scaling will drift by "
                f"{(new_alpha_int / new_alpha_exact - 1.0) * 100:.4f}%.",
                stacklevel=2,
            )
        new_alpha: int | float = new_alpha_int if new_alpha_int > 0 else new_alpha_exact

        if self._modifies_all_modules:
            cfg.r = new_rank
            cfg.lora_alpha = new_alpha
        else:
            warnings.warn(
                "target_modules or layers filter is active — peft_config.r "
                "and peft_config.lora_alpha are not updated.  The model will "
                "have mixed ranks across modules, which may cause issues with "
                "save_pretrained / from_pretrained.",
                stacklevel=2,
            )

        if use_rslora:
            new_scaling = float(new_alpha) / math.sqrt(new_rank)
        else:
            new_scaling = float(new_alpha) / new_rank

        for _, module in self._iter_targeted_modules():
            A = module.lora_A[adapter].weight
            B = module.lora_B[adapter].weight

            new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank=new_rank)
            module.lora_A[adapter].weight = nn.Parameter(new_A)
            module.lora_B[adapter].weight = nn.Parameter(new_B)
            module.r[adapter] = new_rank
            module.scaling[adapter] = new_scaling


class LoRaScaling(BaseLoRaModifier):
    """Scales the LoRA contribution by modifying ``module.scaling[adapter]``.

    The scaling factor is applied multiplicatively on top of the existing
    scaling value: ``new_scaling = original_scaling * scale_factor``.  This
    means ``scale_factor=1.0`` is a no-op, ``0.0`` disables the adapter,
    and negative values invert the LoRA direction.

    Use ``target_modules`` and ``layers`` to apply scaling to specific
    matrices or layers only.  For multi-adapter scaling, create one
    ``LoraScaling`` instance per adapter.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        scale_factor: float,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> None:
        super().__init__(
            model,
            adapter_name,
            target_modules=target_modules,
            layers=layers,
            layer_idx_extractor=layer_idx_extractor,
        )
        self._scale_factor = scale_factor

    def apply(self) -> Self:
        """Scale the adapter, after guarding that it is active.

        Scaling only affects the forward pass via the *active* adapter, so
        scaling an inactive one is almost always a mistake (e.g. loading an
        adapter under one name and scaling it under another) — that raises.
        Inside a :class:`LoRaPipeline`, the pipeline activates all of its
        adapters first, so multi-adapter scaling works.
        """
        self._warn_if_inactive()
        return super().apply()

    def _apply_to_modules(self) -> None:
        adapter = self._adapter_name

        for _, module in self._iter_targeted_modules():
            # The saved snapshot has the original scaling; after
            # _restore_from() in apply(), module.scaling is back to original.
            # We multiply by scale_factor to get the desired new scaling.
            module.scaling[adapter] = module.scaling[adapter] * self._scale_factor


class LoRaAdapterZeroing(LoRaScaling):
    """Zeros out the LoRA contribution for specific layers and modules.

    This is a convenience wrapper around ``LoraScaling`` that sets the
    scaling factor to 0.0 for the targeted modules.
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_name: str,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> None:
        super().__init__(
            model=model,
            adapter_name=adapter_name,
            scale_factor=0.0,
            target_modules=target_modules,
            layers=layers,
            layer_idx_extractor=layer_idx_extractor,
        )


# ---------------------------------------------------------------------------
# Pipeline for composing multiple modifications
# ---------------------------------------------------------------------------


class LoRaPipeline:
    """Applies multiple LoRA modifications in sequence.

    A pipeline can apply changes to one or more adapters. Each step is applied
    in order, with modifications compounding. The pipeline snapshots the
    original state at init, and ``restore()`` returns to that state.

    Unlike individual modifiers, a pipeline can modify different adapters in
    different steps, enabling complex multi-adapter transformations.

    Parameters
    ----------
    model:
        A PEFT model with LoRA adapters.
    steps:
        List of ``(modifier_class, adapter_name, kwargs)`` tuples defining
        the pipeline. The kwargs should not include ``model`` or ``adapter_name``
        (those are provided automatically).

    Examples
    --------
    Apply multiple modifications to a single adapter::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "default", {"new_rank": 4, "layers": [0, 1]}),
                (LoraScaling, "default", {"scale_factor": 0.5}),
            ]
        )
        pipeline.apply()  # Rank reduced first, then scaled
        pipeline.restore()  # Returns to original state

    Modify different adapters in a single pipeline::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "adapter_a", {"new_rank": 2}),
                (LoraScaling, "adapter_b", {"scale_factor": 0.1}),
                (LoraLayerZeroing, "adapter_a", {"layers": [0]}),
            ]
        )

    Complex per-module transformations::

        pipeline = LoraPipeline(
            model,
            steps=[
                (LoraRankReducer, "default", {
                    "new_rank": 2,
                    "target_modules": {0: ["q_proj"], 1: ["v_proj"]}
                }),
                (LoraScaling, "default", {
                    "scale_factor": 2.0,
                    "target_modules": ["up_proj", "down_proj"]
                }),
            ]
        )
    """

    def __init__(
        self,
        model: PeftModel,
        steps: list[tuple[type[BaseLoRaModifier], str, dict]],
    ) -> None:
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        # Validate steps
        for modifier_class, adapter_name, kwargs in steps:
            if not issubclass(modifier_class, BaseLoRaModifier):
                raise TypeError(
                    f"{modifier_class} is not a subclass of BaseLoraModifier"
                )
            if "model" in kwargs or "adapter_name" in kwargs:
                raise ValueError(
                    "Step kwargs should not include 'model' or 'adapter_name' "
                    "(these are provided automatically)"
                )
            if adapter_name not in model.peft_config:
                available = list(model.peft_config.keys())
                raise ValueError(
                    f"Adapter '{adapter_name}' not found. "
                    f"Available adapters: {available}"
                )

        self._model = model
        self._steps = steps
        self._is_applied = False

        # Snapshot each unique adapter once at init
        adapters = {adapter_name for _, adapter_name, _ in steps}
        self._snapshots: dict[str, _SavedLoraState] = {
            adapter_name: _snapshot_adapter(model, adapter_name)
            for adapter_name in adapters
        }

    @property
    def is_applied(self) -> bool:
        return self._is_applied

    def apply(self) -> Self:
        """Apply all pipeline steps in sequence.

        For steps affecting the same adapter, modifications compound. The
        pipeline first restores all affected adapters to their original state,
        then applies each modification in order.
        """
        # Restore all affected adapters to original state
        self.restore()

        # Activate every adapter this pipeline touches before applying any
        # step. PEFT keeps a single adapter active by default, so scaling
        # steps on the pipeline's other adapters would otherwise be rejected
        # by LoRaScaling's inactive-adapter guard.
        set_active_adapters(
            self._model,
            list({adapter_name for _, adapter_name, _ in self._steps}),
        )

        # Apply each step in order.  Each modifier snapshots the current
        # state at creation time, so apply() restores to that snapshot
        # (a no-op) then applies the modification on top.
        for modifier_class, adapter_name, kwargs in self._steps:
            modifier_class(self._model, adapter_name, **kwargs).apply()

        self._is_applied = True
        return self

    def restore(self) -> Self:
        """Restore all adapters to their original states."""
        for adapter_name, saved_state in self._snapshots.items():
            _restore_adapter(self._model, adapter_name, saved_state)

        self._is_applied = False
        return self

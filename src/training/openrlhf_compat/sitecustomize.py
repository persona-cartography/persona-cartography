"""Runtime compatibility patches for OpenRLHF subprocesses.

This module is auto-imported by Python when present on ``PYTHONPATH``.
We use it to patch upstream OpenRLHF behavior without editing the
installed package inside transient ``uv`` environments.
"""

from __future__ import annotations

from typing import Any


def _patch_openrlhf_optimizer_grouping() -> None:
    """Keep optimizer and scheduler param-group counts aligned.

    OpenRLHF always creates two optimizer groups (decay / no-decay). When the
    requested weight decay is zero, those groups end up with identical
    hyperparameters and can be collapsed by downstream DeepSpeed/FusedAdam
    handling. The Hugging Face scheduler is created before that collapse, so it
    later tries to step two LR values against a one-group optimizer and raises:

        ValueError: zip() argument 2 is longer than argument 1

    For zero-weight-decay runs, we collapse the groups up front so the
    scheduler is initialized against the same group structure the optimizer
    keeps during training.
    """

    try:
        from openrlhf.utils.deepspeed import deepspeed_utils
        from openrlhf.utils.deepspeed import deepspeed as deepspeed_module
    except Exception:
        return

    original = getattr(deepspeed_utils, "get_optimizer_grouped_parameters", None)
    if original is None or getattr(original, "_persona_shattering_patched", False):
        return

    default_no_decay = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"]

    def patched_get_optimizer_grouped_parameters(
        model: Any,
        weight_decay: float,
        no_decay_name_list: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        groups = original(
            model,
            weight_decay,
            no_decay_name_list=default_no_decay if no_decay_name_list is None else no_decay_name_list,
        )
        groups = [group for group in groups if group.get("params")]

        if abs(float(weight_decay)) > 1e-12:
            return groups

        merged_params = []
        for group in groups:
            merged_params.extend(group["params"])
        return [{"params": merged_params, "weight_decay": 0.0}]

    patched_get_optimizer_grouped_parameters._persona_shattering_patched = True  # type: ignore[attr-defined]
    deepspeed_utils.get_optimizer_grouped_parameters = patched_get_optimizer_grouped_parameters
    deepspeed_module.get_optimizer_grouped_parameters = patched_get_optimizer_grouped_parameters


_patch_openrlhf_optimizer_grouping()

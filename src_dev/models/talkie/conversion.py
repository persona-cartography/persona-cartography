"""Reference checkpoint → HF state-dict conversion.

The reference ``talkie.model.load_checkpoint`` handles three layouts and
strips ``torch.compile``'s ``_orig_mod.`` prefix. We do the same and then
return the dict ready to feed into ``TalkieForCausalLM.load_state_dict``.

Key names in the reference checkpoint already match the wrapper's
``named_parameters()`` for the inner ``TalkieModel`` (``embed.weight``,
``blocks.<L>.<...>``). Only the two global params need to be re-rooted to
the top-level wrapper:

  embed.weight, blocks.* → model.embed.weight, model.blocks.*
  lm_head, lm_head_gain.w_g → lm_head, lm_head_gain.w_g (unchanged)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


_BLOCK_OR_EMBED_PREFIXES = ("embed.", "blocks.")


def load_reference_state_dict(pt_path: str | Path) -> dict[str, torch.Tensor]:
    """Load ``rl-refined.pt`` into a flat ``{name: Tensor}`` dict.

    Mirrors ``talkie.model.load_checkpoint:275-282``.
    """
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def state_dict_to_hf(reference_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Re-root the reference state-dict for ``TalkieForCausalLM``.

    Block + embedding params live under ``model.``. The reference's
    ``lm_head`` (an ``nn.Parameter``) plus ``lm_head_gain.w_g`` (scalar
    WeightGain) are folded into a single ``lm_head.weight`` tensor so the
    HF wrapper can use a standard ``nn.Linear`` for the head and vLLM's
    ``ParallelLMHead`` can load it as a normal linear weight.

    The math is exact: the reference computes
    ``F.linear(x, w_g * lm_head) == x @ (w_g * lm_head).T``; precomputing
    ``lm_head.weight = w_g * lm_head`` gives the same result.
    """
    out: dict[str, torch.Tensor] = {}
    lm_head: Optional[torch.Tensor] = None
    lm_head_gain: Optional[torch.Tensor] = None
    for k, v in reference_sd.items():
        if k == "lm_head":
            lm_head = v
            continue
        if k == "lm_head_gain.w_g":
            lm_head_gain = v
            continue
        if k.startswith(_BLOCK_OR_EMBED_PREFIXES):
            out[f"model.{k}"] = v
        else:
            out[k] = v

    if lm_head is None:
        raise KeyError("Reference state dict is missing 'lm_head' tensor.")
    if lm_head_gain is None:
        raise KeyError("Reference state dict is missing 'lm_head_gain.w_g' tensor.")
    # Cast to lm_head's dtype then scale; preserves bf16 throughout.
    scale = lm_head_gain.to(lm_head.dtype).item()
    out["lm_head.weight"] = lm_head * scale
    return out


__all__ = ["load_reference_state_dict", "state_dict_to_hf"]

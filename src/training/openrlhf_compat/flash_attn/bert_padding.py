"""Pure-torch equivalents of the ``flash_attn.bert_padding`` helpers OpenRLHF imports.

``openrlhf.models.ring_attn_utils`` does
``from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input``
at module load. ``rearrange`` is just re-exported from einops (as the real
flash-attn does). The pad/unpad helpers pack and unpack variable-length sequences;
they are only used on the sequence-parallel (ring-attention) path, which the OCT
DPO/SFT runs do not take — but they are implemented correctly here so the shim is
functional rather than merely import-satisfying.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat  # re-export: OpenRLHF imports `rearrange` from here

__all__ = ["index_first_axis", "pad_input", "unpad_input", "rearrange", "repeat"]


def index_first_axis(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows ``indices`` along the first axis (forward-equivalent to flash-attn)."""
    return input[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Drop padding tokens.

    Returns the 5-tuple ``(packed, indices, cu_seqlens, max_seqlen_in_batch,
    None)`` — matching the contract OpenRLHF's ring-attention code unpacks (the
    trailing element is flash-attn's ``seqused``, unused here).
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat = rearrange(hidden_states, "b s ... -> (b s) ...")
    return flat[indices], indices, cu_seqlens, max_seqlen_in_batch, None


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int):
    """Re-scatter packed tokens back into a padded ``(batch, seqlen, ...)`` tensor."""
    trailing = hidden_states.shape[1:]
    output = hidden_states.new_zeros((batch * seqlen, *trailing))
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

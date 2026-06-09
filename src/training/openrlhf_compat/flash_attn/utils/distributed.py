"""Pure-torch stand-in for ``flash_attn.utils.distributed.all_gather``.

``openrlhf.models.ring_attn_utils`` imports this at module load. It is only used
on the sequence-parallel path (which the OCT runs don't take); the single-GPU /
no-distributed fallback returns the input unchanged.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["all_gather"]


def all_gather(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Concatenate ``tensor`` gathered from every rank along axis 0.

    Returns the input unchanged when torch.distributed is unavailable /
    uninitialised or the world size is 1 — the single-GPU case this shim targets.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return tensor
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor.contiguous(), group=group)
    return torch.cat(tensors, dim=0)

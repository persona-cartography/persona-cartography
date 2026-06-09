"""Minimal pure-PyTorch stand-in for ``flash_attn``.

Placed on ``PYTHONPATH`` (via :func:`src.training.oct_adapter._run_openrlhf_training`)
only for the OpenRLHF/DeepSpeed training subprocess, for environments where the
real CUDA ``flash-attn`` isn't installed (it's slow/fragile to build, so the OCT
stack is set up with ``--no-deps`` and we run with ``--attn_implementation eager``).

OpenRLHF imports a few names from ``flash_attn`` at module load —
``openrlhf.models.ring_attn_utils`` pulls ``flash_attn.bert_padding`` (4 names)
and ``flash_attn.utils.distributed.all_gather`` at the top level — so those
imports must resolve even though, with eager attention and no sequence
parallelism, the imported functions are never actually called. This package
provides correct pure-torch implementations so the imports succeed (and work if
ever invoked) without the real flash-attn.

Deliberately NOT provided: ``flash_attn.ops.triton.cross_entropy`` — OpenRLHF
imports it under ``try/except ImportError`` (``openrlhf.models.utils``) and falls
back to a pure-torch path, so we let that import fail. This package also installs
no distribution metadata, so ``transformers.utils.is_flash_attn_2_available()``
(which checks ``importlib.metadata``) stays ``False`` and won't auto-select FA2.
"""

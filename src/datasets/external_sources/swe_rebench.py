"""Adapter for ``nebius/SWE-rebench-openhands-trajectories``.

Coding-agent rollouts produced by ``Qwen3-Coder-480B-A35B-Instruct``
driving the OpenHands scaffold on real GitHub-issue resolution tasks.
~67k trajectories (32k resolved), median ~37k tokens (~44 turns), with a
long right tail up to ~100k tokens. Only ~36% fit in a 32k context
window under the Qwen2.5 chat template — use ``max_context_tokens >=
65536`` to keep most of the distribution.

Note on serving: Qwen3-Coder-480B is an MoE (35B active / 480B total)
and typically needs an 8×H100 or 4×H200 node at BF16, or ~4×H100 at
AWQ-4bit. For local experiments on smaller hardware you can either:

  - Administer with a smaller Qwen3 sibling (e.g. Qwen3-8B) and accept
    the cross-model mismatch (set ``QUESTIONNAIRE_MODEL_OVERRIDE``); or
  - Use a hosted inference provider (Together / Fireworks / DeepInfra)
    for the questionnaire pass.

The adapter always reports the default assistant model as
Qwen3-Coder-480B for provenance; the *administering* model for Stage 2
is whatever the preset chooses.
"""

from __future__ import annotations

from typing import Any, Iterator

from src.datasets.external_sources.base import (
    canonicalise_messages,
    register_adapter,
)

ADAPTER_NAME = "swe_rebench"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
SOURCE_HF_REPO = "nebius/SWE-rebench-openhands-trajectories"


def _iter_raw(filter_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream SWE-rebench rows, yielding canonical row dicts.

    Filter keys understood:
        min_assistant_turns (int): drop rows with fewer assistant turns.
        resolved_only (bool): keep only trajectories that solved the
            task (``resolved == 1``). Default False.
    """
    from datasets import load_dataset

    min_turns = int(filter_config.get("min_assistant_turns", 0) or 0)
    resolved_only = bool(filter_config.get("resolved_only", False))

    ds = load_dataset(SOURCE_HF_REPO, split="train", streaming=True)
    for i, row in enumerate(ds):
        if resolved_only and int(row.get("resolved", 0) or 0) != 1:
            continue
        raw_messages = row.get("trajectory") or []
        msgs = canonicalise_messages(raw_messages)
        if not msgs:
            continue
        if min_turns:
            n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
            if n_assistant < min_turns:
                continue
        trajectory_id = row.get("trajectory_id") or f"row{i}"
        yield {
            "sample_id": f"{ADAPTER_NAME}::{trajectory_id}",
            "messages": msgs,
            "assistant_model": None,  # use preset default (Qwen3-Coder-480B)
            "source_info": {
                "source": ADAPTER_NAME,
                "source_hf_repo": SOURCE_HF_REPO,
                "trajectory_id": trajectory_id,
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "resolved": int(row.get("resolved", 0) or 0),
                "row_index": i,
            },
        }


register_adapter(
    name=ADAPTER_NAME,
    default_assistant_model=DEFAULT_MODEL,
    default_assistant_provider="vllm",
    notes=(
        "OpenHands coding-agent trajectories from Qwen3-Coder-480B. "
        "~67k conversations, median ~37k tokens, ~44 assistant turns. "
        "Only ~36% fit in 32k ctx; use >=64k context window."
    ),
    iter_raw=_iter_raw,
)

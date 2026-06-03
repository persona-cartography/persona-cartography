"""Adapter for ``Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k``.

Agent rollouts produced by Qwen3-8B on SWE-bench-style tasks.
66k conversations, median ~13k tokens, 24 assistant turns. ~98% fit in a
32k context window under the Qwen2.5 chat template. Good size-match
candidate for our Llama-3.1-8B baseline.
"""

from __future__ import annotations

from typing import Any, Iterator

from src.datasets.external_sources.base import (
    canonicalise_messages,
    register_adapter,
)

ADAPTER_NAME = "kwai_swe_smith"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
SOURCE_HF_REPO = "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"


def _iter_raw(filter_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream Kwai-Klear rows, yielding canonical row dicts.

    Filter keys understood:
        min_assistant_turns (int): drop rows whose assistant turn count is
            below this threshold before sampling.
    """
    # Deferred import — ``datasets`` is heavy and may not be installed in
    # environments that only use other parts of ``src.datasets``.
    from datasets import load_dataset

    min_turns = int(filter_config.get("min_assistant_turns", 0) or 0)

    ds = load_dataset(SOURCE_HF_REPO, split="train", streaming=True)
    for i, row in enumerate(ds):
        raw_messages = row.get("messages") or []
        msgs = canonicalise_messages(raw_messages)
        if not msgs:
            continue
        if min_turns:
            n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
            if n_assistant < min_turns:
                continue
        instance_id = row.get("instance_id") or f"row{i}"
        yield {
            "sample_id": f"{ADAPTER_NAME}::{instance_id}",
            "messages": msgs,
            "assistant_model": None,  # use preset default (Qwen3-8B)
            "source_info": {
                "source": ADAPTER_NAME,
                "source_hf_repo": SOURCE_HF_REPO,
                "instance_id": instance_id,
                "row_index": i,
            },
        }


register_adapter(
    name=ADAPTER_NAME,
    default_assistant_model=DEFAULT_MODEL,
    default_assistant_provider="vllm",
    notes=(
        "SWE-bench-style agent rollouts from Qwen3-8B. 66k conversations, "
        "median ~13k tokens, ~24 assistant turns. 98% fit in 32k ctx."
    ),
    iter_raw=_iter_raw,
)

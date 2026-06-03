"""Adapter for ``HannahRoseKirk/prism-alignment`` (PRISM).

PRISM is a large-scale values-elicitation dataset where real users engaged
LLMs on contentious, personal, and values-laden topics (inequality, AI
ethics, relationships, politics, religion, creativity, etc.). 8,011
conversation trees × ~15 total turns on average. The task framing of the
original data collection (values alignment, disagreement, controversial
topics) makes it the most **persona-eliciting** of our candidate datasets
by intent.

Dataset structure
-----------------

Each row is one conversation tree. ``conversation_history`` contains all
messages, but at each assistant turn the user was presented with
responses from **multiple models simultaneously** (``within_turn_id=0,
1, 2, ...``) and picked one (``if_chosen=True``). The "thread the user
actually engaged with" is therefore:

    user turn 0 → chosen assistant turn 0 → user turn 1 → chosen
    assistant turn 1 → ...

This adapter reconstructs exactly that thread. Non-chosen alternatives
are discarded (they were never seen by the user within the conversation
flow — though they are retained in ``source_info`` for provenance).

Open-source assistant models present
------------------------------------

From a sample of 500 conversations, chosen-assistant model frequencies
(roughly proportional to full-dataset distribution) include:

    Zephyr-7B-beta              ~126   HuggingFaceH4/zephyr-7b-beta
    Llama-2-7B-Chat             ~119   meta-llama/Llama-2-7b-chat-hf
    Llama-2-13B-Chat             ~93   meta-llama/Llama-2-13b-chat-hf
    Mistral-7B-Instruct-v0.1     ~78   mistralai/Mistral-7B-Instruct-v0.1
    Guanaco-33B                  ~78   timdettmers/guanaco-33b-merged
    Llama-2-70B-Chat             ~67   meta-llama/Llama-2-70b-chat-hf
    Falcon-7B-Instruct           ~55   tiiuae/falcon-7b-instruct
    OpenAssistant pythia-12b     ~55   OpenAssistant/oasst-sft-4-pythia-12b

Closed-source (disqualified for same-model administration): Anthropic
Claude variants, OpenAI GPT variants, Google chat-bison, Cohere Command
variants, Aleph Luminous. These are silently filtered out by the default
open-model policy — override via ``filter_config["model_allowlist"]`` to
keep specific closed models for research purposes.

Length distribution (under Qwen2.5 chat template, 500-conversation sample)
--------------------------------------------------------------------------

    turns p50 / p90:      8 / 14 assistant turns
    tokens p50 / p90:     852 / 1,927 tokens
    max tokens observed:  ~5k

Comfortably fits in a 4k-context Llama-2 window for nearly all rows;
fits trivially in 32k Mistral / Zephyr.

Filter keys understood
----------------------

    model_allowlist (list[str]):
        Full ``provider/model`` strings (as seen in PRISM's raw fields) or
        just the bare model names. When set, only keep conversations whose
        *chosen-assistant turns all come from models in the allowlist*.
        Matches via substring (case-insensitive) so
        ``"Llama-2-7b-chat"`` matches
        ``"huggingface_api/meta-llama/Llama-2-7b-chat-hf"``. When unset, a
        default "any HuggingFace-served open model" policy applies (see
        ``_DEFAULT_OPEN_POLICY``).

    min_assistant_turns (int):
        Standard filter, applied after thread reconstruction.
"""

from __future__ import annotations

from typing import Any, Iterator

from src.datasets.external_sources.base import (
    canonicalise_messages,
    register_adapter,
)

ADAPTER_NAME = "prism_open"
# PRISM is multi-model; this is a provenance placeholder. Every preset
# declares its concrete assistant_model (which must match one of the
# models in its filter_config["model_allowlist"]).
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SOURCE_HF_REPO = "HannahRoseKirk/prism-alignment"


# Default "any open-weights LLM" policy: conversations where EVERY chosen
# assistant turn came from an OSS model. Substrings matched case-insensitively.
_DEFAULT_OPEN_POLICY = (
    "llama", "mistral", "zephyr", "falcon", "guanaco", "pythia",
    "oasst", "vicuna", "gemma", "qwen", "yi",
)


def _row_matches_allowlist(
    chosen_models: list[str],
    allowlist: list[str] | None,
) -> bool:
    """Return True if every chosen-assistant model matches the allowlist
    (substring match, case-insensitive)."""
    if not chosen_models:
        return False
    if allowlist is None:
        policy = _DEFAULT_OPEN_POLICY
    else:
        policy = tuple(s.lower() for s in allowlist)
    return all(
        any(p in m.lower() for p in policy)
        for m in chosen_models
    )


def _iter_raw(filter_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    allowlist = filter_config.get("model_allowlist")
    min_turns = int(filter_config.get("min_assistant_turns", 0) or 0)

    ds = load_dataset(
        SOURCE_HF_REPO, "conversations", split="train", streaming=True,
    )
    for i, row in enumerate(ds):
        hist = row.get("conversation_history") or []
        if not hist:
            continue

        # Reconstruct the "thread" the user actually saw: every user msg
        # plus the chosen assistant reply at each turn. Non-chosen model
        # alternatives are dropped.
        thread_raw: list[dict[str, str]] = []
        chosen_models: list[str] = []
        for m in hist:
            role = m.get("role")
            if role == "user":
                thread_raw.append({"role": "user",
                                   "content": m.get("content") or ""})
            elif role == "model" and m.get("if_chosen"):
                thread_raw.append({"role": "assistant",
                                   "content": m.get("content") or ""})
                mp = m.get("model_provider") or ""
                mn = m.get("model_name") or ""
                chosen_models.append(f"{mp}/{mn}" if mp else mn)

        if not _row_matches_allowlist(chosen_models, allowlist):
            continue

        msgs = canonicalise_messages(thread_raw)
        if not msgs:
            continue
        if min_turns:
            n_assistant = sum(1 for mm in msgs if mm["role"] == "assistant")
            if n_assistant < min_turns:
                continue

        conversation_id = row.get("conversation_id") or f"row{i}"
        yield {
            "sample_id": f"{ADAPTER_NAME}::{conversation_id}",
            "messages": msgs,
            "assistant_model": chosen_models[0] if chosen_models else None,
            "source_info": {
                "source": ADAPTER_NAME,
                "source_hf_repo": SOURCE_HF_REPO,
                "conversation_id": conversation_id,
                "conversation_type": row.get("conversation_type"),
                "chosen_models": chosen_models,
                "included_in_balanced_subset": row.get(
                    "included_in_balanced_subset"
                ),
                "row_index": i,
            },
        }


register_adapter(
    name=ADAPTER_NAME,
    default_assistant_model=DEFAULT_MODEL,
    default_assistant_provider="vllm",
    notes=(
        "PRISM values-alignment dialogue. ~8k conversation trees, median "
        "~8 assistant turns, ~850 tokens. Use model_allowlist to pin to a "
        "single OSS model for same-model administration."
    ),
    iter_raw=_iter_raw,
)

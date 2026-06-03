"""Adapter for ``lmsys/lmsys-chat-1m`` (LMSYS-Chat-1M).

1 million real-user conversations with 25 chatbot backends (September
2023 – April 2024), collected via the Chatbot Arena / LMSYS web UI.
Mixed assistant models: some are open-source (Vicuna, Llama-2, Koala,
MPT, WizardLM, ChatGLM, Guanaco, ...); others are closed (GPT-3.5/4,
Claude, PaLM). For same-model administration, **pin to a single
open-source model per preset** via ``filter_config["model_allowlist"]``.

This is the best "persona-rich real-user chat with open assistants"
candidate in our shortlist. 154 languages; very diverse topics.

Access
------

**Gated dataset.** The HF repo requires user agreement to LMSYS's terms
of use. Set ``HF_TOKEN`` in ``.env`` and accept the terms at
https://huggingface.co/datasets/lmsys/lmsys-chat-1m. The adapter surfaces
a clear error if access is missing.

Dataset schema (per row)
------------------------

    conversation_id : str
    model           : str   -- the assistant model (see list below)
    conversation    : list[{role, content}]  -- full multi-turn chat
    turn            : int   -- assistant-turn count
    language        : str   -- ISO-ish language tag
    openai_moderation : dict   -- per-turn moderation flags
    redacted        : bool  -- PII redaction applied

Open-source models + HuggingFace repo mapping
----------------------------------------------

Counts below are chosen-row tallies from a 100,000-row scan of the
LMSYS-Chat-1M stream (streaming throughput ~2800 rows/s on an ordinary
connection). The distinct-model set of exactly 25 is stable across
samples of this size, so the table is complete. Extrapolated
full-dataset counts are ~10x these numbers.

``model`` string                HF repo                                              Ctx   Seen(100k)  Share
─────────────────────────────   ─────────────────────────────────────────────────   ───   ──────────  ──────
``vicuna-13b``                  lmsys/vicuna-13b-v1.5                                4k    48,939      49%
``koala-13b``                   TheBloke/koala-13B-HF (delta-merged)                 2k     8,190       8%
``alpaca-13b``                  (Stanford Alpaca ft of Llama-1; not on HF)           2k     6,083       6%
``chatglm-6b``                  THUDM/chatglm-6b                                     2k     3,630       4%
``llama-13b``                   huggyllama/llama-13b (base, not chat)                2k     3,208       3%
``llama-2-13b-chat``            meta-llama/Llama-2-13b-chat-hf                       4k     3,081       3%
``vicuna-33b``                  lmsys/vicuna-33b-v1.3                                2k     3,076       3%
``fastchat-t5-3b``              lmsys/fastchat-t5-3b-v1.0                            2k     2,672       3%
``oasst-pythia-12b``            OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5       2k     2,626       3%
``dolly-v2-12b``                databricks/dolly-v2-12b                              2k     2,328       2%
``wizardlm-13b``                WizardLMTeam/WizardLM-13B-V1.2                       4k     1,725       2%
``vicuna-7b``                   lmsys/vicuna-7b-v1.5                                 4k     1,642       2%
``RWKV-4-Raven-14B``            BlinkDL/rwkv-4-raven                                 8k     1,597       2%
``mpt-7b-chat``                 mosaicml/mpt-7b-chat                                 2k     1,485       1%
``guanaco-33b``                 timdettmers/guanaco-33b-merged                       4k     1,355       1%
``stablelm-tuned-alpha-7b``     stabilityai/stablelm-tuned-alpha-7b                  4k     1,170       1%
``mpt-30b-chat``                mosaicml/mpt-30b-chat                                8k       894       1%
``gpt4all-13b-snoozy``          nomic-ai/gpt4all-13b-snoozy                          2k       774       1%
``llama-2-7b-chat``             meta-llama/Llama-2-7b-chat-hf                        4k       369       <1%

**Confirmed NOT in LMSYS-Chat-1M** (don't bother pinning these — the
adapter will yield zero rows even on a full scan): Mistral-7B-Instruct,
Mistral-7B-Instruct-v0.2/v0.3, Llama-2-70B-Chat, ChatGLM2/ChatGLM3,
Baichuan-13B-Chat, Tulu-30B, Gemma, Qwen. The dataset was collected
Sep 2023 – Apr 2024 but its backend set appears frozen early in that
window.

Closed-source (disqualified): ``gpt-3.5-turbo`` (790), ``gpt-4`` (737),
``palm-2`` (559), ``claude-instant-1`` (554), ``claude-1`` (2,313),
``claude-2`` (203).

Scan cost
---------

Exhaustively streaming 1M rows takes ~6 minutes over a normal link.
``max_scan=None`` (the preset default) exhausts the source; use
``max_scan=100_000`` or so only for fast iteration / smoke tests.
Running with a too-small ``max_scan`` on a rare-model preset can silently
yield 0 rows.

**Context-window gotcha:** Most LMSYS open models have 2-4k native
context. Even LMSYS's short-by-nature conversations (median ~500-700
tokens) occasionally exceed 4k with long turns. The preset's
``max_context_tokens`` (and the Stage-2 context filter) handles this,
but be aware you may drop a non-trivial fraction of rows for 2-4k
models.

Filter keys understood
----------------------

    model_allowlist (list[str]):
        LMSYS ``model`` field values (exact string match). When set,
        only rows whose ``model`` is in the list are kept. Required for
        same-model administration: one preset, one concrete
        ``model_allowlist=["<single_model>"]``.

    min_assistant_turns (int):
        Minimum number of assistant turns. PRISM-like filter applied
        after canonicalisation.

    min_turn (int):
        Keep only rows with ``row["turn"] >= min_turn``. Cheap pre-filter
        on the source field without having to count assistant messages.
        Typical value: 5 (LMSYS-specific — the bulk is single-exchange).

    languages (list[str]):
        Optional ISO-ish language allowlist (e.g. ``["English"]``).
"""

from __future__ import annotations

from typing import Any, Iterator

from src.datasets.external_sources.base import (
    canonicalise_messages,
    register_adapter,
)

ADAPTER_NAME = "lmsys_open"
SOURCE_HF_REPO = "lmsys/lmsys-chat-1m"

# Placeholder default — every real preset must set a concrete assistant
# model and a matching model_allowlist.
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


def _iter_raw(filter_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    allowlist = filter_config.get("model_allowlist")
    allowlist_set = set(allowlist) if allowlist else None
    languages = filter_config.get("languages")
    lang_set = set(languages) if languages else None
    min_turn = int(filter_config.get("min_turn", 0) or 0)
    min_assistant_turns = int(
        filter_config.get("min_assistant_turns", 0) or 0
    )

    try:
        ds = load_dataset(SOURCE_HF_REPO, split="train", streaming=True)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "authenticated" in msg.lower():
            raise RuntimeError(
                "lmsys/lmsys-chat-1m is a gated HF dataset. To use this "
                "adapter: (1) set HF_TOKEN in .env, (2) accept the terms at "
                "https://huggingface.co/datasets/lmsys/lmsys-chat-1m. "
                f"Original error: {e}"
            ) from e
        raise

    for i, row in enumerate(ds):
        if allowlist_set is not None and row.get("model") not in allowlist_set:
            continue
        if lang_set is not None and row.get("language") not in lang_set:
            continue
        if min_turn and int(row.get("turn", 0) or 0) < min_turn:
            continue

        raw_messages = row.get("conversation") or []
        msgs = canonicalise_messages(raw_messages)
        if not msgs:
            continue
        if min_assistant_turns:
            n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
            if n_assistant < min_assistant_turns:
                continue

        conversation_id = row.get("conversation_id") or f"row{i}"
        yield {
            "sample_id": f"{ADAPTER_NAME}::{conversation_id}",
            "messages": msgs,
            "assistant_model": row.get("model"),
            "source_info": {
                "source": ADAPTER_NAME,
                "source_hf_repo": SOURCE_HF_REPO,
                "conversation_id": conversation_id,
                "lmsys_model": row.get("model"),
                "language": row.get("language"),
                "turn": row.get("turn"),
                "redacted": row.get("redacted"),
                "row_index": i,
            },
        }


register_adapter(
    name=ADAPTER_NAME,
    default_assistant_model=DEFAULT_MODEL,
    default_assistant_provider="vllm",
    notes=(
        "LMSYS-Chat-1M real-user chat. 1M conversations, 25 backends "
        "(mixed open/closed). Gated — needs HF auth. Pin one open model "
        "per preset via model_allowlist for same-model administration."
    ),
    iter_raw=_iter_raw,
)

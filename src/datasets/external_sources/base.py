"""Protocol, registry, and shared helpers for external rollout-dataset
adapters.

An "external" dataset is a pre-existing multi-turn conversation corpus
hosted on HuggingFace (e.g. ``Kwai-Klear/SWE-smith-mini_swe_agent_plus-
trajectories-66k``) whose assistant turns were produced by a known
open-source model. Adapters normalise these into a canonical row shape
that the psychometric pipeline can ingest via
``src.datasets.ingest_source_dataset``.

Canonical row shape (emitted by ``iter_raw``)::

    {
        "sample_id": str,          # stable, unique per source row
        "messages": [               # full multi-turn conversation,
            {"role": str, "content": str},   # including assistant turns
            ...
        ],
        "assistant_model": str | None,  # per-row override; None → preset default
        "source_info": dict,        # opaque provenance (original id, source name, ...)
    }

Register adapters by importing ``register_adapter`` from this module.
Adapter modules should self-register on import; the
``src.datasets.external_sources.__init__`` eager-imports each known
adapter to populate the registry at module load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


# Canonical row dict keys, for readability in call sites.
REQUIRED_ROW_KEYS = ("sample_id", "messages", "source_info")


IterRawFn = Callable[[dict[str, Any]], Iterator[dict[str, Any]]]


@dataclass(frozen=True)
class AdapterRegistryEntry:
    """Metadata + iterator callable for one external source."""

    name: str
    default_assistant_model: str
    default_assistant_provider: str
    notes: str
    iter_raw: IterRawFn = field(repr=False)


_REGISTRY: dict[str, AdapterRegistryEntry] = {}


def register_adapter(
    *,
    name: str,
    default_assistant_model: str,
    default_assistant_provider: str,
    notes: str,
    iter_raw: IterRawFn,
) -> AdapterRegistryEntry:
    """Register an external-source adapter.

    Called at module import time by each adapter file; see the base docstring.
    Raises if ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"External adapter {name!r} already registered")
    entry = AdapterRegistryEntry(
        name=name,
        default_assistant_model=default_assistant_model,
        default_assistant_provider=default_assistant_provider,
        notes=notes,
        iter_raw=iter_raw,
    )
    _REGISTRY[name] = entry
    return entry


def get_adapter(name: str) -> AdapterRegistryEntry:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown external adapter {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_adapters() -> list[AdapterRegistryEntry]:
    return list(_REGISTRY.values())


def canonicalise_messages(raw_messages: list[dict]) -> list[dict]:
    """Strip messages to ``{role, content}`` only.

    - Preserves system/user/assistant/tool roles.
    - Coerces non-string content (lists, dicts, tool-call structures) to
      JSON strings so length-count and chat-template application don't
      blow up on tool-use trajectories (e.g. SWE-rebench OpenHands rows).
    - Drops messages with empty/whitespace content after coercion.
    """
    out: list[dict] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        content = m.get("content")
        if content is None:
            tc = m.get("tool_calls")
            if tc:
                content = json.dumps(tc, ensure_ascii=False)
            else:
                continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out

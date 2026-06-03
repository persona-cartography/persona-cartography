"""Content-addressed fingerprint for a cell sweep's non-adapter configuration.

The fingerprint is a SHA-256 hash over the fields that materially affect the
*inputs* a cell sees — seeds, sample counts, dataset, generation/scoring
parameters — excluding the adapter set (which is encoded in the cell identity
itself). Sweeps that share the fingerprint share cached cells.

Each pipeline (LLM-judge sweep, TRAIT-benchmark sweep, …) decides which
fields enter the hash; this module just standardises the hashing.
"""

from __future__ import annotations

from typing import Any

from src.eval_stages.run_id import run_id_from_dict


def fingerprint_from_fields(fields: dict[str, Any], *, length: int = 10) -> str:
    """Compute a deterministic cell-sweep fingerprint.

    Args:
        fields: JSON-serialisable dict of fingerprint-affecting config values.
            Callers must include everything that changes what the model sees
            or how it is scored; excluding a material field silently forks
            the cache and makes results irreproducible.
        length: Hex digits to keep (default 10; matches the judge-side
            historical value).

    Returns:
        Short hex prefix of the SHA-256 hash.
    """
    return run_id_from_dict(fields, length=length)

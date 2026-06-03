"""Deterministic content-addressed run IDs for eval stages.

Algorithm matches the proven pattern from the Bloom eval script:
``json.dumps(data, sort_keys=True, ensure_ascii=False)`` piped through
SHA-256, truncated to ``length`` hex characters.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def run_id_from_dict(data: dict[str, Any], *, length: int = 12) -> str:
    """Compute a deterministic short hex ID from a dict of config fields.

    Args:
        data: Config fields that materially affect the stage output.
            Must be JSON-serializable.
        length: Number of hex characters to keep (default 12).

    Returns:
        Hex prefix of the SHA-256 hash.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def chained_run_id(
    stage_name: str,
    config_fields: dict[str, Any],
    parent_run_id: str | None = None,
    *,
    length: int = 12,
) -> str:
    """Compute a run ID that chains to a parent stage's ID.

    The parent_run_id is included in the hash input so that any change
    to an upstream stage automatically invalidates all downstream stages.

    Args:
        stage_name: Name of this stage (included in hash for disambiguation).
        config_fields: Stage-specific config fields that affect output.
        parent_run_id: Run ID of the upstream stage, if any.
        length: Number of hex characters to keep (default 12).

    Returns:
        Hex prefix of the SHA-256 hash.
    """
    payload: dict[str, Any] = {"stage": stage_name, **config_fields}
    if parent_run_id is not None:
        payload["parent_run_id"] = parent_run_id
    return run_id_from_dict(payload, length=length)

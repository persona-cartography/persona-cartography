"""Shared runtime helpers for multi-turn conversation components."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """Return current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def message_append_id(sample_id: str, role: str, turn_index: int) -> str:
    """Build a deterministic message ID for appended conversation turns."""
    digest = hashlib.sha256(f"{sample_id}:{role}:{turn_index}".encode("utf-8")).hexdigest()[:24]
    return f"msg_{digest}"


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of at most ``size`` items."""
    if size <= 0:
        size = 1
    return [items[start : start + size] for start in range(0, len(items), size)]


def format_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Render a compact fixed-width ASCII progress bar."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, int(width * current / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_turn_label(turn_indices: list[int], total_turns: int) -> str:
    """Render one-based turn labels for logging."""
    if not turn_indices:
        return f"turn ?/{total_turns}"
    one_based = sorted({index + 1 for index in turn_indices})
    if len(one_based) == 1:
        return f"turn {one_based[0]}/{total_turns}"
    return f"turns {one_based[0]}-{one_based[-1]}/{total_turns}"


def log_phase_batch_progress(
    logger: logging.Logger,
    *,
    stage_name: str,
    batch_index: int,
    num_batches: int,
    items_processed: int,
    items_total: int,
    turn_label: str,
) -> None:
    """Emit consistent phase-level progress logs."""
    logger.info(
        "%s | %s | batch %d/%d | %s %d/%d",
        stage_name,
        turn_label,
        batch_index,
        num_batches,
        format_progress_bar(items_processed, items_total),
        items_processed,
        items_total,
    )

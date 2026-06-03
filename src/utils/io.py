"""Common I/O helpers for JSON and JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_jsonl_rows(path: str | Path) -> int:
    """Count non-empty rows in a JSONL file."""
    rows = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def iter_jsonl_batches(
    path: str | Path,
    batch_size: int,
    skip_rows: int = 0,
) -> Iterator[list[dict[str, Any]]]:
    """Yield JSONL records in batches, preserving file order.

    Args:
        path: Path to the JSONL file.
        batch_size: Maximum rows per yielded batch.
        skip_rows: Number of initial non-empty rows to skip.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")

    skipped = 0
    batch: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped < skip_rows:
                skipped += 1
                continue

            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of JSON-serializable dicts.
        path: Output path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path

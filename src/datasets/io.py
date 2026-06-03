"""Durable JSONL helpers for canonical dataset storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(data: Any) -> str:
    """Serialize JSON with stable separators and key ordering."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=_json_default)


def read_jsonl_tolerant(
    path: str | Path,
    *,
    recover_truncated: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Read JSONL and optionally recover from a truncated trailing line.

    Returns:
        Tuple of (records, recovered_truncated_line).
    """
    file_path = Path(path)
    if not file_path.exists():
        return [], False

    records: list[dict[str, Any]] = []
    recovered = False
    with file_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                if recover_truncated and lineno == _line_count(file_path):
                    recovered = True
                    break
                raise
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected object JSON row in {file_path} at line {lineno}, got {type(value).__name__}."
                )
            records.append(value)
    return records, recovered


def append_jsonl(
    path: str | Path, record: dict[str, Any], *, fsync: bool = True
) -> None:
    """Append one row to a JSONL file.

    Args:
        path: Target JSONL file path.
        record: Dict to serialize as one JSON line.
        fsync: If True (default), flush and fsync after writing for durability.
            Set to False in high-throughput loops where durability per-row
            is not required.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        handle.write("\n")
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())


def write_jsonl_atomic(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically rewrite a JSONL file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")
    tmp_path.replace(file_path)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)

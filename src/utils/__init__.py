"""General utility code.

Re-exports the JSONL/io and logging helpers at the package level so callers can
``from src.utils import count_jsonl_rows`` (matching the dev ``src_dev.utils``
surface). Heavier helpers (HF hub, LoRA composition/baking, PEFT manipulations)
are imported from their submodules directly to avoid pulling optional GPU
dependencies at package-import time.
"""

from src.utils.io import (
    count_jsonl_rows,
    iter_jsonl_batches,
    read_jsonl,
    write_jsonl,
)
from src.utils.logging import setup_logging

__all__ = [
    "count_jsonl_rows",
    "iter_jsonl_batches",
    "read_jsonl",
    "write_jsonl",
    "setup_logging",
]

"""Deterministic reservoir sampling for streaming external sources.

We want a reproducible N-row subsample from a streaming HF dataset. True
reservoir sampling scans the full source; for multi-million-row sources
(LMSYS, WildChat) we instead scan up to ``max_scan`` rows and reservoir-
sample within that bound. This is what the plan calls "post-filter"
sampling: filtering is done by the adapter's ``iter_raw`` before the rows
reach the sampler, so the output count is guaranteed to be either ``n``
(if the filter yielded ≥ n rows) or all available rows (if fewer).
"""

from __future__ import annotations

import random
from typing import Any, Iterator


def deterministic_sample(
    rows: Iterator[dict[str, Any]],
    *,
    n: int,
    seed: int,
    max_scan: int | None = None,
) -> list[dict[str, Any]]:
    """Reservoir-sample ``n`` rows from ``rows`` with a seeded RNG.

    Uses a fresh ``random.Random(seed)`` so we don't pollute the module
    RNG (which callers may have seeded for other reasons). Iterator order
    drives the deterministic selection.

    Args:
        rows: post-filter iterator of dicts.
        n: target sample count.
        seed: int seed pinning selection order.
        max_scan: optional upper bound on rows scanned. None = exhaust.

    Returns:
        Up to ``n`` rows, in insertion order into the reservoir.
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    scanned = 0
    for row in rows:
        scanned += 1
        if max_scan is not None and scanned > max_scan:
            break
        if len(reservoir) < n:
            reservoir.append(row)
        else:
            # Standard algorithm R: replace a random slot with prob n/i.
            j = rng.randint(0, scanned - 1)
            if j < n:
                reservoir[j] = row
    return reservoir

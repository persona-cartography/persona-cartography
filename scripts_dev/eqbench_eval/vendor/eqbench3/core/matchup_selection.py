
# File: ai/eqbench3/core/matchup_selection.py

# core/matchup_selection.py

import logging
import math
from typing import Dict, Any, List, Tuple, Optional, Set

from .elo_config import (
    rng # Import the global random generator
)


def create_matchup_signature(model_a: str, model_b: str, scenario_id: str, iter_idx: str) -> tuple:
    """
    Creates a consistent signature for a model A + model B + scenario + iteration matchup,
    regardless of which model is considered "test" vs "neighbor".
    Models A and B are expected to be logical names.

    Returns:
        tuple: A hashable signature (model1, model2, scenario_id, iter_idx) where model1, model2 are alphabetically sorted
    """
    # Sort the logical model names alphabetically for consistent signatures
    model1, model2 = sorted([model_a, model_b])
    return (model1, model2, scenario_id, iter_idx)

def build_existing_matchup_set(all_comparisons: List[Dict[str, Any]]) -> set:
    """
    Builds a set of signatures for matchups that already exist in previous comparisons.
    Assumes comparison pairs contain logical model names.

    Args:
        all_comparisons: List of comparison result dictionaries

    Returns:
        set: Signatures of existing matchups
    """
    existing_matchups = set()

    for comp in all_comparisons:
        if "error" in comp or "pair" not in comp or "scenario_id" not in comp:
            continue

        pair = comp.get("pair", {})
        # These should be logical names
        test_model = pair.get("test_model")
        neighbor_model = pair.get("neighbor_model")
        scenario_id = comp.get("scenario_id")

        # Get iteration index if available (structure might vary in older results)
        iter_idx = None
        if "iteration_index" in pair:
            iter_idx = pair["iteration_index"]
        else:
            # Try to extract from other possible locations if needed
            pass

        if all([test_model, neighbor_model, scenario_id, iter_idx is not None]):
            # Create signature using logical names
            sig = create_matchup_signature(test_model, neighbor_model, scenario_id, str(iter_idx))
            existing_matchups.add(sig)

    return existing_matchups

# Create a function to extract newly generated matchups and update the set
def update_existing_matchups_from_comparisons(comparisons, existing_set):
    """
    Extracts unique matchup signatures from newly generated comparisons
    (using logical names) and adds them to the existing_matchups set.
    """
    new_count = 0
    for comp in comparisons:
        if "error" in comp or "pair" not in comp or "scenario_id" not in comp:
            continue

        pair = comp.get("pair", {})
        # These should be logical names
        test_model = pair.get("test_model")
        neighbor_model = pair.get("neighbor_model")
        scenario_id = comp.get("scenario_id")

        # Get iteration index if available (structure might vary)
        iter_idx = None
        if "iteration_index" in pair:
            iter_idx = pair["iteration_index"]

        if all([test_model, neighbor_model, scenario_id, iter_idx is not None]):
            # Create signature using logical names
            sig = create_matchup_signature(test_model, neighbor_model, scenario_id, str(iter_idx))
            if sig not in existing_set:
                existing_set.add(sig)
                new_count += 1

    return new_count


def _evenly_spaced(pool: List[int], k: int) -> List[int]:
    """
    Deterministically choose *k* indices from *pool* so they are as
    evenly spread as possible.  If k >= len(pool) the whole pool is
    returned.

    Example:
        pool = [4, 5, 8, 10, 11, 12], k = 3  ->  [4, 10, 12]
    """
    if k <= 0 or not pool:
        return []
    # k == 1 would make (k-1) == 0 ⇒ ZeroDivisionError.
    # In that case we deterministically pick the first item.
    if k == 1:
        return [pool[0]]
    if k >= len(pool):
        return list(pool)

    step = (len(pool) - 1) / (k - 1)          # float step
    out  = [pool[0]]
    for i in range(1, k - 1):
        out.append(pool[int(round(i * step))])
    out.append(pool[-1])
    return sorted(set(out))                    # safety: drop dups


def _pick_matchups(rank: int,
                   ladder_len: int,
                   radius_tiers: Tuple[Optional[int], ...],
                   samples: int) -> List[int]:
    """
    Deterministic neighbour selection matching the spec:

    • Stage‑1  (radius_tiers == (None,)):
        evenly‑spaced anchors, depth = 1, ≤ `samples` picks.

    • Stage‑2/3 (radius_tiers == (1,2,3)):
        tier‑1 (±1)  → `samples`
        tier‑2 (±2)  → samples/2
        tier‑3 (±3)  → samples/4
        No global cap and no randomness.
    """
    if ladder_len <= 1 or samples <= 0:
        return []

    picks: Set[int] = set()

    # ── Stage-1  – sparse anchors -------------------------------------------
    if radius_tiers == (None,):
        step = max(1, math.ceil(ladder_len / samples))

        # 1. choose a “central” anchor with a small random jitter
        median_idx = (ladder_len - 1) // 2
        jitter     = rng.randint(-step // 2, step // 2)   # rng is the global Random(42)
        anchor     = min(max(0, median_idx + jitter), ladder_len - 1)

        # 2. walk outward in both directions at ‘step’ intervals
        anchors = [anchor]
        offset  = step
        while len(anchors) < samples and (anchor - offset >= 0 or anchor + offset < ladder_len):
            if anchor - offset >= 0:
                anchors.append(anchor - offset)
            if len(anchors) < samples and anchor + offset < ladder_len:
                anchors.append(anchor + offset)
            offset += step

        picks.update(i for i in anchors if i != rank)

         # ── eye-catching debug block ───────────────────────────────
        debug_box = [
            "",
            "╔════════════════════════════════════════════════════╗",
            "║                 STAGE-1 MATCHUPS                   ║",
            "╠════════════════════════════════════════════════════╣",
            f"║  Test-model rank : {rank} of {ladder_len - 1}                         ║",
            f"║  Opponent ranks  : {sorted(picks)}",
            "╚════════════════════════════════════════════════════╝",
        ]
        logging.info("\n".join(debug_box))
        # ──────────────────────────────────────────────────────────

        return sorted(picks)[:samples]

    # ── Stage‑2/3  (±1,±2,±3)  ──────────────────────────────────────────
    for tier, radius in enumerate(radius_tiers, start=1):
        want = samples // (2 ** (tier - 1))    # 8 / 4 / 2   or 40 / 20 / 10
        lo   = max(0, rank - radius)
        hi   = min(ladder_len - 1, rank + radius)
        pool = [i for i in range(lo, hi + 1) if i != rank]

        picks.update(_evenly_spaced(pool, want))

    return sorted(picks)

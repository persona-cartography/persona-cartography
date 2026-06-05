"""Canonical reduction of LLM-judge score JSONL files.

Every paper figure that reads ``judge_runs/.../{trait}_v2.jsonl`` must reduce it
the same way, otherwise the same adapter shows different scores across panels.
Before this module the combo-delta bars filtered failed-judge rows and took a
median over rollout repeats, while the soup heatmaps and the scaling panels took
a plain mean over *all* rows — silently averaging in the error sentinels (e.g.
``-99``) written when a judge call fails. These helpers are the single source of
truth: drop rows whose judge call did not succeed, then average (optionally over
per-response medians).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

# Statuses a judge row may carry. Only these two count as a real score; anything
# else (e.g. ``error``/``failed``, which carry the -99 / -99999 sentinels) is a
# failed call and must be excluded from aggregation.
_KEPT_STATUSES = {"success", "parse_error"}


def judge_scores(
    jsonl_path: Path, *, response_id_key: str = "response_id"
) -> list[tuple[str, float]]:
    """Return ``[(response_id, score)]`` for rows whose judge call succeeded.

    Rows are kept only when their ``status`` is in ``{"success", "parse_error"}``
    (this drops the error sentinels written on a failed call) and ``score`` is a
    real number. Scores are returned as ``float`` (OCEAN scores are integers, so
    this is loss-free).
    """
    rows: list[tuple[str, float]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") not in _KEPT_STATUSES:
                continue
            val = row.get("score")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            rows.append((str(row.get(response_id_key, "")), float(val)))
    return rows


def mean_judge_score(
    jsonl_path: Path,
    *,
    median_over_repeats: bool = False,
    response_id_key: str = "response_id",
) -> float | None:
    """Mean judge score over a jsonl, excluding failed-call rows.

    Args:
        jsonl_path: Path to a ``{trait}_v2.jsonl`` judge-output file.
        median_over_repeats: When True, first take the median over rows sharing a
            ``response_id`` (collapsing rollout repeats of the same response),
            then average those per-response medians. When False, take a plain
            mean over every kept row.
        response_id_key: Field used to group repeats (only used when
            ``median_over_repeats`` is True).

    Returns:
        The mean score, or ``None`` if no rows survived the filter.
    """
    rows = judge_scores(jsonl_path, response_id_key=response_id_key)
    if not rows:
        return None
    if median_over_repeats:
        grouped: dict[str, list[float]] = defaultdict(list)
        for rid, score in rows:
            grouped[rid].append(score)
        medians = [statistics.median(v) for v in grouped.values() if v]
        return statistics.fmean(medians) if medians else None
    return statistics.fmean(score for _, score in rows)

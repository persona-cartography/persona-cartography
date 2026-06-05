"""Aggregation utilities for evaluation results."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any


def aggregate_persona_metric_results(
    all_record_results: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    """Aggregate evaluation results across all records.

    Computes mean, median, min, max, and stdev for each numeric metric.
    Computes value_counts, mode, and unique_count for each string metric.

    Args:
        all_record_results: List of result dicts from each record.

    Returns:
        Dict with keys like "{metric_name}.mean" for numeric metrics
        and "{metric_name}.value_counts" for string metrics.
    """
    if not all_record_results:
        return {}

    numeric_values: dict[str, list[float]] = defaultdict(list)
    string_values: dict[str, list[str]] = defaultdict(list)

    for record_results in all_record_results:
        for key, value in record_results.items():
            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))
            elif isinstance(value, str):
                string_values[key].append(value)

    aggregates: dict[str, Any] = {}

    # Numeric aggregation
    for metric_name, values in sorted(numeric_values.items()):
        if not values:
            continue
        aggregates[f"{metric_name}.mean"] = statistics.mean(values)
        aggregates[f"{metric_name}.median"] = statistics.median(values)
        aggregates[f"{metric_name}.min"] = min(values)
        aggregates[f"{metric_name}.max"] = max(values)
        if len(values) >= 2:
            aggregates[f"{metric_name}.stdev"] = statistics.stdev(values)
        else:
            aggregates[f"{metric_name}.stdev"] = 0.0

    # Categorical/string aggregation
    for metric_name, values in sorted(string_values.items()):
        if not values:
            continue
        counts = Counter(values)
        aggregates[f"{metric_name}.value_counts"] = dict(counts)
        aggregates[f"{metric_name}.mode"] = counts.most_common(1)[0][0]
        aggregates[f"{metric_name}.unique_count"] = len(counts)

    return aggregates

"""Shared calibration utilities for human-rated LLM judges."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCALE_LABELS = [
    (-4, "Extreme low"),
    (-3, "Strong low"),
    (-2, "Moderate low"),
    (-1, "Slight low"),
    (0, "Neutral / no signal"),
    (1, "Slight high"),
    (2, "Moderate high"),
    (3, "Strong high"),
    (4, "Extreme high"),
]


class CalibrationItem(BaseModel):
    """One calibration item used for judge development or heldout evaluation."""

    id: str
    trait: str = "neuroticism"
    split: Literal["dev", "heldout"]
    category: str
    question: str
    response: str
    author_score: int = Field(ge=-4, le=4)
    author_notes: str = ""

    @field_validator("id", "trait", "split", "category", "question", "response")
    @classmethod
    def _ensure_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value


class ReferenceItem(BaseModel):
    """A calibration item with attached human-rating information."""

    id: str
    trait: str
    split: str
    category: str
    question: str
    response: str
    author_score: int
    author_notes: str
    rater_scores: dict[str, int | None]
    median_score: float | None
    available_raters: int


class ReferenceSet(BaseModel):
    """Human-derived reference labels for a calibration split."""

    trait: str
    split: str
    raters: list[str]
    items: list[ReferenceItem]


def load_calibration_items(path: Path) -> list[CalibrationItem]:
    items: list[CalibrationItem] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = CalibrationItem(**json.loads(line))
            if item.id in seen_ids:
                raise ValueError(f"Duplicate calibration item id: {item.id}")
            seen_ids.add(item.id)
            items.append(item)
    return items


def filter_items(items: list[CalibrationItem], *, split: str | None = None) -> list[CalibrationItem]:
    if split is None:
        return items
    return [item for item in items if item.split == split]


def save_reference_set(reference_set: ReferenceSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reference_set.model_dump_json(indent=2), encoding="utf-8")


def load_reference_set(path: Path) -> ReferenceSet:
    return ReferenceSet.model_validate_json(path.read_text(encoding="utf-8"))


def reference_scores(reference_set: ReferenceSet) -> list[float | None]:
    return [item.median_score for item in reference_set.items]


def author_scores(items: list[CalibrationItem]) -> list[int]:
    return [item.author_score for item in items]


def _average_tied_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        start = index
        current = ordered[index][1]
        while index < len(ordered) and ordered[index][1] == current:
            index += 1
        end = index
        mean_rank = (start + 1 + end) / 2.0
        for original_index, _ in ordered[start:end]:
            ranks[original_index] = mean_rank
    return ranks


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    if denominator <= 1e-12:
        return float("nan")
    return numerator / denominator


def spearman_r(xs: list[float], ys: list[float]) -> float:
    return pearson_r(_average_tied_ranks(xs), _average_tied_ranks(ys))


def mae(xs: list[float], ys: list[float]) -> float:
    return statistics.mean(abs(x - y) for x, y in zip(xs, ys))


def within_one_rate(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(abs(x - y) <= 1.0 for x, y in zip(xs, ys)) / len(xs)


def exact_match_rate(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(x == y for x, y in zip(xs, ys)) / len(xs)


def quadratic_weighted_agreement(
    xs: list[int],
    ys: list[int],
    *,
    score_min: int | None = None,
    score_max: int | None = None,
) -> float:
    if len(xs) < 2:
        return float("nan")
    _min = score_min if score_min is not None else min(min(xs), min(ys))
    _max = score_max if score_max is not None else max(max(xs), max(ys))
    values = list(range(_min, _max + 1))
    index_by_value = {value: idx for idx, value in enumerate(values)}
    size = len(values)
    observed = [[0.0 for _ in range(size)] for _ in range(size)]
    for x, y in zip(xs, ys):
        observed[index_by_value[x]][index_by_value[y]] += 1.0

    counts_x = [sum(row) for row in observed]
    counts_y = [sum(observed[row][col] for row in range(size)) for col in range(size)]
    total = float(len(xs))
    expected = [
        [(counts_x[row] * counts_y[col]) / total for col in range(size)]
        for row in range(size)
    ]

    weighted_observed = 0.0
    weighted_expected = 0.0
    max_distance = float((size - 1) ** 2)
    for row in range(size):
        for col in range(size):
            weight = ((row - col) ** 2) / max_distance
            weighted_observed += weight * observed[row][col]
            weighted_expected += weight * expected[row][col]
    if weighted_expected <= 1e-12:
        return float("nan")
    return 1.0 - (weighted_observed / weighted_expected)


def _valid_pairs(
    reference: list[int | float | None],
    predicted: list[int | float | None],
) -> tuple[list[float], list[float]]:
    pairs = [
        (float(ref), float(pred))
        for ref, pred in zip(reference, predicted)
        if ref is not None and pred is not None
    ]
    return [left for left, _ in pairs], [right for _, right in pairs]


def summarize_pair(
    reference: list[int | float | None],
    predicted: list[int | float | None],
) -> dict[str, float | int]:
    ref_values, pred_values = _valid_pairs(reference, predicted)
    if not ref_values:
        return {
            "n": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mae": float("nan"),
            "within_one": float("nan"),
            "exact": float("nan"),
        }
    return {
        "n": len(ref_values),
        "pearson": pearson_r(ref_values, pred_values),
        "spearman": spearman_r(ref_values, pred_values),
        "mae": mae(ref_values, pred_values),
        "within_one": within_one_rate(ref_values, pred_values),
        "exact": exact_match_rate(ref_values, pred_values),
    }


def load_scores_from_csv(csv_path: Path, items: list[CalibrationItem]) -> dict[str, list[int | None]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = row.get("id", "").strip()
            if item_id:
                rows_by_id[item_id] = row

    if not rows_by_id:
        return {}
    sample_row = next(iter(rows_by_id.values()))
    score_columns = [key for key in sample_row if key.startswith("score_")]
    if not score_columns:
        raise ValueError(f"No score_* columns found in {csv_path}")

    merged: dict[str, list[int | None]] = {
        column.removeprefix("score_"): [] for column in score_columns
    }
    for item in items:
        row = rows_by_id.get(item.id, {})
        for column in score_columns:
            raw = row.get(column, "").strip()
            if not raw:
                merged[column.removeprefix("score_")].append(None)
                continue
            score = int(raw)
            if score < -4 or score > 4:
                raise ValueError(f"Invalid score {score} for item {item.id} in {csv_path}")
            merged[column.removeprefix("score_")].append(score)
    return merged


def aggregate_reference_from_csvs(
    csv_paths: list[Path],
    items: list[CalibrationItem],
    *,
    require_complete: bool = False,
) -> ReferenceSet:
    merged_scores: dict[str, list[int | None]] = {}
    for csv_path in csv_paths:
        scores = load_scores_from_csv(csv_path, items)
        duplicates = set(scores).intersection(merged_scores)
        if duplicates:
            raise ValueError(f"Duplicate rater names in CSV imports: {sorted(duplicates)}")
        merged_scores.update(scores)

    raters = sorted(merged_scores)
    reference_items: list[ReferenceItem] = []
    for index, item in enumerate(items):
        rater_scores = {rater: merged_scores[rater][index] for rater in raters}
        available = [score for score in rater_scores.values() if score is not None]
        if require_complete and len(available) != len(raters):
            raise ValueError(f"Incomplete ratings for item {item.id}")
        median_score = statistics.median(available) if available else None
        reference_items.append(
            ReferenceItem(
                id=item.id,
                trait=item.trait,
                split=item.split,
                category=item.category,
                question=item.question,
                response=item.response,
                author_score=item.author_score,
                author_notes=item.author_notes,
                rater_scores=rater_scores,
                median_score=median_score,
                available_raters=len(available),
            )
        )

    split_names = {item.split for item in items}
    split_name = split_names.pop() if len(split_names) == 1 else "mixed"
    return ReferenceSet(
        trait=items[0].trait if items else "neuroticism",
        split=split_name,
        raters=raters,
        items=reference_items,
    )


def summarize_inter_rater(reference_set: ReferenceSet) -> dict[str, object]:
    pairwise: list[dict[str, object]] = []
    rater_names = reference_set.raters
    for index, left in enumerate(rater_names):
        for right in rater_names[index + 1:]:
            left_scores = [item.rater_scores[left] for item in reference_set.items]
            right_scores = [item.rater_scores[right] for item in reference_set.items]
            stats = summarize_pair(left_scores, right_scores)
            valid_left = []
            valid_right = []
            for left_score, right_score in zip(left_scores, right_scores):
                if left_score is None or right_score is None:
                    continue
                valid_left.append(int(left_score))
                valid_right.append(int(right_score))
            pairwise.append(
                {
                    "rater_a": left,
                    "rater_b": right,
                    **stats,
                    "quadratic_weighted_agreement": quadratic_weighted_agreement(
                        valid_left,
                        valid_right,
                    ),
                }
            )

    if not pairwise:
        return {"pairwise": [], "summary": {}}
    summary: dict[str, float] = {}
    for key in [
        "pearson",
        "spearman",
        "mae",
        "within_one",
        "exact",
        "quadratic_weighted_agreement",
    ]:
        values = [float(pair[key]) for pair in pairwise if not math.isnan(float(pair[key]))]
        if values:
            summary[f"mean_{key}"] = statistics.mean(values)
    return {"pairwise": pairwise, "summary": summary}


def category_breakdown(
    items: list[CalibrationItem],
    reference: list[int | float | None],
    predicted: list[int | float | None],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, dict[str, list[int | float | None]]] = defaultdict(
        lambda: {"reference": [], "predicted": []}
    )
    for item, ref_value, pred_value in zip(items, reference, predicted):
        grouped[item.category]["reference"].append(ref_value)
        grouped[item.category]["predicted"].append(pred_value)
    return {
        category: summarize_pair(values["reference"], values["predicted"])
        for category, values in sorted(grouped.items())
    }

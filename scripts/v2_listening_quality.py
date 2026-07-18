#!/usr/bin/env python3
"""Absolute listening-quality gates shared by V2 selection and release."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


SCORE_FIELDS = ("prompt_alignment", "melody", "structure", "audio_quality")
MINIMUM_DIMENSION_MEAN = 3.0
MINIMUM_INDIVIDUAL_SCORE = 2


def summarize_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require acceptable absolute scores, not merely the best bad candidate."""
    values: dict[str, list[int]] = defaultdict(list)
    errors: list[str] = []
    if not records:
        errors.append("listening_records_missing")
    for index, record in enumerate(records):
        scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
        for field in SCORE_FIELDS:
            try:
                score = int(scores.get(field))
            except (TypeError, ValueError):
                score = 0
            values[field].append(score)
            if not 1 <= score <= 5:
                errors.append(f"invalid_score:{index}:{field}:{score}")

    dimension_means = {
        field: (sum(field_values) / len(field_values) if field_values else 0.0)
        for field, field_values in values.items()
    }
    for field in SCORE_FIELDS:
        mean = dimension_means.get(field, 0.0)
        if mean < MINIMUM_DIMENSION_MEAN:
            errors.append(
                f"dimension_mean_below_minimum:{field}:{mean:.3f}:{MINIMUM_DIMENSION_MEAN:.3f}"
            )
        minimum = min(values.get(field, [0]))
        if minimum < MINIMUM_INDIVIDUAL_SCORE:
            errors.append(
                f"individual_score_below_minimum:{field}:{minimum}:{MINIMUM_INDIVIDUAL_SCORE}"
            )
    all_scores = [score for field_values in values.values() for score in field_values]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "status": "pass" if not errors else "failed",
        "quality_accepted": not errors,
        "records": len(records),
        "dimension_means": dimension_means,
        "overall_mean": overall_mean,
        "minimum_dimension_mean": MINIMUM_DIMENSION_MEAN,
        "minimum_individual_score": MINIMUM_INDIVIDUAL_SCORE,
        "errors": errors,
    }

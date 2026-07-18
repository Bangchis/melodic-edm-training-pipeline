#!/usr/bin/env python3
"""Absolute listening-quality gates shared by V2 selection and release."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


SCORE_FIELDS = ("prompt_alignment", "melody", "structure", "audio_quality")
MINIMUM_DIMENSION_MEAN = 3.0
MINIMUM_INDIVIDUAL_SCORE = 2
MINIMUM_INDIVIDUAL_BY_DIMENSION = {
    "prompt_alignment": 3,
    "melody": 2,
    "structure": 2,
    "audio_quality": 2,
}


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
        failure_modes = record.get("failure_modes")
        if not isinstance(failure_modes, dict):
            errors.append(f"failure_modes_missing:{index}")
        else:
            for name in ("distorted", "collapsed", "static_loop"):
                if failure_modes.get(name) is True:
                    errors.append(f"audible_failure_mode:{index}:{name}")
                elif failure_modes.get(name) is not False:
                    errors.append(f"failure_mode_not_boolean:{index}:{name}")

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
        required_minimum = MINIMUM_INDIVIDUAL_BY_DIMENSION[field]
        if minimum < required_minimum:
            errors.append(
                f"individual_score_below_minimum:{field}:{minimum}:{required_minimum}"
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
        "minimum_individual_by_dimension": MINIMUM_INDIVIDUAL_BY_DIMENSION,
        "errors": errors,
    }

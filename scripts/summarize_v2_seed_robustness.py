#!/usr/bin/env python3
"""Summarize multi-seed quality without allowing a single lucky seed to pass."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2_common import atomic_json


SCORE_FIELDS = ("prompt_alignment", "melody", "structure", "audio_quality")
FAILURE_MODES = ("distorted", "collapsed", "static_loop", "intelligible_vocals")


def seed_passes(record: dict[str, Any], minimum_score: int) -> bool:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    failures = record.get("failure_modes") if isinstance(record.get("failure_modes"), dict) else {}
    return all(int(scores.get(field, 0)) >= minimum_score for field in SCORE_FIELDS) and not any(
        failures.get(name) is True for name in FAILURE_MODES
    )


def summarize(
    records: list[dict[str, Any]],
    *,
    expected_seeds: int,
    minimum_score: int,
    minimum_pass_rate: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("source_prompt_id") or record.get("prompt_id") or "")].append(record)
    errors: list[str] = []
    prompts: dict[str, Any] = {}
    all_passes = 0
    all_records = 0
    for prompt_id, rows in sorted(grouped.items()):
        passes = [seed_passes(row, minimum_score) for row in rows]
        pass_count = sum(passes)
        pass_rate = pass_count / len(rows) if rows else 0.0
        dimension_means = {
            field: sum(int(row.get("scores", {}).get(field, 0)) for row in rows) / len(rows)
            if rows
            else 0.0
            for field in SCORE_FIELDS
        }
        failure_counts = {
            name: sum(row.get("failure_modes", {}).get(name) is True for row in rows)
            for name in FAILURE_MODES
        }
        prompts[prompt_id] = {
            "records": len(rows),
            "seeds": [row.get("seed") for row in rows],
            "passed_seeds": pass_count,
            "pass_rate": pass_rate,
            "dimension_means": dimension_means,
            "failure_counts": failure_counts,
        }
        if len(rows) != expected_seeds:
            errors.append(f"seed_count_mismatch:{prompt_id}:{len(rows)}:{expected_seeds}")
        if pass_rate < minimum_pass_rate:
            errors.append(
                f"prompt_pass_rate_below_minimum:{prompt_id}:{pass_rate:.3f}:{minimum_pass_rate:.3f}"
            )
        all_passes += pass_count
        all_records += len(rows)
    if not grouped:
        errors.append("seed_records_missing")
    overall_pass_rate = all_passes / all_records if all_records else 0.0
    if overall_pass_rate < minimum_pass_rate:
        errors.append(
            f"overall_pass_rate_below_minimum:{overall_pass_rate:.3f}:{minimum_pass_rate:.3f}"
        )
    return {
        "status": "pass" if not errors else "failed",
        "expected_seeds_per_prompt": expected_seeds,
        "minimum_score_per_dimension": minimum_score,
        "minimum_pass_rate": minimum_pass_rate,
        "records": all_records,
        "passed_seeds": all_passes,
        "overall_pass_rate": overall_pass_rate,
        "prompts": prompts,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--minimum-score", type=int, default=3)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.8)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    report = Path(args.report)
    output = Path(args.output)
    if not report.is_absolute():
        report = root / report
    if not output.is_absolute():
        output = root / output
    value = json.loads(report.read_text(encoding="utf-8"))
    if value.get("status") != "pass":
        raise RuntimeError("listening score report has not passed technical validation")
    result = summarize(
        value.get("results", []),
        expected_seeds=args.expected_seeds,
        minimum_score=args.minimum_score,
        minimum_pass_rate=args.minimum_pass_rate,
    )
    result["source_report"] = str(report.relative_to(root))
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

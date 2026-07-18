#!/usr/bin/env python3
"""Apply the absolute V2 listening gate to one MOSS score report."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from v2_common import atomic_json
from v2_listening_quality import summarize_quality


def compare_prompt_alignment(
    quality: dict[str, Any],
    current_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    *,
    reference_checkpoint: str,
    maximum_regression: float,
) -> dict[str, Any]:
    """Reject a final adapter that materially regresses from selected best-val alignment."""
    output = deepcopy(quality)
    current = [int(row["scores"]["prompt_alignment"]) for row in current_records]
    reference = [
        int(row["scores"]["prompt_alignment"])
        for row in reference_records
        if row.get("checkpoint") == reference_checkpoint
    ]
    if not current or not reference:
        output["errors"].append("prompt_alignment_reference_records_missing")
        current_mean = mean(current) if current else 0.0
        reference_mean = mean(reference) if reference else 0.0
    else:
        current_mean = mean(current)
        reference_mean = mean(reference)
        if current_mean + maximum_regression < reference_mean:
            output["errors"].append(
                "final_prompt_alignment_regressed:"
                f"{current_mean:.3f}:{reference_mean:.3f}:{maximum_regression:.3f}"
            )
    accepted = not any(
        error.startswith(("prompt_alignment_reference_records_missing", "final_prompt_alignment_regressed"))
        for error in output["errors"]
    )
    output["prompt_alignment_comparison"] = {
        "accepted": accepted,
        "final_mean": current_mean,
        "reference_checkpoint": reference_checkpoint,
        "reference_mean": reference_mean,
        "maximum_allowed_regression": maximum_regression,
    }
    output["quality_accepted"] = not output["errors"]
    output["status"] = "pass" if output["quality_accepted"] else "failed"
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--reference-report")
    parser.add_argument("--selection-report")
    parser.add_argument("--maximum-prompt-alignment-regression", type=float, default=0.34)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"listening score report has not passed technical validation: {report_path}")
    quality = summarize_quality(report.get("results", []), profile=args.profile)
    if bool(args.reference_report) != bool(args.selection_report):
        raise ValueError("reference-report and selection-report must be supplied together")
    if args.reference_report:
        reference_path = Path(args.reference_report)
        selection_path = Path(args.selection_report)
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        if not selection_path.is_absolute():
            selection_path = root / selection_path
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if reference.get("status") != "pass" or selection.get("status") != "pass":
            raise RuntimeError("reference listening and selection reports must pass")
        quality = compare_prompt_alignment(
            quality,
            report.get("results", []),
            reference.get("results", []),
            reference_checkpoint=str(selection["selected_checkpoint"]),
            maximum_regression=args.maximum_prompt_alignment_regression,
        )
    quality["source_report"] = str(report_path.relative_to(root))
    atomic_json(output_path, quality)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0 if quality["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

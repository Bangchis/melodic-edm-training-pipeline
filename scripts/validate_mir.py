#!/usr/bin/env python3
"""Validate that MIR outputs cover every accepted record exactly once."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def numeric_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(item) for item in values) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    parser.add_argument("--key-confidence-threshold", type=float, default=0.65)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source = [row for row in read_jsonl(root / args.manifest) if row.get("quality_status") == "accepted"]
    expected = {row["sample_id"]: row for row in source}
    mir_dir = root / "data" / "mir"
    files = sorted(mir_dir.glob("*.json"))
    observed = {path.stem for path in files}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for sid in sorted(set(expected) - observed):
        errors.append({"sample_id": sid, "reason": "mir_missing"})
    for sid in sorted(observed - set(expected)):
        errors.append({"sample_id": sid, "reason": "unexpected_mir_file"})

    for sid, manifest_row in sorted(expected.items()):
        path = mir_dir / f"{sid}.json"
        if not path.is_file():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"sample_id": sid, "reason": f"invalid_json:{type(exc).__name__}:{exc}"})
            continue
        rows.append(row)
        if row.get("sample_id") != sid:
            errors.append({"sample_id": sid, "reason": "sample_id_mismatch"})
        if row.get("analysis_status") != "complete":
            errors.append({"sample_id": sid, "reason": "analysis_not_complete"})

        bpm = row.get("bpm")
        if not isinstance(bpm, int) or isinstance(bpm, bool) or not 40 <= bpm <= 250:
            errors.append({"sample_id": sid, "reason": f"invalid_bpm:{bpm}"})
        elif bpm < 80 or bpm > 190:
            warnings.append({"sample_id": sid, "reason": f"bpm_half_or_double_time_candidate:{bpm}"})
        raw_bpm = row.get("bpm_raw")
        normalization = row.get("bpm_normalization")
        if isinstance(raw_bpm, int) and raw_bpm < 80:
            if bpm != raw_bpm * 2 or normalization != "double_half_time_below_80":
                errors.append({"sample_id": sid, "reason": "half_time_bpm_not_normalized"})

        duration = float(manifest_row.get("duration") or 0)
        for name in ("beats", "downbeats"):
            values = numeric_list(row.get(name))
            if not values:
                errors.append({"sample_id": sid, "reason": f"{name}_missing_or_invalid"})
                continue
            if values != sorted(values):
                errors.append({"sample_id": sid, "reason": f"{name}_not_sorted"})
            if values[0] < -0.05 or values[-1] > duration + 1.0:
                errors.append({"sample_id": sid, "reason": f"{name}_outside_audio"})

        sections = row.get("sections")
        if not isinstance(sections, list) or not sections:
            errors.append({"sample_id": sid, "reason": "sections_missing"})
        else:
            previous_start = -1.0
            for section in sections:
                try:
                    start = float(section["start"])
                    end = float(section["end"])
                except (KeyError, TypeError, ValueError):
                    errors.append({"sample_id": sid, "reason": "section_invalid"})
                    break
                if start < previous_start or start < -0.05 or end <= start or end > duration + 1.0:
                    errors.append({"sample_id": sid, "reason": "section_bounds_invalid"})
                    break
                previous_start = start

        confidence = row.get("key_confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append({"sample_id": sid, "reason": "key_confidence_invalid"})
        elif row.get("keyscale") and confidence < args.key_confidence_threshold:
            errors.append({"sample_id": sid, "reason": "key_below_threshold"})
        elif not row.get("keyscale"):
            warnings.append({"sample_id": sid, "reason": "key_omitted_low_confidence"})
        if row.get("timesignature") not in {None, "", "4"}:
            errors.append({"sample_id": sid, "reason": "timesignature_invalid"})
        elif not row.get("timesignature"):
            warnings.append({"sample_id": sid, "reason": "timesignature_omitted"})

    report = {
        "status": "pass" if not errors and len(rows) == len(source) else "failed",
        "expected_records": len(source),
        "mir_files": len(files),
        "validated_records": len(rows),
        "deduplication_performed": False,
        "bpm_min": min((row["bpm"] for row in rows if isinstance(row.get("bpm"), int)), default=None),
        "bpm_max": max((row["bpm"] for row in rows if isinstance(row.get("bpm"), int)), default=None),
        "key_present": sum(bool(row.get("keyscale")) for row in rows),
        "timesignature_present": sum(bool(row.get("timesignature")) for row in rows),
        "bpm_normalized_from_half_time": sum(
            row.get("bpm_normalization") == "double_half_time_below_80" for row in rows
        ),
        "section_label_counts": dict(Counter(
            section.get("label", "") for row in rows for section in row.get("sections", [])
        )),
        "errors": errors,
        "warnings": warnings,
    }
    atomic_json(root / "data" / "mir_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

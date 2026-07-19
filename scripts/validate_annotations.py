#!/usr/bin/env python3
"""Validate exact annotation coverage and summarize the final caption corpus."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from annotate_openrouter import validate_annotation, word_count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source = [row for row in read_jsonl(root / args.manifest) if row.get("quality_status") == "accepted"]
    expected = {row["sample_id"]: row for row in source}
    taxonomy = json.loads((root / "configs" / "taxonomy.json").read_text(encoding="utf-8"))
    annotation_dir = root / "data" / "annotations"
    files = sorted(annotation_dir.glob("*.json"))
    observed = {path.stem for path in files}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    records = []
    captions: dict[str, list[str]] = defaultdict(list)

    for sid in sorted(set(expected) - observed):
        errors.append({"sample_id": sid, "reason": "annotation_missing"})
    for sid in sorted(observed - set(expected)):
        errors.append({"sample_id": sid, "reason": "unexpected_annotation_file"})

    for sid, source_row in sorted(expected.items()):
        path = annotation_dir / f"{sid}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            mir = json.loads((root / "data" / "mir" / f"{sid}.json").read_text(encoding="utf-8"))
            annotation = record["annotation"]
        except Exception as exc:
            errors.append({"sample_id": sid, "reason": f"invalid_annotation_record:{type(exc).__name__}:{exc}"})
            continue
        records.append(record)
        if record.get("sample_id") != sid or record.get("annotation_status") != "accepted":
            errors.append({"sample_id": sid, "reason": "annotation_record_not_accepted"})
        validation_errors = validate_annotation(annotation, source_row, taxonomy, mir)
        for reason in validation_errors:
            errors.append({"sample_id": sid, "reason": reason})
        if float(annotation.get("annotation_confidence", 0)) < 0.70:
            errors.append({"sample_id": sid, "reason": "annotation_confidence_below_threshold"})
        canonical = re.sub(r"\s+", " ", str(annotation.get("canonical_caption", "")).strip().lower())
        captions[canonical].append(sid)

    for caption, sample_ids in captions.items():
        raw_paths = {str(expected[sid].get("raw_path") or expected[sid].get("video_id")) for sid in sample_ids}
        if caption and len(raw_paths) > 1:
            warnings.append({
                "reason": "identical_caption_across_different_audio",
                "sample_ids": sample_ids,
            })

    annotations = [record["annotation"] for record in records]
    word_counts = [word_count(item.get("canonical_caption", "")) for item in annotations]
    report = {
        "status": "pass" if not errors and len(records) == len(source) else "failed",
        "expected_records": len(source),
        "annotation_files": len(files),
        "validated_records": len(records),
        "deduplication_performed": False,
        "caption_variants_per_record": 4,
        "canonical_word_count": {
            "min": min(word_counts, default=None),
            "max": max(word_counts, default=None),
            "mean": round(mean(word_counts), 2) if word_counts else None,
        },
        "primary_genres": dict(Counter(item.get("primary_genre", "") for item in annotations)),
        "style_families": dict(Counter(value for item in annotations for value in item.get("style_families", []))),
        "moods": dict(Counter(value for item in annotations for value in item.get("moods", []))),
        "instruments": dict(Counter(
            value.get("name", "") for item in annotations for value in item.get("main_instruments", [])
        )),
        "sanitization_actions": dict(Counter(
            action.get("action", "")
            for record in records
            for action in (record.get("annotation_sanitization") or [])
        )),
        "total_cost_usd": round(sum(
            float((record.get("annotation_usage") or {}).get("cost") or 0) for record in records
        ), 6),
        "errors": errors,
        "warnings": warnings,
    }
    atomic_json(root / "data" / "annotation_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge Gemini/Qwen annotations with MOSS into exactly three captions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from annotate_moss_music import PROMPT_REVISION
from v2_common import (
    CAPTION_TYPES,
    atomic_json,
    atomic_jsonl,
    caption_map,
    file_sha256,
    grouped_split,
    object_sha256,
    parent_song_id,
    read_jsonl,
    validate_caption_set,
    word_count,
)


def merge_record(
    row: dict[str, Any], old_record: dict[str, Any], moss_record: dict[str, Any], split: str
) -> dict[str, Any]:
    """Merge one accepted base annotation with its audio-grounded supplement."""
    old_annotation = old_record["annotation"]
    if object_sha256(old_annotation) != moss_record.get("existing_annotation_sha256"):
        raise ValueError("existing_annotation_changed_after_moss_run")
    audio_path = Path(row["final_audio_path"])
    if file_sha256(audio_path) != moss_record.get("audio_sha256"):
        raise ValueError("audio_changed_after_moss_run")
    supplement = moss_record["supplement"]
    if moss_record.get("prompt_revision") != PROMPT_REVISION:
        raise ValueError("moss_prompt_revision_mismatch")
    captions = caption_map(supplement.get("captions"))
    errors = validate_caption_set(
        captions,
        artist=str(row.get("expected_artist") or ""),
        title=str(row.get("expected_title") or ""),
    )
    if errors:
        raise ValueError(";".join(errors))

    parent = parent_song_id(row)
    master = {
        "schema_version": "2.0",
        "base_annotation": old_annotation,
        "moss_music_supplement": supplement["audible_facts"],
        "merged_captions": captions,
        "annotation_confidence": {
            "base": old_annotation.get("annotation_confidence"),
            "moss": supplement.get("confidence"),
        },
        "provenance": {
            "base_model": old_record.get("annotation_model"),
            "base_provider": old_record.get("annotation_provider"),
            "base_annotation_sha256": moss_record["existing_annotation_sha256"],
            "moss_model": moss_record["model_id"],
            "moss_model_revision": moss_record["model_revision"],
            "moss_source_revision": moss_record["source_revision"],
            "moss_prompt_revision": moss_record["prompt_revision"],
            "moss_response_sha256": moss_record["raw_response_sha256"],
            "audio_sha256": moss_record["audio_sha256"],
        },
    }
    return {
        "schema_version": "2.0",
        "moss_prompt_revision": PROMPT_REVISION,
        "sample_id": row["sample_id"],
        "record_key": row["record_key"],
        "parent_song_id": parent,
        "split": split,
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "caption": captions["canonical"],
        "caption_variants": [
            {"type": name, "text": captions[name]}
            for name in CAPTION_TYPES
        ],
        "master_annotation": master,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-count", type=int, default=35)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = sorted(
        read_jsonl(root / "data" / "final_manifest.jsonl"),
        key=lambda row: row["sample_id"],
    )
    splits = grouped_split(rows, args.validation_count, args.seed)
    output_dir = root / "data_v2" / "annotations"
    merged_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row["sample_id"])
        try:
            old_record = json.loads(
                (root / "data" / "annotations" / f"{sample_id}.json").read_text(encoding="utf-8")
            )
            moss_record = json.loads(
                (root / "data_v2" / "moss_annotations" / f"{sample_id}.json").read_text(encoding="utf-8")
            )
            merged = merge_record(row, old_record, moss_record, splits[sample_id])
            atomic_json(output_dir / f"{sample_id}.json", merged)
            merged_rows.append({
                **row,
                "parent_song_id": merged["parent_song_id"],
                "split": merged["split"],
                "v2_annotation_path": str(output_dir / f"{sample_id}.json"),
                "caption_types": list(CAPTION_TYPES),
            })
            print(f"[{index}/{len(rows)}] {sample_id} PASS {merged['split']}", flush=True)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}:{exc}"})
            print(f"[{index}/{len(rows)}] {sample_id} FAILED {errors[-1]['reason']}", flush=True)

    parent_splits: dict[str, set[str]] = {}
    for row in merged_rows:
        parent_splits.setdefault(row["parent_song_id"], set()).add(row["split"])
    crossing = sorted(parent for parent, values in parent_splits.items() if len(values) > 1)
    counts = Counter(row["split"] for row in merged_rows)
    caption_counts = [
        word_count(
            json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))["caption"]
        )
        for row in merged_rows
    ]
    if crossing:
        errors.append({"sample_id": "*", "reason": f"parent_split_crossing:{crossing}"})
    if counts != Counter({"train": len(rows) - args.validation_count, "validation": args.validation_count}):
        errors.append({"sample_id": "*", "reason": f"split_counts_invalid:{dict(counts)}"})
    status = "pass" if not errors and len(merged_rows) == len(rows) else "failed"
    report = {
        "status": status,
        "schema_version": "2.0",
        "moss_prompt_revision": PROMPT_REVISION,
        "records_expected": len(rows),
        "records_merged": len(merged_rows),
        "train_records": counts.get("train", 0),
        "validation_records": counts.get("validation", 0),
        "test_records": 0,
        "parent_groups": len(parent_splits),
        "parent_split_crossings": crossing,
        "caption_variants_per_record": 3,
        "caption_types": list(CAPTION_TYPES),
        "canonical_word_count": {
            "min": min(caption_counts, default=None),
            "max": max(caption_counts, default=None),
            "mean": round(mean(caption_counts), 2) if caption_counts else None,
        },
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "annotation_merge_report.json", report)
    atomic_jsonl(root / "data_v2" / "manifest.jsonl", merged_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

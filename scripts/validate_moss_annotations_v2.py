#!/usr/bin/env python3
"""Validate complete, normalized and provenance-locked MOSS V2 supplements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from annotate_moss_music import MODEL_REVISION, PROMPT_REVISION, validate_supplement
from v2_common import (
    atomic_json,
    file_sha256,
    object_sha256,
    parent_song_id,
    read_jsonl,
    word_count,
)


FORBIDDEN_REASONING_KEYS = {"raw_response", "reasoning", "thinking", "chain_of_thought"}


def forbidden_keys(value: Any) -> set[str]:
    """Return forbidden model-reasoning keys found recursively."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_REASONING_KEYS:
                found.add(str(key))
            found.update(forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(forbidden_keys(item))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data" / "final_manifest.jsonl"), key=lambda row: row["sample_id"])
    expected = {str(row["sample_id"]) for row in rows}
    annotation_dir = root / "data_v2" / "moss_annotations"
    observed = {path.stem for path in annotation_dir.glob("*.json")}
    errors: list[dict[str, Any]] = []
    for sample_id in sorted(expected - observed):
        errors.append({"sample_id": sample_id, "reason": "moss_annotation_missing"})
    for sample_id in sorted(observed - expected):
        errors.append({"sample_id": sample_id, "reason": "unexpected_moss_annotation"})

    confidences: list[float] = []
    caption_words: dict[str, list[int]] = {
        "canonical": [], "composition": [], "production": [],
    }
    for row in rows:
        sample_id = str(row["sample_id"])
        path = annotation_dir / f"{sample_id}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            supplement, validation_errors = validate_supplement(record["supplement"], row)
            if validation_errors:
                raise ValueError(";".join(validation_errors))
            if record["supplement"] != supplement:
                raise ValueError("supplement_is_not_schema_normalized")
            if record.get("sample_id") != sample_id:
                raise ValueError("sample_id_mismatch")
            if record.get("parent_song_id") != parent_song_id(row):
                raise ValueError("parent_song_id_mismatch")
            if record.get("model_revision") != MODEL_REVISION:
                raise ValueError("moss_model_revision_mismatch")
            if record.get("prompt_revision") != PROMPT_REVISION:
                raise ValueError("moss_prompt_revision_mismatch")
            audio_path = Path(row["final_audio_path"])
            if record.get("audio_sha256") != file_sha256(audio_path):
                raise ValueError("audio_sha256_mismatch")
            base = json.loads(
                (root / "data" / "annotations" / f"{sample_id}.json").read_text(encoding="utf-8")
            )["annotation"]
            if record.get("existing_annotation_sha256") != object_sha256(base):
                raise ValueError("existing_annotation_sha256_mismatch")
            response_hash = str(record.get("raw_response_sha256") or "")
            if len(response_hash) != 64:
                raise ValueError("raw_response_sha256_invalid")
            leaked = sorted(forbidden_keys(record))
            if leaked:
                raise ValueError(f"forbidden_reasoning_keys:{leaked}")
            confidences.append(float(supplement["confidence"]))
            for name, text in supplement["captions"].items():
                caption_words[name].append(word_count(text))
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}:{exc}"})

    failure_files = sorted(
        path.stem for path in (root / "data_v2" / "moss_failures").glob("*.json")
    )
    if failure_files:
        errors.append({"sample_id": "*", "reason": f"unresolved_failure_files:{failure_files}"})
    report = {
        "status": "pass" if not errors and len(confidences) == len(rows) else "failed",
        "records_expected": len(rows),
        "records_valid": len(confidences),
        "model_revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "confidence": {
            "min": min(confidences, default=None),
            "max": max(confidences, default=None),
            "mean": round(mean(confidences), 4) if confidences else None,
        },
        "caption_word_counts": {
            name: {
                "min": min(values, default=None),
                "max": max(values, default=None),
                "mean": round(mean(values), 2) if values else None,
            }
            for name, values in caption_words.items()
        },
        "raw_model_responses_stored": False,
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "moss_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

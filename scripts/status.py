#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report_status(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")
    except Exception:
        return "invalid_json"


audio = read_jsonl(ROOT / "data" / "audio_manifest.jsonl")
vocal = read_jsonl(ROOT / "data" / "vocal_manifest.jsonl")
training = read_jsonl(ROOT / "data" / "training_audio_manifest.jsonl")
training_rejected = read_jsonl(ROOT / "data" / "training_audio_rejected.jsonl")
mir_files = list((ROOT / "data" / "mir").glob("*.json"))
mir = []
for path in mir_files:
    try:
        mir.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        mir.append({"analysis_status": "invalid_json"})
annotation_state = read_jsonl(ROOT / "data" / "annotation_manifest.jsonl")
annotations = list((ROOT / "data" / "annotations").glob("*.json"))
final_audio = list((ROOT / "data" / "final_dataset").rglob("*.flac"))
tensors = list((ROOT / "data" / "tensors_all").rglob("*.pt"))
print(json.dumps({
    "audio_records": len(audio),
    "audio_valid": sum(bool(r.get("audio_valid")) for r in audio),
    "audio_rejected": sum(not bool(r.get("audio_valid")) for r in audio),
    "deduplication_performed": False,
    "vocal_records": len(vocal),
    "vocal_status": dict(Counter(r.get("vocal_status", "") for r in vocal)),
    "training_records": len(training),
    "training_sources": dict(Counter(r.get("audio_source", "") for r in training)),
    "training_rejected": len(training_rejected),
    "mir_files": len(mir_files),
    "mir_status": dict(Counter(r.get("analysis_status", "") for r in mir)),
    "annotation_state_records": len(annotation_state),
    "annotation_status": dict(Counter(r.get("annotation_status", "") for r in annotation_state)),
    "annotation_cost_usd": round(sum(
        float((r.get("annotation_usage") or {}).get("cost") or 0) for r in annotation_state
    ), 6),
    "annotation_sanitization_actions": dict(Counter(
        action.get("action", "")
        for row in annotation_state
        for action in (row.get("annotation_sanitization") or [])
    )),
    "annotation_files": len(annotations),
    "final_audio": len(final_audio),
    "tensors": len(tensors),
    "gates": {
        "mir": report_status(ROOT / "data" / "mir_validation_report.json"),
        "annotations": report_status(ROOT / "data" / "annotation_validation_report.json"),
        "dataset": report_status(ROOT / "data" / "final_validation_report.json"),
        "tensors": report_status(ROOT / "data" / "tensor_validation_report.json"),
        "smoke": report_status(ROOT / "outputs" / "smoke" / "smoke_validation_report.json"),
        "training": report_status(ROOT / "outputs" / "training" / "melodic-edm-core-v1" / "training_validation_report.json"),
        "evaluation": report_status(ROOT / "outputs" / "inference" / "checkpoint_comparison" / "evaluation_report.json"),
        "release": report_status(ROOT / "outputs" / "release" / "melodic-edm-core-v1" / "release_report.json"),
        "release_verify": report_status(ROOT / "outputs" / "release_verify" / "generated" / "inference_report.json"),
    },
}, ensure_ascii=False, indent=2))

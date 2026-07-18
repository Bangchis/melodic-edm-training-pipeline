#!/usr/bin/env python3
"""Validate ACE-Step tensors against the record-preserving final manifest."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_TENSORS = {
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def expected_by_split(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result = {"train": set(), "validation": set()}
    for row in rows:
        split = row.get("split")
        if split not in result:
            raise ValueError(f"invalid_split:{split}")
        sid = str(row["sample_id"])
        if sid in result["train"] or sid in result["validation"]:
            raise ValueError(f"duplicate_sample_id:{sid}")
        result[split].add(sid)
    return result


def finite_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor) and value.numel() > 0 and bool(torch.isfinite(value).all())


def validate_one(path: Path, sample_id: str) -> list[str]:
    import torch

    errors: list[str] = []
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        return [f"load_failed:{type(exc).__name__}:{exc}"]
    if not isinstance(data, dict):
        return ["root_not_dict"]
    missing = sorted(REQUIRED_TENSORS - set(data))
    if missing:
        errors.append("missing_keys:" + ",".join(missing))
    for key in sorted(REQUIRED_TENSORS & set(data)):
        if not finite_tensor(data[key]):
            errors.append(f"nonfinite_or_empty:{key}")
    states = data.get("encoder_hidden_states_variants")
    masks = data.get("encoder_attention_mask_variants")
    if not isinstance(states, list) or len(states) != 4:
        errors.append("encoder_variants_not_four")
    elif any(not finite_tensor(value) for value in states):
        errors.append("encoder_variant_nonfinite_or_empty")
    if not isinstance(masks, list) or len(masks) != 4:
        errors.append("encoder_masks_not_four")
    elif any(not finite_tensor(value) for value in masks):
        errors.append("encoder_mask_nonfinite_or_empty")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata_missing")
    else:
        if Path(str(metadata.get("filename", ""))).stem != sample_id:
            errors.append("metadata_filename_mismatch")
        variants = metadata.get("caption_variants")
        if not isinstance(variants, list) or len(variants) != 4 or any(not str(x).strip() for x in variants):
            errors.append("metadata_caption_variants_not_four")
        if not str(metadata.get("caption", "")).strip():
            errors.append("metadata_caption_empty")
        if not bool(metadata.get("is_instrumental")):
            errors.append("metadata_not_instrumental")
        duration = metadata.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) <= 0:
            errors.append("metadata_duration_invalid")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/final_manifest.jsonl")
    parser.add_argument("--train-dir", default="data/tensors_all")
    parser.add_argument("--validation-dir", default="data/tensors_validation")
    parser.add_argument("--report", default="data/tensor_validation_report.json")
    parser.add_argument("--expected-total", type=int, default=231)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    rows = read_jsonl(root / args.manifest)
    errors: list[dict[str, Any]] = []
    try:
        expected = expected_by_split(rows)
    except Exception as exc:
        expected = {"train": set(), "validation": set()}
        errors.append({"reason": str(exc)})

    directories = {
        "train": root / args.train_dir,
        "validation": root / args.validation_dir,
    }
    actual: dict[str, set[str]] = {}
    temporary_files = []
    for split, directory in directories.items():
        actual[split] = {path.stem for path in directory.glob("*.pt") if not path.name.endswith(".tmp.pt")}
        temporary_files.extend(str(path) for path in directory.glob("*.tmp.pt"))
        missing = sorted(expected[split] - actual[split])
        extra = sorted(actual[split] - expected[split])
        if missing or extra:
            errors.append({"split": split, "reason": "membership_mismatch", "missing": missing, "extra": extra})

    overlap = sorted(actual.get("train", set()) & actual.get("validation", set()))
    if overlap:
        errors.append({"reason": "train_validation_overlap", "sample_ids": overlap})
    if temporary_files:
        errors.append({"reason": "temporary_tensors_remaining", "paths": temporary_files})

    tensor_errors = []
    for split, stems in actual.items():
        directory = directories[split]
        for index, sid in enumerate(sorted(stems), 1):
            found = validate_one(directory / f"{sid}.pt", sid)
            if found:
                tensor_errors.append({"split": split, "sample_id": sid, "errors": found})
            print(f"[{split} {index}/{len(stems)}] {sid} {'FAIL' if found else 'PASS'}", flush=True)
    errors.extend(tensor_errors)

    raw_counts = Counter(str(row.get("raw_path") or row.get("video_id") or row["sample_id"]) for row in rows)
    total = len(rows)
    if total != args.expected_total:
        errors.append({"reason": "unexpected_total", "expected": args.expected_total, "actual": total})
    report = {
        "status": "pass" if not errors else "failed",
        "manifest_records": total,
        "train_expected": len(expected["train"]),
        "validation_expected": len(expected["validation"]),
        "train_tensors": len(actual.get("train", set())),
        "validation_tensors": len(actual.get("validation", set())),
        "validated_tensors": sum(len(value) for value in actual.values()) - len(tensor_errors),
        "deduplication_performed": False,
        "shared_audio_records_preserved": sum(count - 1 for count in raw_counts.values()),
        "errors": errors,
    }
    atomic_json(root / args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

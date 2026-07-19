#!/usr/bin/env python3
"""Build deterministic, storage-free tensor views for unique V2 audio content."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_common import atomic_json, read_jsonl


VIEW_NAMES = {
    "train": "tensors_train_unique",
    "validation": "tensors_validation_unique",
    "all": "tensors_all_unique",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"training audio is missing: {path}")
    return path.resolve()


def select_unique_records(root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministic representatives and a complete duplicate-group audit."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        audio_hash = sha256_file(resolve_audio(root, str(row["final_audio_path"])))
        grouped[audio_hash].append(row)
        if index % 20 == 0 or index == len(rows):
            print(f"hashed audio {index}/{len(rows)}", flush=True)

    representatives: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for audio_hash, members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: str(item["sample_id"]))
        splits = sorted({str(item["split"]) for item in members})
        if len(splits) != 1:
            raise ValueError(
                f"identical audio crosses train/validation: {audio_hash} "
                f"{[item['sample_id'] for item in members]}"
            )
        representative = members[0]
        representatives.append({
            "audio_sha256": audio_hash,
            "sample_id": str(representative["sample_id"]),
            "split": splits[0],
        })
        groups.append({
            "audio_sha256": audio_hash,
            "split": splits[0],
            "representative_sample_id": str(representative["sample_id"]),
            "sample_ids": [str(item["sample_id"]) for item in members],
            "records": len(members),
            "redundant_records": len(members) - 1,
        })
    representatives.sort(key=lambda item: item["sample_id"])
    groups.sort(key=lambda item: item["representative_sample_id"])
    return representatives, groups


def replace_view(root: Path, view_name: str, source_name: str, sample_ids: list[str], timestamp: str) -> None:
    data_root = root / "data_v2"
    source = data_root / source_name
    staging = data_root / f".{view_name}.staging-{timestamp}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for sample_id in sample_ids:
            tensor = source / f"{sample_id}.pt"
            if not tensor.is_file():
                raise FileNotFoundError(f"representative tensor is missing: {tensor}")
            os.link(tensor, staging / tensor.name)
        atomic_json(staging / "manifest.json", {"samples": [f"{sample_id}.pt" for sample_id in sample_ids]})
        destination = data_root / view_name
        if destination.exists():
            backup_root = data_root / "dedup_view_backups" / timestamp
            backup_root.mkdir(parents=True, exist_ok=True)
            os.replace(destination, backup_root / view_name)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    if len(rows) != 231:
        raise ValueError(f"expected 231 catalog records, found {len(rows)}")

    representatives, groups = select_unique_records(root, rows)
    by_split = {
        split: [item["sample_id"] for item in representatives if split == "all" or item["split"] == split]
        for split in VIEW_NAMES
    }
    expected = {"train": 184, "validation": 33, "all": 217}
    observed = {name: len(values) for name, values in by_split.items()}
    if observed != expected:
        raise ValueError(f"unexpected unique-audio counts: {observed}, expected {expected}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sources = {"train": "tensors_train", "validation": "tensors_validation", "all": "tensors_all"}
    for split, view_name in VIEW_NAMES.items():
        replace_view(root, view_name, sources[split], by_split[split], timestamp)

    duplicate_groups = [group for group in groups if group["records"] > 1]
    report = {
        "status": "pass",
        "catalog_records": len(rows),
        "unique_audio_records": len(representatives),
        "train_unique_tensors": observed["train"],
        "validation_unique_tensors": observed["validation"],
        "all_unique_tensors": observed["all"],
        "redundant_records_excluded_from_training": len(rows) - len(representatives),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": 0,
        "representative_selection": "lexicographically_smallest_sample_id_per_exact_audio_sha256",
        "storage": "hardlinks_to_full_231_record_tensor_set",
        "representatives": representatives,
        "groups": duplicate_groups,
    }
    atomic_json(root / "data_v2" / "dedup_training_view_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage every V2 train/validation FLAC for a private Hugging Face dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from v2_common import atomic_json, read_jsonl


SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[oprsu]_[A-Za-z0-9]{20,}"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not os.path.samefile(source, destination):
            raise FileExistsError(f"unrelated staged audio exists: {destination}")
        return
    os.link(source, destination)


def public_row(row: dict[str, Any], repo_path: str, digest: str, size: int) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "record_key": row["record_key"],
        "split": row["split"],
        "parent_song_id": row["parent_song_id"],
        "expected_artist": row.get("expected_artist"),
        "expected_title": row.get("expected_title"),
        "expected_version": row.get("expected_version"),
        "video_id": row.get("video_id"),
        "audio": repo_path,
        "audio_sha256": digest,
        "audio_bytes": size,
        "duration_seconds": row.get("final_duration") or row.get("duration"),
        "sample_rate": row.get("sample_rate"),
        "channels": row.get("channels"),
        "caption_types": row.get("caption_types"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-audio-v2")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    final_report = root / "outputs" / "v2" / "final-all-data" / "final_validation_report.json"
    if not final_report.is_file() or json.loads(final_report.read_text()).get("status") != "pass":
        raise RuntimeError("final-all-data training gate has not passed")

    source_rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    split_counts = Counter(str(row.get("split")) for row in source_rows)
    if len(source_rows) != 231 or split_counts != Counter({"train": 196, "validation": 35}):
        raise RuntimeError(f"expected 231 records split 196/35, got {len(source_rows)} {split_counts}")
    sample_ids = [str(row.get("sample_id")) for row in source_rows]
    if len(set(sample_ids)) != 231:
        raise RuntimeError("sample_id values are not unique")

    staging = root / "outputs" / "v2" / "hf-audio-dataset-v2"
    staging.mkdir(parents=True, exist_ok=True)
    staged_rows: list[dict[str, Any]] = []
    hash_counts: Counter[str] = Counter()
    total_bytes = 0
    for index, row in enumerate(sorted(source_rows, key=lambda value: value["sample_id"]), 1):
        split = row["split"]
        source = Path(row["final_audio_path"]).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file() or source.is_symlink():
            raise RuntimeError(f"unsafe final audio path: {source}")
        if source.suffix.casefold() != ".flac":
            raise RuntimeError(f"non-FLAC training audio: {source}")
        repo_path = f"audio/{split}/{row['sample_id']}.flac"
        destination = staging / repo_path
        ensure_hardlink(source, destination)
        digest = sha256(source)
        size = source.stat().st_size
        hash_counts[digest] += 1
        total_bytes += size
        staged_rows.append(public_row(row, repo_path, digest, size))
        if index % 25 == 0 or index == len(source_rows):
            print(f"[{index}/{len(source_rows)}] staged and hashed", flush=True)

    metadata = staging / "metadata"
    for split in ("train", "validation"):
        rows = [row for row in staged_rows if row["split"] == split]
        atomic_text(
            metadata / f"{split}.jsonl",
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        )
    atomic_text(
        metadata / "manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in staged_rows),
    )
    dataset_report = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "records": 231,
        "train_records": 196,
        "validation_records": 35,
        "audio_format": "flac",
        "audio_bytes": total_bytes,
        "unique_audio_hashes": len(hash_counts),
        "duplicate_content_records_retained": sum(count - 1 for count in hash_counts.values()),
        "deduplication_performed": False,
    }
    atomic_json(metadata / "dataset_report.json", dataset_report)
    card = """---
pretty_name: Melodic EDM Audio V2
configs:
- config_name: default
  data_files:
  - split: train
    path: metadata/train.jsonl
  - split: validation
    path: metadata/validation.jsonl
---

# Melodic EDM Audio V2

Private research backup of the exact 231 FLAC records used by Melodic EDM Core V2.
The grouped split contains 196 train and 35 validation records. Every catalog record
is retained as its own file; repeated content is not removed or down-weighted.

`metadata/manifest.jsonl` records the split, source identity, duration, byte size and
SHA-256 digest for every audio file. `SHA256SUMS` covers all staged release files.
Access does not grant redistribution rights; users remain responsible for source rights.
"""
    atomic_text(staging / "README.md", card)
    atomic_text(staging / ".gitattributes", "*.flac filter=lfs diff=lfs merge=lfs -text\n")

    findings = []
    for path in (staging / "metadata").rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"file": str(path.relative_to(staging)), "pattern": pattern.pattern.decode()})
    if findings:
        raise RuntimeError(f"audio dataset secret scan failed: {findings}")

    checksums = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file() and ".cache" not in item.parts):
        if path.name == "SHA256SUMS":
            continue
        checksums.append(f"{sha256(path)}  {path.relative_to(staging)}")
    atomic_text(staging / "SHA256SUMS", "\n".join(checksums) + "\n")

    prepare_report = {
        **dataset_report,
        "staging_root": str(staging),
        "staged_files": len([path for path in staging.rglob("*") if path.is_file() and ".cache" not in path.parts]),
        "manifest_sha256": sha256(metadata / "manifest.jsonl"),
        "checksums_sha256": sha256(staging / "SHA256SUMS"),
        "secret_scan_findings": 0,
        "hardlink_staging": True,
    }
    atomic_json(root / "outputs" / "v2" / "audio_dataset_prepare_report.json", prepare_report)
    print(json.dumps(prepare_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

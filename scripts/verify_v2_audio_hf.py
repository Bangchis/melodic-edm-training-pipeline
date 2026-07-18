#!/usr/bin/env python3
"""Clean-download and SHA-256 verify all 231 private HF audio records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from v2_common import atomic_json, read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "audio_dataset_clean_verification_report.json"
    upload = json.loads(
        (root / "outputs" / "v2" / "audio_dataset_upload_report.json").read_text(encoding="utf-8")
    )
    if upload.get("status") != "pass" or not upload.get("sha"):
        raise RuntimeError("audio dataset upload gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token missing")
    free_gib = shutil.disk_usage("/workspace").free / 1024**3
    if free_gib < 20:
        raise RuntimeError(f"at least 20 GiB free required for clean audio verification, found {free_gib:.1f}")

    from huggingface_hub import snapshot_download

    clean_root = Path(tempfile.mkdtemp(prefix="melodic-edm-audio-v2-verify.", dir="/workspace"))
    errors: list[str] = []
    checked = 0
    checksum_mismatches = 0
    split_counts: Counter[str] = Counter()
    try:
        snapshot_download(
            repo_id=upload["repo_id"],
            repo_type="dataset",
            revision=upload["sha"],
            local_dir=clean_root,
            token=token,
            force_download=True,
            max_workers=8,
        )
        manifest_path = clean_root / "metadata" / "manifest.jsonl"
        if sha256(manifest_path) != upload["manifest_sha256"]:
            errors.append("clean_manifest_sha256_mismatch")
        rows = read_jsonl(manifest_path)
        split_counts.update(str(row.get("split")) for row in rows)
        if len(rows) != 231 or split_counts != Counter({"train": 196, "validation": 35}):
            errors.append(f"clean_manifest_counts_invalid:{len(rows)}:{dict(split_counts)}")
        for index, row in enumerate(rows, 1):
            audio = clean_root / row["audio"]
            if not audio.is_file():
                errors.append(f"clean_audio_missing:{row['audio']}")
                continue
            if audio.stat().st_size != int(row["audio_bytes"]):
                errors.append(f"clean_audio_size_mismatch:{row['audio']}")
                continue
            if sha256(audio) != row["audio_sha256"]:
                checksum_mismatches += 1
                errors.append(f"clean_audio_sha256_mismatch:{row['audio']}")
                continue
            checked += 1
            if index % 25 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] clean audio verified", flush=True)
        downloaded_audio = list((clean_root / "audio").rglob("*.flac"))
        if len(downloaded_audio) != 231:
            errors.append(f"clean_audio_file_count_invalid:{len(downloaded_audio)}")
    except Exception as exc:  # report a resumable external failure without leaking tokens
        errors.append(f"clean_download_or_verify_exception:{type(exc).__name__}:{exc}")
    finally:
        shutil.rmtree(clean_root, ignore_errors=True)

    result = {
        "status": "pass" if not errors else "failed",
        "repo_id": upload.get("repo_id"),
        "verified_revision": upload.get("sha"),
        "records": checked,
        "train_records": split_counts.get("train", 0),
        "validation_records": split_counts.get("validation", 0),
        "checksum_mismatches": checksum_mismatches,
        "manifest_sha256": upload.get("manifest_sha256"),
        "force_download": True,
        "clean_download_removed_after_verification": not clean_root.exists(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

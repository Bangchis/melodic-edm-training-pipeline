#!/usr/bin/env python3
"""Upload a secret-free resumable V2 annotation checkpoint to private HF."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from v2_common import atomic_json


SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh(?:p|o|u|s|r)_[A-Za-z0-9_]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(root: Path) -> list[Path]:
    data = root / "data_v2"
    files: list[Path] = []
    for directory in ("moss_annotations", "claim_consensus", "caption_repairs"):
        files.extend(sorted((data / directory).glob("*.json")))
    for name in (
        "dataset_train.json",
        "dataset_validation.json",
        "dataset_all.json",
        "moss_validation_report.json",
        "annotation_merge_report.json",
        "dataset_build_report.json",
        "downstream_reset_report.json",
    ):
        path = data / name
        if path.is_file():
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--repo-id", default="Bangchis/melodic-edm-core-v2-r32-experimental"
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    files = selected_files(root)
    if not files:
        raise RuntimeError("no V2 annotation files found")
    findings = []
    for path in files:
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append(str(path.relative_to(root)))
    if findings:
        raise RuntimeError(f"secret scan failed: {sorted(set(findings))}")

    counts = {
        directory: sum(1 for path in files if path.parent.name == directory)
        for directory in ("moss_annotations", "claim_consensus", "caption_repairs")
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"v2-annotation-wip-{timestamp}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="v2-annotation-wip-", dir=root / "outputs") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / name
        manifest = {
            "status": "in_progress",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "files": len(files),
            "raw_model_responses_included": False,
            "audio_included": False,
            "tokens_included": False,
        }
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            info = tarfile.TarInfo("work-in-progress-manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(manifest_bytes))
        digest = sha256(archive)
        checksum = temporary_root / f"{name}.sha256"
        checksum.write_text(f"{digest}  {name}\n", encoding="utf-8")
        prefix = "work-in-progress/annotation"
        api = HfApi(token=token)
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Checkpoint V2 annotation progress {counts['claim_consensus']}/231",
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(archive)
                ),
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/{name}.sha256",
                    path_or_fileobj=str(checksum),
                ),
            ],
        )
        revision = str(commit.oid)
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="model",
                filename=f"{prefix}/{name}",
                revision=revision,
                token=token,
            )
        )
        verified = sha256(downloaded) == digest

    report = {
        "status": "pass" if verified else "failed",
        "repo_id": args.repo_id,
        "private": True,
        "revision": revision,
        "archive": f"{prefix}/{name}",
        "sha256": digest,
        "counts": counts,
        "files": len(files) + 1,
        "secret_scan_findings": len(findings),
        "clean_download_hash_match": verified,
    }
    atomic_json(root / "outputs" / "v2" / "annotation_wip_upload_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())

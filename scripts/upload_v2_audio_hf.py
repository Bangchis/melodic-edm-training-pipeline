#!/usr/bin/env python3
"""Resumably upload the complete V2 audio dataset to a private HF dataset repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-audio-v2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    report_path = root / "outputs" / "v2" / "audio_dataset_upload_report.json"
    prepare_path = root / "outputs" / "v2" / "audio_dataset_prepare_report.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare.get("status") != "pass" or prepare.get("records") != 231:
        raise RuntimeError("audio dataset preparation gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token missing")
    staging = Path(prepare["staging_root"])
    if sha256(staging / "metadata" / "manifest.jsonl") != prepare["manifest_sha256"]:
        raise RuntimeError("staged manifest changed after preparation")

    from huggingface_hub import HfApi, RepoFile, hf_hub_download

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=staging,
        private=True,
        ignore_patterns=[".cache/**"],
        num_workers=max(1, args.workers),
        print_report=True,
        print_report_every=30,
    )
    info = api.repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    tree = {
        item.path: item
        for item in api.list_repo_tree(
            args.repo_id, repo_type="dataset", recursive=True, expand=True, revision=info.sha
        )
        if isinstance(item, RepoFile)
    }
    manifest_rows = read_jsonl(staging / "metadata" / "manifest.jsonl")
    errors: list[str] = []
    for row in manifest_rows:
        path = row["audio"]
        remote = tree.get(path)
        if remote is None:
            errors.append(f"remote_audio_missing:{path}")
            continue
        if int(remote.size or 0) != int(row["audio_bytes"]):
            errors.append(f"remote_audio_size_mismatch:{path}:{remote.size}:{row['audio_bytes']}")
    remote_manifest = Path(hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename="metadata/manifest.jsonl",
        revision=info.sha,
        token=token,
    ))
    if sha256(remote_manifest) != prepare["manifest_sha256"]:
        errors.append("remote_manifest_sha256_mismatch")
    if info.private is not True:
        errors.append("audio_dataset_repo_is_not_private")
    result = {
        "status": "pass" if not errors else "failed",
        "repo_id": args.repo_id,
        "private": info.private,
        "sha": info.sha,
        "records": len(manifest_rows),
        "train_records": prepare["train_records"],
        "validation_records": prepare["validation_records"],
        "audio_bytes": prepare["audio_bytes"],
        "remote_files": len(tree),
        "manifest_sha256": prepare["manifest_sha256"],
        "resumable_upload_cache": str(staging / ".cache" / "huggingface"),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    atomic_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

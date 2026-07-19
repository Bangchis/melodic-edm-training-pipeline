#!/usr/bin/env python3
"""Back up merged v2 annotations and schemas to a private dataset repo."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from v2_common import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-training-metadata-v2")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    merge_report = json.loads(
        (root / "data_v2" / "annotation_merge_report.json").read_text(encoding="utf-8")
    )
    if merge_report.get("status") != "pass":
        raise RuntimeError("v2 annotation merge gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    commits = []
    for folder, remote in (
        (root / "data_v2" / "annotations", "annotations"),
        (root / "data_v2" / "moss_annotations", "moss_supplements"),
        (root / "configs" / "v2", "configs"),
    ):
        info = api.upload_folder(
            folder_path=str(folder),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload v2 {remote}",
        )
        commits.append(str(info.oid))
    for name in ("manifest.jsonl", "annotation_merge_report.json", "dataset_build_report.json"):
        path = root / "data_v2" / name
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload v2 {name}",
        )
        commits.append(str(info.oid))
    report = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "records": merge_report["records_merged"],
        "commits": commits,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "audio_uploaded": False,
    }
    atomic_json(root / "data_v2" / "metadata_upload_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

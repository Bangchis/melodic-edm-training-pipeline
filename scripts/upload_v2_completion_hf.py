#!/usr/bin/env python3
"""Back up final V2 acceptance and clean-inference evidence to private Hub."""
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
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2-training")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    acceptance = json.loads(
        (root / "outputs" / "v2" / "final_acceptance_report.json").read_text(encoding="utf-8")
    )
    if acceptance.get("status") != "pass":
        raise RuntimeError("final V2 acceptance gate has not passed")
    release = root / "outputs" / "release" / "melodic-edm-core-v2"
    files = (
        (root / "outputs" / "v2" / "final_acceptance_report.json", "completion/final_acceptance_report.json"),
        (release / "clean_verification_report.json", "completion/clean_verification_report.json"),
        (release / "upload_report.json", "completion/final_model_upload_report.json"),
        (root / "outputs" / "v2" / "final_plan.json", "completion/final_plan.json"),
        (root / "outputs" / "v2" / "final-all-data" / "final_validation_report.json", "completion/final_validation_report.json"),
        (root / "outputs" / "v2" / "final-all-data" / "evaluation" / "generation_report.json", "completion/final_generation_report.json"),
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commits = []
    for path, remote in files:
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Back up V2 completion evidence {Path(remote).name}",
        )
        commits.append(str(info.oid))
    report = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "files_uploaded": len(files),
        "commits": commits,
        "final_model_revision": acceptance["hugging_face_model_revision"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(root / "outputs" / "v2" / "completion_upload_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

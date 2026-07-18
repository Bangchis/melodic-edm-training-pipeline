#!/usr/bin/env python3
"""Upload the validated V2 release to a private Hugging Face model repo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    release = root / "outputs" / "release" / "melodic-edm-core-v2"
    report = json.loads((release / "release_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("secret_scan_findings") != 0:
        raise SystemExit("V2 release validation gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF token missing")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(release),
        commit_message="Publish validated Melodic EDM Core V2 adapters",
    )
    info = api.repo_info(args.repo_id, repo_type="model")
    result = {
        "status": "pass" if info.private else "failed",
        "repo_id": args.repo_id,
        "private": info.private,
        "sha": info.sha,
        "files": len(info.siblings),
        "commit_url": str(commit),
    }
    output = release / "upload_report.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload the validation-selected preview to the private V2 model repository."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from v2_common import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    preview = root / "outputs" / "release" / "melodic-edm-core-v2-preview"
    package = json.loads((preview / "preview_report.json").read_text(encoding="utf-8"))
    if package.get("status") != "pass" or package.get("secret_scan_findings") != 0:
        raise RuntimeError("V2 preview package gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token missing")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(preview),
        commit_message="Publish V2 validation-selected inference preview",
    )
    info = api.repo_info(args.repo_id, repo_type="model")
    siblings = {item.rfilename for item in info.siblings}
    required = {
        "best-val/adapter_config.json",
        "best-val/adapter_model.safetensors",
        "notebooks/melodic_edm_core_v2_colab.ipynb",
        "SHA256SUMS",
    }
    errors = []
    if not info.private:
        errors.append("preview_repo_is_not_private")
    if missing := sorted(required - siblings):
        errors.append(f"preview_remote_files_missing:{missing}")
    result = {
        "status": "pass" if not errors else "failed",
        "repo_id": args.repo_id,
        "private": info.private,
        "sha": info.sha,
        "files": len(siblings),
        "commit_url": str(commit),
        "stage": "best-val-preview-before-final-all-data",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    atomic_json(
        root / "outputs" / "v2" / "checkpoint-evaluation" / "preview_upload_report.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

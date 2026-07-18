#!/usr/bin/env python3
"""Upload the clearly labelled R32 experimental preview to a private HF model repo."""
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
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2-r32-experimental")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    package = root / "outputs" / "release" / "melodic-edm-core-v2-r32-experimental"
    report = json.loads((package / "package_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("secret_scan_findings") != 0:
        raise RuntimeError("experimental package gate has not passed")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token missing")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(package),
        commit_message="Publish measured R32 experimental Colab preview (6/15)",
    )
    info = api.repo_info(args.repo_id, repo_type="model")
    siblings = {item.rfilename for item in info.siblings}
    required = {
        "experimental-r32/adapter_config.json",
        "experimental-r32/adapter_model.safetensors",
        "notebooks/melodic_edm_core_v2_colab.ipynb",
        "scripts/infer_v2_release.py",
        "reports/paired_base_lora_comparison.json",
        "SHA256SUMS",
    }
    errors = []
    if not info.private:
        errors.append("experimental_repo_is_not_private")
    if missing := sorted(required - siblings):
        errors.append(f"remote_files_missing:{missing}")
    result = {
        "status": "pass" if not errors else "failed",
        "repo_id": args.repo_id,
        "private": info.private,
        "sha": info.sha,
        "files": len(siblings),
        "commit_url": str(commit),
        "stage": "r32_experimental_not_final",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    atomic_json(package / "upload_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Clean-redownload an immutable V2 release and verify final adapter inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from v2_common import atomic_json


def sha256(path: Path) -> str:
    """Return one file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    clean_root = Path(tempfile.mkdtemp(prefix="melodic-edm-v2-redownload.", dir="/workspace"))
    downloaded = clean_root / "release"
    generated = clean_root / "generated"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "download_and_infer_v2.py"),
            "--repo-id", args.repo_id,
            "--revision", args.revision,
            "--download-dir", str(downloaded),
            "--ace-root", str(root / "vendor" / "ACE-Step-1.5-v2"),
            "--checkpoint-root", str(root / "checkpoints"),
            "--adapter-subdirectory", "final-all-data",
            "--prompt-index", "0",
            "--output-dir", str(generated),
        ],
        check=True,
        env={**os.environ, "HF_TOKEN": token},
    )
    inference = json.loads((generated / "inference_report.json").read_text(encoding="utf-8"))
    packaged_adapter = (
        root / "outputs" / "release" / "melodic-edm-core-v2"
        / "final-all-data" / "adapter_model.safetensors"
    )
    downloaded_adapter = downloaded / "final-all-data" / "adapter_model.safetensors"
    packaged_hash = sha256(packaged_adapter)
    downloaded_hash = sha256(downloaded_adapter)
    errors = []
    if packaged_hash != downloaded_hash:
        errors.append("downloaded_final_adapter_hash_mismatch")
    if inference.get("status") != "pass":
        errors.append("clean_inference_failed")
    probe = inference.get("probe", {})
    if probe.get("sample_rate") != 48000 or probe.get("channels") != 2:
        errors.append("clean_inference_format_invalid")
    report = {
        "status": "pass" if not errors else "failed",
        "repo_id": args.repo_id,
        "verified_revision": args.revision,
        "clean_download_root": str(clean_root),
        "sha256sums_verified_by_downloader": True,
        "packaged_final_adapter_sha256": packaged_hash,
        "downloaded_final_adapter_sha256": downloaded_hash,
        "adapter_hash_match": packaged_hash == downloaded_hash,
        "inference": inference,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    output = (
        root / "outputs" / "release" / "melodic-edm-core-v2"
        / "clean_verification_report.json"
    )
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

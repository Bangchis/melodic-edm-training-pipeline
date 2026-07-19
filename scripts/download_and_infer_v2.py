#!/usr/bin/env python3
"""Download an immutable private v2 release, verify it, and infer once."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA")
    parser.add_argument("--download-dir", default="./melodic-edm-core-v2-clean")
    parser.add_argument("--ace-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--adapter-subdirectory",
        choices=("experimental-r32", "best-val", "final-all-data"),
        default="final-all-data",
    )
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-dir", default="./generated-v2-clean")
    parser.add_argument("--offload-to-cpu", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for the private model")

    from huggingface_hub import snapshot_download

    release = Path(args.download_dir).resolve()
    if release.exists() and any(release.iterdir()):
        raise FileExistsError(f"clean download directory is not empty: {release}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=token,
        local_dir=release,
    )
    checksum = release / "SHA256SUMS"
    if not checksum.is_file():
        raise RuntimeError("downloaded release is missing SHA256SUMS")
    subprocess.run(["sha256sum", "-c", checksum.name], cwd=release, check=True)

    command = [
        sys.executable,
        str(release / "scripts" / "infer_v2_release.py"),
        "--ace-root", str(Path(args.ace_root).resolve()),
        "--checkpoint-root", str(Path(args.checkpoint_root).resolve()),
        "--release-dir", str(release),
        "--adapter-subdirectory", args.adapter_subdirectory,
        "--prompt-index", str(args.prompt_index),
        "--output-dir", str(Path(args.output_dir).resolve()),
    ]
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

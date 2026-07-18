#!/usr/bin/env python3
"""Download the private release at an immutable revision and run one inference."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v1")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA")
    parser.add_argument("--download-dir", default="./melodic-edm-core-v1")
    parser.add_argument("--ace-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-dir", default="./generated")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for the private model")

    from huggingface_hub import snapshot_download

    release = Path(args.download_dir).resolve()
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

    subprocess.run(
        [
            sys.executable,
            str(release / "infer_release.py"),
            "--ace-root",
            str(Path(args.ace_root).resolve()),
            "--checkpoint-root",
            str(Path(args.checkpoint_root).resolve()),
            "--adapter-dir",
            str(release),
            "--prompt-index",
            str(args.prompt_index),
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

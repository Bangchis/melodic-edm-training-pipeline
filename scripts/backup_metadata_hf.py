#!/usr/bin/env python3
"""Back up text metadata to a private HF dataset without audio, tensors or secrets."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi


SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"gh[oprsu]_[A-Za-z0-9_]{20,}"),
    "openrouter_token": re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(rb"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def selected_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for name in ("README.md", "RUNBOOK.md", "STATUS.md"):
        path = root / name
        if path.is_file():
            files.add(path)
    for pattern in (
        "configs/*.json",
        "data/*.json",
        "data/*.jsonl",
        "data/*.csv",
        "data/mir/*.json",
        "data/mir/allin1/*.json",
        "data/annotations/*.json",
        "data/final_dataset/**/*.json",
        "data/final_dataset/**/*.lyrics.txt",
    ):
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--env-file", default="/workspace/.env")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-training-metadata")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    load_env_file(Path(args.env_file))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF token is missing")
    files = selected_files(root)
    if not files:
        raise SystemExit("no metadata files selected")

    findings = []
    for path in files:
        data = path.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(root)), "rule": name})
    if findings:
        raise SystemExit("secret scan failed: " + json.dumps(findings, ensure_ascii=False))

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=str(path.relative_to(root)), path_or_fileobj=str(path))
        for path in files
    ]
    annotation_count = len(list((root / "data" / "annotations").glob("*.json")))
    mir_count = len(list((root / "data" / "mir").glob("*.json")))
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Checkpoint metadata: MIR {mir_count}, annotations {annotation_count}",
    )
    report = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "uploaded_files": len(files),
        "mir_files": mir_count,
        "annotation_files": annotation_count,
        "secret_scan_findings": 0,
        "commit_url": commit.commit_url,
        "backed_up_at": datetime.now(timezone.utc).isoformat(),
        "excluded": ["audio", "separated stems", "tensors", "model checkpoints", "tokens"],
    }
    atomic_json(root / "data" / "metadata_backup_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

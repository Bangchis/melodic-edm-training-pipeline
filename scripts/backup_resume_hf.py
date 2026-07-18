#!/usr/bin/env python3
"""Back up the final resumable checkpoint to a private Hugging Face dataset."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from latest_checkpoint import epoch_checkpoints  # noqa: E402


SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"gh[oprsu]_[A-Za-z0-9_]{20,}"),
    "openrouter_token": re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(rb"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def inspect_file(path: Path) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    findings: set[str] = set()
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            searchable = overlap + chunk
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(searchable):
                    findings.add(name)
            overlap = searchable[-160:]
    return digest.hexdigest(), sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-training-resume")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = root / "outputs" / "training" / "melodic-edm-core-v1"
    gate_path = output / "training_validation_report.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    if gate.get("status") != "pass":
        raise SystemExit("main training validation gate has not passed")
    checkpoints = epoch_checkpoints(output / "checkpoints")
    if not checkpoints:
        raise SystemExit("no resumable epoch checkpoint found")
    latest_epoch, latest = checkpoints[-1]
    if latest_epoch != int(gate.get("completed_epoch") or 0):
        raise SystemExit("latest checkpoint does not match validated completed epoch")

    sources = {
        "resume/latest/training_state.pt": latest / "training_state.pt",
        "resume/latest/training_state.safetensors": latest / "training_state.safetensors",
        "resume/latest/adapter/adapter_config.json": latest / "adapter" / "adapter_config.json",
        "resume/latest/adapter/adapter_model.safetensors": latest / "adapter" / "adapter_model.safetensors",
        "reports/training_validation_report.json": gate_path,
        "reports/validation_state.json": output / "validation_state.json",
        "logs/training.log": output / "training.log",
        "logs/gpu_metrics.csv": output / "gpu_metrics.csv",
        "configs/train_lora_2x4090.json": root / "configs" / "train_lora_2x4090.json",
        "patches/acestep-xl-validation-caption-variants.patch": root / "patches" / "acestep-xl-validation-caption-variants.patch",
    }
    missing = [name for name, path in sources.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise SystemExit(f"resume backup inputs missing or unsafe: {missing}")

    checksums: dict[str, str] = {}
    findings = []
    for name, path in sources.items():
        digest, rules = inspect_file(path)
        checksums[name] = digest
        findings.extend({"file": name, "rule": rule} for rule in rules)
    if findings:
        raise SystemExit("secret scan failed: " + json.dumps(findings, ensure_ascii=False))

    manifest = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "completed_epoch": latest_epoch,
        "global_step": gate.get("global_step"),
        "best_epoch": gate.get("validation_state", {}).get("best_epoch"),
        "ace_step_commit": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        "deduplication_performed": False,
        "training_records": 231,
        "secret_scan_findings": 0,
        "files": len(sources) + 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "excluded": ["source audio", "separated stems", "preprocessed tensors", "tokens", "cookies"],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    checksums["resume_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())).encode()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF token missing")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in sources.items()
    ]
    operations.extend([
        CommitOperationAdd(path_in_repo="resume_manifest.json", path_or_fileobj=io.BytesIO(manifest_bytes)),
        CommitOperationAdd(path_in_repo="SHA256SUMS", path_or_fileobj=io.BytesIO(checksum_bytes)),
    ])
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Back up resumable epoch {latest_epoch}",
    )
    info = api.repo_info(args.repo_id, repo_type="dataset")
    remote_files = {item.rfilename for item in info.siblings}
    expected = set(sources) | {"resume_manifest.json", "SHA256SUMS"}
    result = {
        **manifest,
        "status": "pass" if info.private and expected <= remote_files else "failed",
        "sha": info.sha,
        "remote_files": len(remote_files),
        "commit_url": commit.commit_url,
    }
    atomic_json(root / "outputs" / "resume_backup" / "upload_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload fixed checkpoint audio and selection evidence to the private training repo."""
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
    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"
    required = {
        name: json.loads((evaluation / name).read_text(encoding="utf-8"))
        for name in ("generation_report.json", "listening_scores.json", "selection.json")
    }
    failed = [name for name, value in required.items() if value.get("status") != "pass"]
    if failed:
        raise RuntimeError(f"evaluation gates have not passed: {failed}")
    audio_root = evaluation / "audio"
    wavs = sorted(audio_root.rglob("*.wav"))
    expected = int(required["generation_report.json"]["expected_outputs"])
    if len(wavs) != expected:
        raise RuntimeError(f"expected {expected} fixed audio examples, found {len(wavs)}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    commits = []
    info = api.upload_folder(
        folder_path=str(audio_root),
        path_in_repo="audio_examples",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload fixed V2 checkpoint audio examples",
    )
    commits.append(str(info.oid))
    for name in required:
        info = api.upload_file(
            path_or_fileobj=str(evaluation / name),
            path_in_repo=f"evaluation/{name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload V2 evaluation {name}",
        )
        commits.append(str(info.oid))
    for path, remote in (
        (root / "configs" / "v2" / "fixed_eval_prompts.json", "evaluation/fixed_eval_prompts.json"),
        (root / "outputs" / "v2" / "train-validation" / "metrics_history.jsonl", "metrics/metrics_history.jsonl"),
        (root / "outputs" / "v2" / "train-validation" / "training_validation_report.json", "metrics/training_validation_report.json"),
    ):
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload V2 evidence {Path(remote).name}",
        )
        commits.append(str(info.oid))
    report = {
        "status": "pass",
        "repo_id": args.repo_id,
        "private": True,
        "audio_examples": len(wavs),
        "checkpoint_count": required["generation_report.json"]["checkpoint_count"],
        "fixed_prompt_count": required["generation_report.json"]["fixed_prompt_count"],
        "selected_checkpoint": required["selection.json"]["selected_checkpoint"],
        "commits": commits,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(evaluation / "hf_evaluation_upload_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

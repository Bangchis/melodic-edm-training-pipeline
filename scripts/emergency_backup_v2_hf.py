#!/usr/bin/env python3
"""Continuously mirror V2 checkpoints, evidence and generated audio to Hugging Face."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from v2_common import atomic_json


def upload_folder_if_present(
    api: HfApi,
    *,
    source: Path,
    remote: str,
    repo_id: str,
    allow_patterns: list[str] | None = None,
) -> str | None:
    if not source.is_dir() or not any(path.is_file() for path in source.rglob("*")):
        return None
    info = api.upload_folder(
        folder_path=str(source),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,
        commit_message=f"Emergency backup {remote}",
    )
    return str(info.oid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2-r32-training")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-minutes", type=int, default=180)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)

    output = root / "outputs" / "v2" / "train-validation"
    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"
    state_path = root / "outputs" / "v2" / "emergency_hf_backup_state.json"
    commits: list[str] = []

    remote_files = set(api.list_repo_files(args.repo_id, repo_type="model"))
    checkpoint_sources = []
    for path in sorted((output / "checkpoints").glob("epoch_*_loss_*")):
        epoch = int(path.name.split("_", 2)[1])
        if epoch % 5 == 0:
            checkpoint_sources.append((path, f"checkpoints/epoch_{epoch:03d}"))
    checkpoint_sources.append((output / "checkpoints" / "best_val", "checkpoints/best_val"))
    for source, remote in checkpoint_sources:
        if not any(name.startswith(f"{remote}/") for name in remote_files):
            commit = upload_folder_if_present(
                api, source=source, remote=remote, repo_id=args.repo_id
            )
            if commit:
                commits.append(commit)

    evidence_files = (
        "metrics_history.jsonl",
        "training_validation_report.json",
        "validation_state.json",
        "training_config.json",
        "prompt_selection_counts.json",
        "gpu_usage.jsonl",
        "hf_sync_state.json",
    )
    for name in evidence_files:
        source = output / name
        if not source.is_file():
            continue
        info = api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=f"emergency/evidence/{name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Emergency backup evidence {name}",
        )
        commits.append(str(info.oid))
    config_commit = upload_folder_if_present(
        api,
        source=root / "configs" / "v2",
        remote="emergency/configs-v2",
        repo_id=args.repo_id,
        allow_patterns=["*.json", "**/*.json"],
    )
    if config_commit:
        commits.append(config_commit)

    for archive in sorted((root / "outputs" / "archive").glob("checkpoint-evaluation-*")):
        commit = upload_folder_if_present(
            api,
            source=archive,
            remote=f"emergency/archived-evaluations/{archive.name}",
            repo_id=args.repo_id,
            allow_patterns=["**/*.wav", "**/*.json"],
        )
        if commit:
            commits.append(commit)

    deadline = time.monotonic() + args.max_minutes * 60
    previous_signature: tuple[object, ...] | None = None
    while time.monotonic() < deadline:
        wavs = sorted((evaluation / "audio").rglob("*.wav")) if evaluation.is_dir() else []
        report_paths = [
            evaluation / "generation_report.json",
            evaluation / "listening_scores.json",
            evaluation / "selection.json",
        ]
        signature = (
            len(wavs),
            *(
                (path.stat().st_size, path.stat().st_mtime_ns)
                if path.is_file() else None
                for path in report_paths
            ),
        )
        if signature != previous_signature:
            commit = upload_folder_if_present(
                api,
                source=evaluation,
                remote="emergency/checkpoint-evaluation",
                repo_id=args.repo_id,
                allow_patterns=["audio/**/*.wav", "*.json"],
            )
            if commit:
                commits.append(commit)
            previous_signature = signature
        generation_path = evaluation / "generation_report.json"
        generation = (
            json.loads(generation_path.read_text(encoding="utf-8"))
            if generation_path.is_file()
            else {}
        )
        expected = int(generation.get("expected_outputs") or 0)
        listening_path = evaluation / "listening_scores.json"
        listening = (
            json.loads(listening_path.read_text(encoding="utf-8"))
            if listening_path.is_file()
            else {}
        )
        selection_path = evaluation / "selection.json"
        selection = (
            json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_path.is_file()
            else {}
        )
        generation_complete = (
            generation.get("status") == "pass" and expected > 0 and len(wavs) == expected
        )
        listening_terminal = listening.get("status") in {"pass", "failed"}
        selection_terminal = selection.get("status") in {"pass", "failed"}
        complete = generation_complete and listening_terminal and selection_terminal
        state = {
            "status": "pass" if complete else "running",
            "repo_id": args.repo_id,
            "checkpoint_prefixes": [remote for _, remote in checkpoint_sources],
            "audio_uploaded": len(wavs),
            "audio_expected": expected or None,
            "listening_status": listening.get("status"),
            "listening_results": len(listening.get("results") or []),
            "selection_status": selection.get("status"),
            "selected_checkpoint": selection.get("selected_checkpoint"),
            "commits": commits,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if complete:
            return 0
        time.sleep(max(5, args.poll_seconds))
    raise TimeoutError("emergency Hugging Face backup watcher timed out")


if __name__ == "__main__":
    raise SystemExit(main())

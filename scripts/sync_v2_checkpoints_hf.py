#!/usr/bin/env python3
"""Upload every fifth v2 checkpoint and current metrics to a private Hub repo."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from v2_common import atomic_json


EPOCH_PATTERN = re.compile(r"epoch_(\d+)_loss_")


def load_state(path: Path) -> dict[str, Any]:
    """Load resumable upload state."""
    if not path.is_file():
        return {"uploaded_epochs": {}, "commits": []}
    return json.loads(path.read_text(encoding="utf-8"))


def discover(checkpoint_root: Path) -> list[tuple[int, Path]]:
    """Return complete checkpoints whose epoch is a multiple of five."""
    output = []
    for path in checkpoint_root.glob("epoch_*_loss_*"):
        match = EPOCH_PATTERN.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch % 5 == 0 and (path / "training_state.pt").is_file():
            output.append((epoch, path))
    return sorted(output)


def upload_metrics(api: HfApi, repo_id: str, output: Path) -> list[str]:
    """Upload current small training-state and metric files."""
    commits = []
    for name in (
        "metrics_history.jsonl",
        "validation_state.json",
        "prompt_selection_counts.json",
        "gpu_metrics.csv",
        "training_validation_report.json",
    ):
        path = output / name
        if not path.is_file():
            continue
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"metrics/{name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Sync v2 metric {name}",
        )
        commits.append(str(info.oid))
    return commits


def sync_once(api: HfApi, repo_id: str, root: Path, state: dict[str, Any]) -> bool:
    """Upload newly completed tenth-epoch checkpoints; return whether state changed."""
    output = root / "outputs" / "v2" / "train-validation"
    changed = False
    for epoch, path in discover(output / "checkpoints"):
        key = str(epoch)
        if key in state["uploaded_epochs"]:
            continue
        info = api.upload_folder(
            folder_path=str(path),
            path_in_repo=f"checkpoints/epoch_{epoch:03d}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload v2 checkpoint epoch {epoch}",
        )
        state["uploaded_epochs"][key] = {
            "source": str(path),
            "commit": str(info.oid),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        state["commits"].append(str(info.oid))
        upload_metrics(api, repo_id, output)
        changed = True
        print(f"uploaded epoch {epoch} commit {info.oid}", flush=True)
    return changed


def sync_best(api: HfApi, repo_id: str, root: Path, state: dict[str, Any]) -> bool:
    """Upload a newly improved best-val checkpoint even between tenth epochs."""
    output = root / "outputs" / "v2" / "train-validation"
    validation_path = output / "validation_state.json"
    best_path = output / "checkpoints" / "best_val"
    state_path = best_path / "training_state.pt"
    if not validation_path.is_file() or not state_path.is_file():
        return False
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    best_epoch = int(validation.get("best_epoch") or 0)
    best_step = int(validation.get("best_optimizer_step") or 0)
    if best_epoch <= 0 or best_step <= 0:
        return False
    uploaded = state.get("uploaded_best") or {}
    if (int(uploaded.get("epoch") or 0), int(uploaded.get("optimizer_step") or 0)) == (
        best_epoch,
        best_step,
    ):
        return False
    info = api.upload_folder(
        folder_path=str(best_path),
        path_in_repo="checkpoints/best_val",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload v2 best-val epoch {best_epoch} step {best_step}",
    )
    # Re-read after upload. If validation improved during transfer, leave the
    # old state unrecorded so the next watch iteration replaces it immediately.
    current = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        int(current.get("best_epoch") or 0) != best_epoch
        or int(current.get("best_optimizer_step") or 0) != best_step
    ):
        print("best-val changed during upload; scheduling immediate replacement", flush=True)
        return True
    state["uploaded_best"] = {
        "epoch": best_epoch,
        "optimizer_step": best_step,
        "source": str(best_path),
        "commit": str(info.oid),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    state["commits"].append(str(info.oid))
    upload_metrics(api, repo_id, output)
    print(f"uploaded best-val epoch {best_epoch} step {best_step} commit {info.oid}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repo-id", default="Bangchis/melodic-edm-core-v2-r32-training")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-idle-minutes", type=int, default=360)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    output = root / "outputs" / "v2" / "train-validation"
    state_path = output / "hf_sync_state.json"
    state = load_state(state_path)
    last_change = time.monotonic()
    while True:
        changed = sync_once(api, args.repo_id, root, state)
        changed = sync_best(api, args.repo_id, root, state) or changed
        if changed:
            last_change = time.monotonic()
            atomic_json(state_path, state)
        gate = output / "training_validation_report.json"
        if gate.is_file() and json.loads(gate.read_text(encoding="utf-8")).get("status") == "pass":
            state["metric_commits"] = upload_metrics(api, args.repo_id, output)
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
            return 0
        if not args.watch:
            atomic_json(state_path, state)
            return 0
        if time.monotonic() - last_change > args.max_idle_minutes * 60:
            raise TimeoutError("checkpoint uploader exceeded idle timeout")
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resume-safe Supervisor orchestrator for the complete V2 server pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ACTIVE_STATES = {"STARTING", "RUNNING", "BACKOFF", "STOPPING"}


def supervisor(*arguments: str) -> str:
    """Run supervisorctl and return its text output."""
    result = subprocess.run(
        ["supervisorctl", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = (result.stdout or result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"supervisorctl {' '.join(arguments)} failed: {output}")
    return output


def state(job: str) -> str:
    """Return one Supervisor process state."""
    output = supervisor("status", job)
    parts = output.split()
    if len(parts) < 2:
        raise RuntimeError(f"cannot parse Supervisor status for {job}: {output}")
    return parts[1]


def wait_for_exit(job: str, poll_seconds: int = 15) -> None:
    """Wait until a job leaves its active states."""
    while True:
        current = state(job)
        if current not in ACTIVE_STATES:
            print(f"[{job}] state={current}", flush=True)
            return
        time.sleep(poll_seconds)


def start(job: str) -> None:
    """Start a stopped job, or leave an already active job untouched."""
    current = state(job)
    if current in ACTIVE_STATES:
        print(f"[{job}] already {current}", flush=True)
        return
    output = supervisor("start", job)
    print(output, flush=True)


def json_pass(path: Path) -> bool:
    """Return whether a JSON gate exists with status=pass."""
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    except (OSError, json.JSONDecodeError):
        return False


def require_json_pass(path: Path, label: str) -> None:
    """Fail the orchestrator if a stage did not produce a passing gate."""
    if not json_pass(path):
        raise RuntimeError(f"{label} did not produce a passing gate: {path}")


def run_stage(job: str, gate: Path, label: str) -> None:
    """Run one resumable stage unless its gate already passes."""
    if json_pass(gate):
        print(f"[{label}] gate already passed; skipping {job}", flush=True)
        return
    start(job)
    wait_for_exit(job)
    require_json_pass(gate, label)
    print(f"[{label}] PASS", flush=True)


def run_parallel(jobs: list[str]) -> None:
    """Start independent jobs and wait for all to leave active states."""
    for job in jobs:
        start(job)
    for job in jobs:
        wait_for_exit(job)


def sync_complete(path: Path) -> bool:
    """Return whether the private checkpoint watcher completed successfully."""
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return bool(value.get("completed_at") and value.get("uploaded_epochs"))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    print("[v2] waiting for both MOSS annotation shards", flush=True)
    wait_for_exit("edm-v2-moss-0")
    wait_for_exit("edm-v2-moss-1")
    run_stage(
        "edm-v2-build-annotations",
        root / "data_v2" / "dataset_build_report.json",
        "annotations-and-dataset",
    )

    tensor_gate = root / "data_v2" / "tensor_validation_report.json"
    metadata_gate = root / "data_v2" / "metadata_upload_report.json"
    if not json_pass(tensor_gate):
        jobs = [
            "edm-v2-preprocess-train-0",
            "edm-v2-preprocess-train-1",
        ]
        if not json_pass(metadata_gate):
            jobs.append("edm-v2-upload-metadata")
        run_parallel(jobs)
        if not json_pass(metadata_gate):
            require_json_pass(metadata_gate, "private-metadata-upload")
        start("edm-v2-preprocess-validation")
        wait_for_exit("edm-v2-preprocess-validation")
        run_stage("edm-v2-merge-tensors", tensor_gate, "merged-tensors")
    else:
        print("[tensors] gate already passed", flush=True)
    require_json_pass(metadata_gate, "private-metadata-upload")

    run_stage(
        "edm-v2-train-smoke",
        root / "outputs" / "v2" / "smoke" / "smoke_validation_report.json",
        "66-step-smoke",
    )

    train_gate = root / "outputs" / "v2" / "train-validation" / "training_validation_report.json"
    sync_gate = root / "outputs" / "v2" / "train-validation" / "hf_sync_state.json"
    if not json_pass(train_gate):
        if not sync_complete(sync_gate):
            start("edm-v2-sync-checkpoints")
        start("edm-v2-train-main")
        wait_for_exit("edm-v2-train-main")
        require_json_pass(train_gate, "train-validation")
        wait_for_exit("edm-v2-sync-checkpoints")
    require_json_pass(train_gate, "train-validation")
    if not sync_complete(sync_gate):
        start("edm-v2-sync-checkpoints")
        wait_for_exit("edm-v2-sync-checkpoints")
    if not sync_complete(sync_gate):
        raise RuntimeError("private checkpoint sync did not complete")

    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"
    run_stage(
        "edm-v2-evaluate-checkpoints",
        evaluation / "generation_report.json",
        "fixed-checkpoint-audio",
    )
    run_stage(
        "edm-v2-score-moss",
        evaluation / "listening_scores.json",
        "checkpoint-listening",
    )
    run_stage(
        "edm-v2-select-checkpoint",
        evaluation / "selection.json",
        "best-step-selection",
    )
    run_stage(
        "edm-v2-upload-evaluation",
        evaluation / "hf_evaluation_upload_report.json",
        "private-evaluation-upload",
    )
    run_stage(
        "edm-v2-train-final",
        root / "outputs" / "v2" / "final-all-data" / "final_validation_report.json",
        "fresh-all-231-training",
    )
    run_stage(
        "edm-v2-evaluate-final",
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "generation_report.json",
        "final-all-data-audio",
    )
    release = root / "outputs" / "release" / "melodic-edm-core-v2"
    run_stage("edm-v2-package-release", release / "release_report.json", "release-package")
    run_stage("edm-v2-upload-model", release / "upload_report.json", "private-model-upload")
    run_stage("edm-v2-verify-release", release / "clean_verification_report.json", "clean-release-verification")
    run_stage(
        "edm-v2-final-audit",
        root / "outputs" / "v2" / "final_acceptance_report.json",
        "final-objective-audit",
    )
    print("[v2] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

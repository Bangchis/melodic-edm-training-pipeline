#!/usr/bin/env python3
"""Resume-safe Supervisor orchestrator for the complete V2 server pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from v2_common import CAPTION_COMPILER_REVISION
except ModuleNotFoundError:  # package import used by local unit tests
    from scripts.v2_common import CAPTION_COMPILER_REVISION

PROMPT_REVISION = "audio-blind-v2.2"


ACTIVE_STATES = {"STARTING", "RUNNING", "BACKOFF", "STOPPING"}


def supervisor(
    *arguments: str,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> str:
    """Run supervisorctl and return its text output."""
    result = subprocess.run(
        ["supervisorctl", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = (result.stdout or result.stderr).strip()
    if result.returncode not in allowed_returncodes:
        raise RuntimeError(f"supervisorctl {' '.join(arguments)} failed: {output}")
    return output


def state(job: str) -> str:
    """Return one Supervisor process state."""
    # supervisorctl intentionally returns 3 when a process is in an expected
    # non-running state such as EXITED or STOPPED.  The status text remains
    # authoritative and must be parsed instead of treating that code as a
    # transport failure.  Code 4 (unknown process/internal error) still fails.
    output = supervisor("status", job, allowed_returncodes=(0, 3))
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


def json_value(path: Path, key: str):
    """Read one top-level JSON value, returning None for missing/invalid data."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, json.JSONDecodeError):
        return None


def archive_stale_training_outputs(root: Path) -> dict[str, object]:
    """Atomically archive every downstream V2 artifact after annotation lineage changes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "outputs" / "archive" / f"v2-before-audio-blind-{timestamp}"
    archived: list[dict[str, str]] = []
    targets = [
        root / "outputs" / "v2",
        root / "outputs" / "release" / "melodic-edm-core-v2-preview",
        root / "outputs" / "release" / "melodic-edm-core-v2",
    ]
    for source in targets:
        if not source.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / source.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite downstream archive {destination}")
        os.replace(source, destination)
        archived.append({"source": str(source), "archive": str(destination)})
    (root / "outputs" / "v2").mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "pass",
        "reason": "annotation_lineage_changed_to_audio-blind-v2.2",
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive) if archived else None,
        "archived": archived,
        "fresh_rank32_outputs_required": True,
    }
    target = root / "data_v2" / "downstream_reset_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return report


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

    moss_gate = root / "data_v2" / "moss_validation_report.json"
    print("[v2] waiting for both MOSS annotation shards", flush=True)
    for repair_round in range(4):
        wait_for_exit("edm-v2-moss-0")
        wait_for_exit("edm-v2-moss-1")
        validation = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "validate_moss_annotations_v2.py"),
                "--project-root", str(root),
            ],
            check=False,
        )
        if validation.returncode == 0 and json_pass(moss_gate):
            print("[moss-annotations] PASS", flush=True)
            break
        if repair_round == 3:
            raise RuntimeError("MOSS annotation gate still failed after three repair rounds")
        print(f"[moss-annotations] starting repair round {repair_round + 1}", flush=True)
        run_parallel(["edm-v2-moss-0", "edm-v2-moss-1"])
    annotation_merge_gate = root / "data_v2" / "annotation_merge_report.json"
    dataset_build_gate = root / "data_v2" / "dataset_build_report.json"
    annotations_rebuilt = json_value(annotation_merge_gate, "moss_prompt_revision") != PROMPT_REVISION
    if annotations_rebuilt:
        start("edm-v2-build-annotations")
        wait_for_exit("edm-v2-build-annotations")
        require_json_pass(annotation_merge_gate, "audio-blind-annotation-merge")
        require_json_pass(dataset_build_gate, "annotations-and-dataset")
        reset = archive_stale_training_outputs(root)
        print(json.dumps(reset, ensure_ascii=False), flush=True)
        for downstream_gate in (
            "claim_consensus_report.json",
            "caption_repair_report.json",
            "tensor_validation_report.json",
            "metadata_upload_report.json",
            "annotation_quality_audit.json",
            "annotation_fidelity_audit.json",
        ):
            (root / "data_v2" / downstream_gate).unlink(missing_ok=True)
    else:
        run_stage(
            "edm-v2-build-annotations",
            dataset_build_gate,
            "annotations-and-dataset",
        )

    claim_consensus_gate = root / "data_v2" / "claim_consensus_report.json"
    if not json_pass(claim_consensus_gate):
        for consensus_round in range(4):
            run_parallel(["edm-v2-verify-claims-0", "edm-v2-verify-claims-1"])
            start("edm-v2-validate-claim-consensus")
            wait_for_exit("edm-v2-validate-claim-consensus")
            if json_pass(claim_consensus_gate):
                print("[multi-view-audible-claim-consensus] PASS", flush=True)
                break
            if consensus_round == 3:
                raise RuntimeError("multi-view claim consensus still failed after four rounds")
            print(f"[claim-consensus] retry round {consensus_round + 1}", flush=True)

    caption_repair_gate = root / "data_v2" / "caption_repair_report.json"
    caption_repair_current = (
        json_pass(caption_repair_gate)
        and json_value(caption_repair_gate, "caption_compiler_revision")
        == CAPTION_COMPILER_REVISION
    )
    if not caption_repair_current:
        for caption_round in range(4):
            run_parallel(["edm-v2-repair-captions-0", "edm-v2-repair-captions-1"])
            start("edm-v2-apply-caption-repairs")
            wait_for_exit("edm-v2-apply-caption-repairs")
            if (
                json_pass(caption_repair_gate)
                and json_value(caption_repair_gate, "caption_compiler_revision")
                == CAPTION_COMPILER_REVISION
            ):
                print("[audio-grounded-caption-repairs] PASS", flush=True)
                break
            if caption_round == 3:
                raise RuntimeError("caption repair still failed after four rounds")
            print(f"[caption-repair] retry round {caption_round + 1}", flush=True)
    annotation_quality_gate = root / "data_v2" / "annotation_quality_audit.json"
    if not json_pass(annotation_quality_gate):
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "audit_v2_annotation_quality.py"),
                "--project-root", str(root),
            ],
            check=False,
        )
    require_json_pass(annotation_quality_gate, "annotation-static-quality")

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
        "edm-v2-audit-annotation-fidelity",
        root / "data_v2" / "annotation_fidelity_audit.json",
        "stratified-annotation-fidelity",
    )
    runtime_audit_gate = root / "data_v2" / "trainer_runtime_audit.json"
    subprocess.run(
        [
            str(root / "vendor" / "ACE-Step-1.5-v2" / ".venv" / "bin" / "python"),
            str(root / "scripts" / "audit_v2_trainer_runtime.py"),
            "--project-root", str(root),
            "--vendor-root", str(root / "vendor" / "ACE-Step-1.5-v2"),
        ],
        check=False,
    )
    require_json_pass(runtime_audit_gate, "two-gpu-trainer-runtime")
    run_stage(
        "edm-v2-evaluate-baseline",
        root / "outputs" / "v2" / "baseline-xl-base" / "generation_report.json",
        "pristine-xl-base-audio",
    )
    run_stage(
        "edm-v2-score-baseline",
        root / "outputs" / "v2" / "baseline-xl-base" / "listening_scores.json",
        "pristine-xl-base-listening",
    )
    baseline_quality_gate = (
        root / "outputs" / "v2" / "baseline-xl-base" / "listening_quality_report.json"
    )
    if not json_pass(baseline_quality_gate):
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "validate_v2_listening_quality.py"),
                "--project-root", str(root),
                "--report", "outputs/v2/baseline-xl-base/listening_scores.json",
                "--output", "outputs/v2/baseline-xl-base/listening_quality_report.json",
                "--profile", "baseline",
            ],
            check=False,
        )
    require_json_pass(baseline_quality_gate, "pristine-xl-base-absolute-quality")

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

    robust_lora = root / "outputs" / "v2" / "robust-evaluation"
    robust_base = root / "outputs" / "v2" / "robust-base-evaluation"
    if not (
        json_pass(robust_lora / "generation_report.json")
        and json_pass(robust_base / "generation_report.json")
    ):
        run_parallel(["edm-v2-evaluate-robust", "edm-v2-evaluate-robust-base"])
    require_json_pass(robust_lora / "generation_report.json", "five-seed-lora-generation")
    require_json_pass(robust_base / "generation_report.json", "five-seed-base-generation")
    if not (
        json_pass(robust_lora / "listening_scores.json")
        and json_pass(robust_base / "listening_scores.json")
        and (robust_lora / "seed_robustness.json").is_file()
        and (robust_base / "seed_robustness.json").is_file()
    ):
        run_parallel(["edm-v2-score-moss-robust", "edm-v2-score-moss-robust-base"])
    require_json_pass(robust_lora / "listening_scores.json", "five-seed-lora-listening")
    require_json_pass(robust_base / "listening_scores.json", "five-seed-base-listening")
    require_json_pass(robust_lora / "seed_robustness.json", "five-seed-lora-quality")
    run_stage(
        "edm-v2-compare-robust",
        root / "outputs" / "v2" / "robust-comparison.json",
        "paired-five-seed-base-lora-comparison",
    )

    run_stage(
        "edm-v2-upload-evaluation",
        evaluation / "hf_evaluation_upload_report.json",
        "private-evaluation-upload",
    )
    preview = root / "outputs" / "release" / "melodic-edm-core-v2-preview"
    run_stage(
        "edm-v2-package-preview",
        preview / "preview_report.json",
        "best-val-preview-package",
    )
    run_stage(
        "edm-v2-upload-preview",
        evaluation / "preview_upload_report.json",
        "best-val-preview-upload",
    )
    run_stage(
        "edm-v2-verify-preview",
        evaluation / "preview_clean_verification_report.json",
        "best-val-preview-clean-inference",
    )
    run_stage(
        "edm-v2-train-final",
        root / "outputs" / "v2" / "final-all-data" / "final_validation_report.json",
        "fresh-all-231-training",
    )
    run_stage(
        "edm-v2-prepare-audio-dataset",
        root / "outputs" / "v2" / "audio_dataset_prepare_report.json",
        "private-audio-dataset-staging",
    )
    run_stage(
        "edm-v2-upload-audio-dataset",
        root / "outputs" / "v2" / "audio_dataset_upload_report.json",
        "private-audio-dataset-upload",
    )
    run_stage(
        "edm-v2-verify-audio-dataset",
        root / "outputs" / "v2" / "audio_dataset_clean_verification_report.json",
        "private-audio-dataset-clean-verification",
    )
    run_stage(
        "edm-v2-evaluate-final",
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "generation_report.json",
        "final-all-data-audio",
    )
    run_stage(
        "edm-v2-score-final",
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "listening_scores.json",
        "final-all-data-listening",
    )
    run_stage(
        "edm-v2-quality-final",
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "listening_quality_report.json",
        "final-all-data-absolute-quality",
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
    run_stage(
        "edm-v2-upload-completion",
        root / "outputs" / "v2" / "completion_upload_report.json",
        "private-completion-backup",
    )
    print("[v2] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

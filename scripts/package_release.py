#!/usr/bin/env python3
"""Create a secret-free Hugging Face model release from a validated adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"gh[oprsu]_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"release source missing or unsafe: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint", choices=("middle", "best_val", "last"), default="best_val")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    train_output = root / "outputs" / "training" / "melodic-edm-core-v1"
    train_gate = json.loads((train_output / "training_validation_report.json").read_text(encoding="utf-8"))
    evaluation_root = root / "outputs" / "inference" / "checkpoint_comparison"
    eval_gate = json.loads((evaluation_root / "evaluation_report.json").read_text(encoding="utf-8"))
    if train_gate.get("status") != "pass" or eval_gate.get("status") != "pass":
        raise SystemExit("training/evaluation release gates have not passed")

    release = root / "outputs" / "release" / "melodic-edm-core-v1"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    adapter = Path(train_gate["selected_checkpoints"][args.checkpoint])
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        copy_file(adapter / name, release / name)
    copy_file(root / "MODEL_CARD.md", release / "README.md")
    copy_file(root / "configs" / "train_lora_2x4090.json", release / "training_config.json")
    copy_file(root / "configs" / "inference_prompts.json", release / "inference_prompts.json")
    copy_file(root / "scripts" / "infer_release.py", release / "infer_release.py")
    atomic_json(release / "training_validation_report.json", {
        key: train_gate.get(key) for key in (
            "status", "configuration", "completed_epoch", "global_step",
            "epoch_loss_count", "validation_loss_count", "validation_state",
            "adapter_summaries", "gpu_observation", "errors",
        )
    })
    atomic_json(release / "evaluation_report.json", {
        "status": eval_gate.get("status"),
        "fixed_prompt_count": eval_gate.get("fixed_prompt_count"),
        "checkpoint_count": eval_gate.get("checkpoint_count"),
        "generated_outputs": eval_gate.get("generated_outputs"),
        "results": [
            {key: row.get(key) for key in ("checkpoint", "prompt_id", "seed", "probe", "status")}
            for row in eval_gate.get("results", [])
        ],
        "errors": eval_gate.get("errors", []),
    })

    selected_examples = [row for row in eval_gate["results"] if row["checkpoint"] == args.checkpoint]
    if len(selected_examples) != 3:
        raise RuntimeError(f"expected 3 generated examples, found {len(selected_examples)}")
    for row in selected_examples:
        copy_file(Path(row["audio_path"]), release / "examples" / f"{row['prompt_id']}.wav")

    release_manifest = {
        "status": "pass", "model_id": "Bangchis/melodic-edm-core-v1",
        "private": True, "selected_checkpoint": args.checkpoint,
        "base_model": "ACE-Step 1.5 XL-Base",
        "ace_step_commit": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        "catalog_records": 240, "training_records": 231,
        "deduplication_performed": False, "generated_examples": 3,
        "excluded": ["source audio", "separated stems", "preprocessed tensors", "optimizer state", "tokens", "cookies"],
    }
    atomic_json(release / "release_manifest.json", release_manifest)

    findings = []
    for path in release.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"file": str(path.relative_to(release)), "pattern": pattern.pattern.decode()})
    if findings:
        raise RuntimeError(f"secret scan failed: {findings}")

    # Two files remain to be written: this validation report and SHA256SUMS.
    current_files = [item for item in release.rglob("*") if item.is_file()]
    release_manifest["release_files"] = len(current_files) + 2
    release_manifest["secret_scan_findings"] = 0
    atomic_json(release / "release_report.json", release_manifest)
    checksums = []
    for path in sorted(item for item in release.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(release)}")
    (release / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(release_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

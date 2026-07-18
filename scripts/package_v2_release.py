#!/usr/bin/env python3
"""Create a secret-free, checksum-verified V2 Hugging Face model release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"gh[oprsu]_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy_file(source: Path, target: Path) -> None:
    """Copy a regular file while rejecting missing paths and symlinks."""
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"release source missing or unsafe: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def adapter_dir(path: Path) -> Path:
    """Resolve PEFT's optional named adapter subdirectory."""
    nested = path / "adapter"
    return nested if nested.is_dir() else path


def copy_adapter(source: Path, target: Path) -> None:
    """Copy only the deployable PEFT adapter files."""
    source = adapter_dir(source)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        copy_file(source / name, target / name)


def read_gate(path: Path) -> dict[str, Any]:
    """Load and require one successful pipeline gate."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "pass":
        raise RuntimeError(f"release gate has not passed: {path}")
    return value


def sanitized_training_report(value: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducibility facts while excluding local paths and bulky details."""
    keys = (
        "status", "best_validation_epoch", "best_optimizer_step",
        "best_validation_loss", "latest_validation_loss", "validation_losses",
        "checkpoint_count", "prompt_selection_counts", "gpu_observation",
        "metric_events", "errors",
    )
    return {key: value.get(key) for key in keys}


def sanitized_selection(value: dict[str, Any]) -> dict[str, Any]:
    """Remove server-local adapter paths from checkpoint selection evidence."""
    candidates = {
        label: {key: item for key, item in candidate.items() if key != "adapter_path"}
        for label, candidate in value.get("candidates", {}).items()
    }
    return {
        key: item for key, item in value.items()
        if key not in ("candidates", "best_val_output")
    } | {"candidates": candidates, "best_val_output": "best-val/"}


def sanitized_listening(value: dict[str, Any]) -> dict[str, Any]:
    """Remove server-local generated-audio paths from listening evidence."""
    output = {key: item for key, item in value.items() if key != "results"}
    output["results"] = [
        {key: item for key, item in row.items() if key != "audio_path"}
        for row in value.get("results", [])
    ]
    return output


def apply_recommended_lora_scale(package: Path, scale: float) -> None:
    """Write the quality-selected LoRA scale into packaged config and Colab defaults."""
    for relative in ("inference_config.json", "configs/inference_config.json"):
        path = package / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["recommended_lora_scale"] = scale
        atomic_json(path, value)
    notebook_path = package / "notebooks" / "melodic_edm_core_v2_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    replaced = 0
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        for index, line in enumerate(source):
            if str(line).startswith("LORA_SCALE = "):
                source[index] = (
                    f"LORA_SCALE = {scale:g}          # quality-selected default; freely adjustable 0.0–1.0\n"
                )
                replaced += 1
    if replaced != 1:
        raise RuntimeError(f"expected one Colab LORA_SCALE control, found {replaced}")
    atomic_json(notebook_path, notebook)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    train_root = root / "outputs" / "v2" / "train-validation"
    evaluation_root = root / "outputs" / "v2" / "checkpoint-evaluation"
    training = read_gate(train_root / "training_validation_report.json")
    generation = read_gate(evaluation_root / "generation_report.json")
    listening = read_gate(evaluation_root / "listening_scores.json")
    selection = read_gate(evaluation_root / "selection.json")
    final = read_gate(root / "outputs" / "v2" / "final-all-data" / "final_validation_report.json")
    final_generation = read_gate(
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "generation_report.json"
    )
    final_listening = read_gate(
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "listening_scores.json"
    )
    final_quality = read_gate(
        root / "outputs" / "v2" / "final-all-data" / "evaluation" / "listening_quality_report.json"
    )
    audio_prepare = read_gate(root / "outputs" / "v2" / "audio_dataset_prepare_report.json")
    audio_upload = read_gate(root / "outputs" / "v2" / "audio_dataset_upload_report.json")
    audio_clean = read_gate(root / "outputs" / "v2" / "audio_dataset_clean_verification_report.json")

    release = root / "outputs" / "release" / "melodic-edm-core-v2"
    if release.exists():
        raise FileExistsError(f"refusing to overwrite existing V2 release: {release}")
    release.mkdir(parents=True)
    copy_adapter(root / "outputs" / "v2" / "best-val", release / "best-val")
    copy_adapter(
        root / "outputs" / "v2" / "final-all-data" / "final",
        release / "final-all-data",
    )

    for source, destination in (
        (root / "MODEL_CARD_V2.md", release / "README.md"),
        (root / "TRAINING_V2.md", release / "docs" / "TRAINING_V2.md"),
        (root / "COLAB_INFERENCE_V2.md", release / "docs" / "COLAB_INFERENCE_V2.md"),
        (root / "notebooks" / "melodic_edm_core_v2_colab.ipynb", release / "notebooks" / "melodic_edm_core_v2_colab.ipynb"),
        (root / "configs" / "v2" / "train_val.json", release / "training_config.json"),
        (root / "configs" / "v2" / "final_all_data.json", release / "final_training_config.json"),
        (root / "configs" / "v2" / "inference_config.json", release / "inference_config.json"),
        (root / "configs" / "v2" / "train_val.json", release / "configs" / "train_val.json"),
        (root / "configs" / "v2" / "final_all_data.json", release / "configs" / "final_all_data.json"),
        (root / "configs" / "v2" / "inference_config.json", release / "configs" / "inference_config.json"),
        (root / "configs" / "v2" / "prompt_schema.json", release / "prompt_schema.json"),
        (root / "configs" / "v2" / "fixed_eval_prompts.json", release / "fixed_eval_prompts.json"),
        (root / "configs" / "v2" / "release_requirements.txt", release / "requirements.txt"),
        (root / "scripts" / "infer_v2_release.py", release / "scripts" / "infer_v2_release.py"),
        (root / "scripts" / "download_and_infer_v2.py", release / "scripts" / "download_and_infer_v2.py"),
        (root / "scripts" / "prompt_enhancer.py", release / "scripts" / "prompt_enhancer.py"),
        (root / "scripts" / "enhance_prompt_openrouter.py", release / "scripts" / "enhance_prompt_openrouter.py"),
        (root / "data_v2" / "trainer_runtime_audit.json", release / "reports" / "trainer_runtime_audit.json"),
        (root / "patches" / "acestep-xl-validation-caption-variants.patch", release / "patches" / "acestep-xl-validation-caption-variants.patch"),
        (root / "patches" / "acestep-ddp-remainder-validation.patch", release / "patches" / "acestep-ddp-remainder-validation.patch"),
    ):
        copy_file(source, destination)
    apply_recommended_lora_scale(release, float(selection["selected_lora_scale"]))

    atomic_json(release / "reports" / "training_validation_report.json", sanitized_training_report(training))
    atomic_json(release / "reports" / "selection.json", sanitized_selection(selection))
    atomic_json(release / "reports" / "final_validation_report.json", final)
    atomic_json(release / "reports" / "final_generation_report.json", {
        **final_generation,
        "results": [
            {key: item for key, item in row.items() if key != "audio_path"}
            for row in final_generation.get("results", [])
        ],
    })
    atomic_json(release / "reports" / "final_listening_scores.json", sanitized_listening(final_listening))
    atomic_json(release / "reports" / "final_listening_quality_report.json", final_quality)
    atomic_json(release / "reports" / "listening_scores.json", sanitized_listening(listening))
    atomic_json(release / "reports" / "audio_dataset_prepare_report.json", {
        key: value for key, value in audio_prepare.items() if key != "staging_root"
    })
    atomic_json(release / "reports" / "audio_dataset_upload_report.json", {
        key: value for key, value in audio_upload.items() if key != "resumable_upload_cache"
    })
    atomic_json(release / "reports" / "audio_dataset_clean_verification_report.json", audio_clean)
    for name in ("metrics_history.jsonl", "validation_state.json", "prompt_selection_counts.json"):
        copy_file(train_root / name, release / "metrics" / name)
    final_root = root / "outputs" / "v2" / "final-all-data"
    for source_name, destination_name in (
        ("metrics_history.jsonl", "final_all_data_metrics_history.jsonl"),
        ("prompt_selection_counts.json", "final_all_data_prompt_selection_counts.json"),
    ):
        copy_file(final_root / source_name, release / "metrics" / destination_name)
    copy_file(
        root / "outputs" / "v2" / "final_plan.json",
        release / "reports" / "final_plan.json",
    )

    examples = [
        row for row in generation["results"]
        if row["checkpoint"] == selection["selected_checkpoint"]
    ]
    if len(examples) != 3:
        raise RuntimeError(f"expected 3 selected-checkpoint examples, found {len(examples)}")
    example_manifest = []
    for row in examples:
        name = f"{row['prompt_id']}.wav"
        copy_file(Path(row["audio_path"]), release / "examples" / "best-val" / name)
        example_manifest.append({
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "source_checkpoint": selection["selected_checkpoint"],
            "optimizer_step": row["optimizer_step"],
            "path": f"examples/best-val/{name}",
            "probe": row["probe"],
        })
    final_examples = final_generation.get("results", [])
    if len(final_examples) != 3:
        raise RuntimeError(f"expected 3 final-all-data examples, found {len(final_examples)}")
    for row in final_examples:
        name = f"{row['prompt_id']}.wav"
        copy_file(Path(row["audio_path"]), release / "examples" / "final-all-data" / name)
        example_manifest.append({
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "source_checkpoint": "final-all-data",
            "optimizer_step": final["observed_optimizer_steps"],
            "path": f"examples/final-all-data/{name}",
            "probe": row["probe"],
        })
    atomic_json(release / "examples" / "manifest.json", example_manifest)

    manifest = {
        "status": "pass",
        "model_id": "Bangchis/melodic-edm-core-v2",
        "private": True,
        "base_model": "ACE-Step/acestep-v15-xl-base",
        "base_model_revision": "220c1166efbdd9583eafcb12eb160594bbfcb241",
        "ace_step_source_revision": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        "adapters": {
            "best-val": {
                "optimizer_step": selection["best_optimizer_step"],
                "recommended_lora_scale": selection["selected_lora_scale"],
                "purpose": "selected checkpoint from the 184/33 unique-audio grouped run",
            },
            "final-all-data": {
                "optimizer_steps": final["observed_optimizer_steps"],
                "recommended_lora_scale": selection["selected_lora_scale"],
                "purpose": "fresh adapter retrained on all 217 unique audio contents",
            },
        },
        "training_records": 231,
        "split_run": {"train": 196, "validation": 35, "test": 0},
        "unique_audio_training_records": 217,
        "unique_audio_split_run": {"train": 184, "validation": 33, "test": 0},
        "caption_variants_per_record": 3,
        "training_prompt_types": ["canonical", "composition", "production"],
        "deduplication_performed": True,
        "private_audio_dataset": {
            "repo_id": audio_upload["repo_id"],
            "revision": audio_upload["sha"],
            "records": audio_clean["records"],
            "train_records": audio_clean["train_records"],
            "validation_records": audio_clean["validation_records"],
            "clean_sha256_verification": audio_clean["status"] == "pass",
        },
        "generated_examples": len(example_manifest),
        "excluded": [
            "source audio", "separated stems", "preprocessed tensors",
            "optimizer state", "tokens", "cookies", "MOSS chain-of-thought",
        ],
    }
    atomic_json(release / "release_manifest.json", manifest)

    findings = []
    for path in release.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({
                    "file": str(path.relative_to(release)),
                    "pattern": pattern.pattern.decode(),
                })
    if findings:
        raise RuntimeError(f"secret scan failed: {findings}")

    current_files = [item for item in release.rglob("*") if item.is_file()]
    manifest["release_files"] = len(current_files) + 2
    manifest["secret_scan_findings"] = 0
    atomic_json(release / "release_report.json", manifest)
    checksums = []
    for path in sorted(
        item for item in release.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.relative_to(release)}")
    (release / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

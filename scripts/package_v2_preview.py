#!/usr/bin/env python3
"""Package the selected best-val adapter for private inference before final retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from package_v2_release import (
    SECRET_PATTERNS,
    atomic_json,
    copy_adapter,
    copy_file,
    read_gate,
    sanitized_listening,
    sanitized_selection,
    sanitized_training_report,
    apply_recommended_lora_scale,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    train_root = root / "outputs" / "v2" / "train-validation"
    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"

    training = read_gate(train_root / "training_validation_report.json")
    generation = read_gate(evaluation / "generation_report.json")
    listening = read_gate(evaluation / "listening_scores.json")
    selection = read_gate(evaluation / "selection.json")
    read_gate(evaluation / "hf_evaluation_upload_report.json")

    preview = root / "outputs" / "release" / "melodic-edm-core-v2-preview"
    if preview.exists():
        raise FileExistsError(f"refusing to overwrite existing V2 preview: {preview}")
    preview.mkdir(parents=True)
    copy_adapter(root / "outputs" / "v2" / "best-val", preview / "best-val")

    for source, destination in (
        (root / "MODEL_CARD_V2_PREVIEW.md", preview / "README.md"),
        (root / "TRAINING_V2.md", preview / "docs" / "TRAINING_V2.md"),
        (root / "COLAB_INFERENCE_V2.md", preview / "docs" / "COLAB_INFERENCE_V2.md"),
        (root / "notebooks" / "melodic_edm_core_v2_colab.ipynb", preview / "notebooks" / "melodic_edm_core_v2_colab.ipynb"),
        (root / "configs" / "v2" / "train_val.json", preview / "training_config.json"),
        (root / "configs" / "v2" / "inference_config.json", preview / "inference_config.json"),
        (root / "configs" / "v2" / "train_val.json", preview / "configs" / "train_val.json"),
        (root / "configs" / "v2" / "inference_config.json", preview / "configs" / "inference_config.json"),
        (root / "configs" / "v2" / "prompt_schema.json", preview / "prompt_schema.json"),
        (root / "configs" / "v2" / "fixed_eval_prompts.json", preview / "fixed_eval_prompts.json"),
        (root / "configs" / "v2" / "release_requirements.txt", preview / "requirements.txt"),
        (root / "scripts" / "infer_v2_release.py", preview / "scripts" / "infer_v2_release.py"),
        (root / "scripts" / "download_and_infer_v2.py", preview / "scripts" / "download_and_infer_v2.py"),
        (root / "scripts" / "prompt_enhancer.py", preview / "scripts" / "prompt_enhancer.py"),
        (root / "scripts" / "enhance_prompt_openrouter.py", preview / "scripts" / "enhance_prompt_openrouter.py"),
    ):
        copy_file(source, destination)
    apply_recommended_lora_scale(preview, float(selection["selected_lora_scale"]))

    atomic_json(preview / "reports" / "training_validation_report.json", sanitized_training_report(training))
    atomic_json(preview / "reports" / "selection.json", sanitized_selection(selection))
    atomic_json(preview / "reports" / "listening_scores.json", sanitized_listening(listening))
    for name in ("metrics_history.jsonl", "validation_state.json", "prompt_selection_counts.json"):
        copy_file(train_root / name, preview / "metrics" / name)

    examples = [
        row for row in generation["results"]
        if row["checkpoint"] == selection["selected_checkpoint"]
    ]
    if len(examples) != 3:
        raise RuntimeError(f"expected 3 selected-checkpoint examples, found {len(examples)}")
    example_manifest = []
    for row in examples:
        name = f"{row['prompt_id']}.wav"
        copy_file(Path(row["audio_path"]), preview / "examples" / "best-val" / name)
        example_manifest.append({
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "source_checkpoint": selection["selected_checkpoint"],
            "optimizer_step": row["optimizer_step"],
            "path": f"examples/best-val/{name}",
            "probe": row["probe"],
        })
    atomic_json(preview / "examples" / "manifest.json", example_manifest)

    report = {
        "status": "pass",
        "release_stage": "best-val-preview-before-final-all-data",
        "model_id": "Bangchis/melodic-edm-core-v2",
        "private": True,
        "selected_checkpoint": selection["selected_checkpoint"],
        "best_optimizer_step": selection["best_optimizer_step"],
        "recommended_lora_scale": selection["selected_lora_scale"],
        "training_records": 196,
        "validation_records": 35,
        "final_all_data_included": False,
        "generated_examples": len(example_manifest),
        "excluded": [
            "source audio", "separated stems", "preprocessed tensors",
            "optimizer state", "tokens", "cookies", "MOSS chain-of-thought",
        ],
    }
    atomic_json(preview / "preview_manifest.json", report)

    findings = []
    for path in preview.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"file": str(path.relative_to(preview)), "pattern": pattern.pattern.decode()})
    if findings:
        raise RuntimeError(f"preview secret scan failed: {findings}")
    report["secret_scan_findings"] = 0
    report["release_files"] = len([path for path in preview.rglob("*") if path.is_file()]) + 2
    atomic_json(preview / "preview_report.json", report)

    checksums = []
    for path in sorted(
        item for item in preview.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.relative_to(preview)}")
    (preview / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

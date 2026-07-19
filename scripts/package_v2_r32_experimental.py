#!/usr/bin/env python3
"""Package the measured epoch-30 R32 adapter as an explicitly experimental preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compare_v2_seed_robustness import compare
from package_v2_release import (
    SECRET_PATTERNS,
    atomic_json,
    copy_adapter,
    copy_file,
    sanitized_listening,
    sanitized_training_report,
)


REPO_ID = "Bangchis/melodic-edm-core-v2-r32-experimental"


def gate(path: Path, *, allow_failed_quality: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"pass", "failed"} if allow_failed_quality else {"pass"}
    if value.get("status") not in allowed:
        raise RuntimeError(f"invalid experimental release input: {path}")
    return value


def sanitize_generation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: item for key, item in value.items() if key != "results"},
        "results": [
            {key: item for key, item in row.items() if key not in {"audio_path", "adapter_path"}}
            for row in value.get("results", [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    train = root / "outputs" / "v2" / "train-validation"
    lora_eval = root / "outputs" / "v2" / "robust-evaluation"
    base_eval = root / "outputs" / "v2" / "robust-base-evaluation"

    training = gate(train / "training_validation_report.json")
    lora_generation = gate(lora_eval / "generation_report.json")
    base_generation = gate(base_eval / "generation_report.json")
    lora_listening = gate(lora_eval / "listening_scores.json")
    base_listening = gate(base_eval / "listening_scores.json")
    lora_robustness = gate(lora_eval / "seed_robustness.json", allow_failed_quality=True)
    base_robustness = gate(base_eval / "seed_robustness.json", allow_failed_quality=True)
    if (lora_robustness.get("records"), lora_robustness.get("passed_seeds")) != (15, 6):
        raise RuntimeError("expected measured LoRA robustness result 6/15")
    if (base_robustness.get("records"), base_robustness.get("passed_seeds")) != (15, 6):
        raise RuntimeError("expected measured Base robustness result 6/15")
    comparison = compare(lora_listening["results"], base_listening["results"], minimum_score=3)
    if comparison.get("status") != "pass" or comparison.get("paired_records") != 15:
        raise RuntimeError("paired 15-record Base/LoRA comparison failed")

    package = root / "outputs" / "release" / "melodic-edm-core-v2-r32-experimental"
    if package.exists():
        raise FileExistsError(f"refusing to overwrite experimental package: {package}")
    package.mkdir(parents=True)
    copy_adapter(train / "final", package / "experimental-r32")
    for source, destination in (
        (root / "MODEL_CARD_V2_R32_EXPERIMENTAL.md", package / "README.md"),
        (root / "notebooks" / "melodic_edm_core_v2_colab.ipynb", package / "notebooks" / "melodic_edm_core_v2_colab.ipynb"),
        (root / "scripts" / "infer_v2_release.py", package / "scripts" / "infer_v2_release.py"),
        (root / "scripts" / "download_and_infer_v2.py", package / "scripts" / "download_and_infer_v2.py"),
        (root / "scripts" / "prompt_enhancer.py", package / "scripts" / "prompt_enhancer.py"),
        (root / "scripts" / "enhance_prompt_openrouter.py", package / "scripts" / "enhance_prompt_openrouter.py"),
        (root / "configs" / "v2" / "fixed_eval_prompts.json", package / "fixed_eval_prompts.json"),
        (root / "configs" / "v2" / "robust_eval_prompts.json", package / "robust_eval_prompts.json"),
        (root / "configs" / "v2" / "prompt_schema.json", package / "prompt_schema.json"),
        (root / "configs" / "v2" / "release_requirements.txt", package / "requirements.txt"),
    ):
        copy_file(source, destination)

    inference_config = json.loads((root / "configs" / "v2" / "inference_config.json").read_text())
    inference_config.update({
        "release": "melodic-edm-core-v2-r32-experimental",
        "repository": REPO_ID,
        "adapter_subdirectory": "experimental-r32",
        "release_status": "experimental_not_final",
        "measured_lora_pass_rate": 0.4,
        "measured_base_pass_rate": 0.4,
    })
    atomic_json(package / "inference_config.json", inference_config)
    atomic_json(package / "reports" / "training_validation_report.json", sanitized_training_report(training))
    atomic_json(package / "reports" / "lora_generation_report.json", sanitize_generation(lora_generation))
    atomic_json(package / "reports" / "base_generation_report.json", sanitize_generation(base_generation))
    atomic_json(package / "reports" / "lora_listening_scores.json", sanitized_listening(lora_listening))
    atomic_json(package / "reports" / "base_listening_scores.json", sanitized_listening(base_listening))
    atomic_json(package / "reports" / "lora_seed_robustness.json", lora_robustness)
    atomic_json(package / "reports" / "base_seed_robustness.json", base_robustness)
    atomic_json(package / "reports" / "paired_base_lora_comparison.json", comparison)
    atomic_json(package / "training_snapshot.json", {
        "status": "historical_experimental_run",
        "base_model": "ACE-Step 1.5 XL-Base",
        "adapter": {"type": "lora", "rank": 32, "alpha": 32, "dropout": 0.1},
        "epochs": 30,
        "catalog_train_records": 196,
        "catalog_validation_records": 35,
        "cfg_dropout": 0.15,
        "learning_rate": 0.00005,
        "known_limitation": "caption lineage and duplicate weighting are superseded by the active retraining pipeline",
    })

    findings = []
    for path in package.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"file": str(path.relative_to(package)), "pattern": pattern.pattern.decode()})
    if findings:
        raise RuntimeError(f"experimental package secret scan failed: {findings}")

    manifest = {
        "status": "pass",
        "release_status": "experimental_not_final",
        "repo_id": REPO_ID,
        "private": True,
        "adapter": "epoch30_rank32_alpha32",
        "recommended_lora_scale": 0.5,
        "lora_passed_seeds": 6,
        "base_passed_seeds": 6,
        "evaluated_seeds_each": 15,
        "secret_scan_findings": 0,
        "final_retrained_model_included": False,
    }
    atomic_json(package / "experimental_manifest.json", manifest)
    manifest["release_files"] = len([path for path in package.rglob("*") if path.is_file()]) + 2
    atomic_json(package / "package_report.json", manifest)
    checksums = []
    for path in sorted(item for item in package.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(package)}")
    (package / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

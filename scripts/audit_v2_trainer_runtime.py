#!/usr/bin/env python3
"""Prove that the live ACE-Step trainer has safe two-GPU tail semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--vendor-root")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    vendor = Path(args.vendor_root).resolve() if args.vendor_root else root / "vendor" / "ACE-Step-1.5-v2"
    report_path = root / "data_v2" / "trainer_runtime_audit.json"
    errors: list[str] = []

    config_path = root / "configs" / "v2" / "train_val.json"
    trainer_path = vendor / "acestep" / "training_v2" / "trainer_fixed.py"
    helper_path = vendor / "acestep" / "training_v2" / "accumulation.py"
    patch_path = root / "patches" / "acestep-ddp-remainder-validation.patch"
    required = (config_path, trainer_path, helper_path, patch_path)
    for path in required:
        if not path.is_file():
            errors.append(f"missing:{path}")

    config: dict[str, object] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    optimization = config.get("optimization", {}) if isinstance(config, dict) else {}
    data = config.get("data", {}) if isinstance(config, dict) else {}
    expected_config = {
        "gpus": 2,
        "gradient_accumulation": 8,
        "batch_per_gpu": 1,
        "effective_batch": 16,
        "maximum_epochs": 30,
    }
    for key, expected in expected_config.items():
        observed = optimization.get(key) if isinstance(optimization, dict) else None
        if observed != expected:
            errors.append(f"config_mismatch:{key}:{observed}:{expected}")

    tensor_expectations = {
        "tensors_train_unique": int(data.get("train_unique_audio_records", 0)) if isinstance(data, dict) else 0,
        "tensors_validation_unique": int(data.get("validation_unique_audio_records", 0)) if isinstance(data, dict) else 0,
        "tensors_all_unique": int(data.get("unique_audio_records", 0)) if isinstance(data, dict) else 0,
    }
    tensor_counts: dict[str, int] = {}
    for directory, expected in tensor_expectations.items():
        observed = len(list((root / "data_v2" / directory).glob("*.pt")))
        tensor_counts[directory] = observed
        if observed != expected:
            errors.append(f"tensor_count:{directory}:{observed}:{expected}")

    trainer_source = trainer_path.read_text(encoding="utf-8") if trainer_path.is_file() else ""
    for marker in (
        "should_sync_gradient(",
        "rescale_remainder_gradients(",
        "use_distributed_sampler=False",
    ):
        if marker not in trainer_source:
            errors.append(f"trainer_marker_missing:{marker}")

    per_rank_microbatches = math.ceil(tensor_expectations["tensors_train_unique"] / 2)
    tail_microbatches = per_rank_microbatches % 8
    helper_checks: dict[str, object] = {}
    if helper_path.is_file():
        sys.path.insert(0, str(vendor))
        try:
            from acestep.training_v2.accumulation import (  # type: ignore
                rescale_remainder_gradients,
                should_sync_gradient,
            )

            final_tail_sync = should_sync_gradient(
                tail_microbatches - 1, 8, per_rank_microbatches - 1, per_rank_microbatches
            )
            pre_tail_sync = should_sync_gradient(
                max(0, tail_microbatches - 2), 8, per_rank_microbatches - 2, per_rank_microbatches
            )
            parameter = torch.nn.Parameter(torch.ones(1))
            parameter.grad = torch.tensor([0.5])
            scale = rescale_remainder_gradients([parameter], tail_microbatches, 8)
            helper_checks = {
                "final_tail_forces_sync": final_tail_sync,
                "pre_final_tail_suppresses_sync": not pre_tail_sync,
                "remainder_gradient_scale": scale,
                "gradient_after_rescale": float(parameter.grad.item()),
            }
            if not final_tail_sync or pre_tail_sync:
                errors.append("ddp_tail_sync_semantics_invalid")
            if scale != 2.0 or float(parameter.grad.item()) != 1.0:
                errors.append("ddp_tail_gradient_rescale_invalid")
        except Exception as exc:  # the report must preserve the exact failed gate
            errors.append(f"helper_runtime_failure:{type(exc).__name__}:{exc}")

    report: dict[str, object] = {
        "status": "pass" if not errors else "failed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "vendor_root": str(vendor),
        "trainer_sha256": sha256(trainer_path) if trainer_path.is_file() else None,
        "accumulation_helper_sha256": sha256(helper_path) if helper_path.is_file() else None,
        "patch_sha256": sha256(patch_path) if patch_path.is_file() else None,
        "train_unique_records": tensor_expectations["tensors_train_unique"],
        "validation_unique_records": tensor_expectations["tensors_validation_unique"],
        "all_unique_records": tensor_expectations["tensors_all_unique"],
        "tensor_counts": tensor_counts,
        "world_size": 2,
        "per_rank_microbatches": per_rank_microbatches,
        "gradient_accumulation": 8,
        "tail_microbatches": tail_microbatches,
        "validation_sampler": "full_33_records_on_each_rank_without_padding",
        "helper_checks": helper_checks,
        "errors": errors,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

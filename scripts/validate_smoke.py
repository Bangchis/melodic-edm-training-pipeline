#!/usr/bin/env python3
"""Validate that the two-GPU smoke run trained, validated, saved and reloads."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_adapter_dir(path: Path) -> Path:
    """Return the PEFT directory used by ACE-Step checkpoint layouts."""
    nested = path / "adapter"
    return nested if nested.is_dir() else path


def inspect_adapter(path: Path) -> tuple[dict[str, Any], list[str]]:
    from safetensors import safe_open

    path = resolve_adapter_dir(path)
    errors: list[str] = []
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file():
        return {}, ["adapter_config_missing"]
    if not weights_path.is_file() or weights_path.stat().st_size < 1024:
        return {}, ["adapter_weights_missing_or_tiny"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.1}
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"adapter_config_mismatch:{key}:{config.get(key)}")
    tensor_count = 0
    nonzero_count = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            tensor_count += 1
            if not bool(value.isfinite().all()):
                errors.append(f"adapter_nonfinite:{key}")
            if bool(value.count_nonzero()):
                nonzero_count += 1
    if tensor_count == 0:
        errors.append("adapter_has_no_tensors")
    if nonzero_count == 0:
        errors.append("adapter_all_zero")
    return {"tensor_count": tensor_count, "nonzero_tensor_count": nonzero_count}, errors


def inspect_gpu_metrics(path: Path) -> tuple[dict[str, Any], list[str]]:
    samples: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    errors: list[str] = []
    if not path.is_file():
        return {}, ["gpu_metrics_missing"]
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                continue
            try:
                index = int(row[0].strip())
                memory = float(row[1].replace("MiB", "").strip())
                utilization = float(row[2].replace("%", "").strip())
            except ValueError:
                continue
            if index in samples:
                samples[index].append((memory, utilization))
    summary = {}
    for index in (0, 1):
        values = samples[index]
        maximum_memory = max((item[0] for item in values), default=0.0)
        maximum_utilization = max((item[1] for item in values), default=0.0)
        summary[str(index)] = {
            "samples": len(values),
            "max_memory_mib": maximum_memory,
            "max_utilization_percent": maximum_utilization,
        }
        if len(values) < 2 or maximum_memory < 1024:
            errors.append(f"gpu_{index}_not_observed_training")
    return summary, errors


def reload_adapter(checkpoint_dir: Path, adapter_dir: Path) -> tuple[bool, str]:
    """Load the saved PEFT adapter back onto a clean XL-Base decoder."""
    try:
        import torch
        from acestep.training.lora_checkpoint import load_lora_weights
        from acestep.training_v2.model_loader import load_decoder_for_training

        model = load_decoder_for_training(checkpoint_dir, "xl_base", "cuda:0", "bf16")
        model = load_lora_weights(model, str(adapter_dir))
        loaded = hasattr(model.decoder, "peft_config") and bool(model.decoder.peft_config)
        del model
        torch.cuda.empty_cache()
        return loaded, "" if loaded else "peft_config_empty_after_reload"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--reload-adapter", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "smoke"
    errors: list[str] = []

    log_path = output / "training.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    epoch_losses = [float(value) for value in re.findall(r"Epoch 1/1[^\n]*Loss:\s*([0-9.eE+-]+)", log)]
    val_losses = [float(value) for value in re.findall(r"Validation epoch 1:\s*([0-9.eE+-]+)", log)]
    if not epoch_losses or not all(math.isfinite(value) for value in epoch_losses):
        errors.append("finite_epoch_loss_not_found")
    if not val_losses or not all(math.isfinite(value) for value in val_losses):
        errors.append("finite_validation_loss_not_found")

    validation_path = output / "validation_state.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    if not validation or any(not math.isfinite(float(validation.get(key, math.nan))) for key in ("best_loss", "latest_loss")):
        errors.append("validation_state_missing_or_nonfinite")

    final_adapter = resolve_adapter_dir(output / "final")
    adapter_summary, adapter_errors = inspect_adapter(final_adapter)
    errors.extend(adapter_errors)
    best_adapter = resolve_adapter_dir(output / "checkpoints" / "best_val")
    _, best_errors = inspect_adapter(best_adapter)
    errors.extend(f"best_val:{value}" for value in best_errors)
    epoch_dirs = sorted((output / "checkpoints").glob("epoch_1_loss_*"))
    if len(epoch_dirs) != 1:
        errors.append(f"epoch_checkpoint_count:{len(epoch_dirs)}")
        training_state = {}
    else:
        epoch_adapter = resolve_adapter_dir(epoch_dirs[0])
        _, epoch_errors = inspect_adapter(epoch_adapter)
        errors.extend(f"epoch_checkpoint:{value}" for value in epoch_errors)
        try:
            import torch

            training_state = torch.load(epoch_dirs[0] / "training_state.pt", map_location="cpu", weights_only=True)
            if int(training_state.get("epoch", 0)) != 1 or int(training_state.get("global_step", 0)) <= 0:
                errors.append("training_state_invalid")
        except Exception as exc:
            training_state = {}
            errors.append(f"training_state_load_failed:{type(exc).__name__}:{exc}")

    gpu_summary, gpu_errors = inspect_gpu_metrics(output / "gpu_metrics.csv")
    errors.extend(gpu_errors)
    reload_ok = False
    reload_error = "not_requested"
    if args.reload_adapter and not adapter_errors:
        reload_ok, reload_error = reload_adapter(root / args.checkpoint_dir, final_adapter)
        if not reload_ok:
            errors.append("adapter_reload_failed:" + reload_error)

    report = {
        "status": "pass" if not errors else "failed",
        "configuration": "xl_base_lora_r32_alpha64_dropout0.1_lr1e-4_ddp2",
        "epoch_losses": epoch_losses,
        "validation_losses": val_losses,
        "validation_state": validation,
        "global_step": training_state.get("global_step"),
        "adapter": adapter_summary,
        "adapter_reload_verified": reload_ok,
        "gpu_observation": gpu_summary,
        "errors": errors,
    }
    atomic_json(output / "smoke_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

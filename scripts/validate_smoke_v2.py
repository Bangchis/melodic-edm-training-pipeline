#!/usr/bin/env python3
"""Validate the 66-step two-GPU v2 smoke and one-step resume."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open

from v2_common import atomic_json


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def adapter_dir(path: Path) -> Path:
    """Resolve PEFT's optional named adapter directory."""
    nested = path / "adapter"
    return nested if nested.is_dir() else path


def inspect_adapter(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Validate v2 LoRA hyperparameters, tensors and projection coverage."""
    path = adapter_dir(path)
    errors: list[str] = []
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        return {}, ["adapter_files_missing"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in (("r", 32), ("lora_alpha", 32), ("lora_dropout", 0.1)):
        if config.get(key) != expected:
            errors.append(f"adapter_config_mismatch:{key}:{config.get(key)}")
    configured_targets = {str(value).rsplit(".", 1)[-1] for value in config.get("target_modules", [])}
    if configured_targets != TARGETS:
        errors.append(f"target_modules_mismatch:{sorted(configured_targets)}")
    tensor_keys: list[str] = []
    nonzero = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            tensor_keys.append(key)
            if not bool(value.isfinite().all()):
                errors.append(f"adapter_nonfinite:{key}")
            if bool(value.count_nonzero()):
                nonzero += 1
    missing = [target for target in sorted(TARGETS) if not any(f".{target}." in key for key in tensor_keys)]
    if missing:
        errors.append(f"adapter_missing_targets:{missing}")
    if nonzero == 0:
        errors.append("adapter_all_zero")
    return {
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "tensor_count": len(tensor_keys),
        "nonzero_tensor_count": nonzero,
        "target_modules": sorted(configured_targets),
    }, errors


def inspect_gpu_metrics(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Confirm both physical GPUs were observed with model-sized allocations."""
    samples: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) != 3:
                    continue
                try:
                    index = int(row[0].strip())
                    memory = float(row[1].strip())
                    utilization = float(row[2].strip())
                except ValueError:
                    continue
                if index in samples:
                    samples[index].append((memory, utilization))
    summary: dict[str, Any] = {}
    errors: list[str] = []
    for index, values in samples.items():
        maximum_memory = max((value[0] for value in values), default=0.0)
        maximum_utilization = max((value[1] for value in values), default=0.0)
        summary[str(index)] = {
            "samples": len(values),
            "max_memory_mib": maximum_memory,
            "max_utilization_percent": maximum_utilization,
        }
        if len(values) < 2 or maximum_memory < 1024:
            errors.append(f"gpu_{index}_not_observed_training")
    return summary, errors


def reload_adapter(root: Path, path: Path) -> tuple[bool, str]:
    """Reload the smoke adapter onto a clean XL-Base decoder."""
    try:
        import torch
        from acestep.training.lora_checkpoint import load_lora_weights
        from acestep.training_v2.model_loader import load_decoder_for_training

        model = load_decoder_for_training(root / "checkpoints", "xl_base", "cuda:0", "bf16")
        model = load_lora_weights(model, str(adapter_dir(path)))
        loaded = hasattr(model.decoder, "peft_config") and bool(model.decoder.peft_config)
        del model
        torch.cuda.empty_cache()
        return loaded, "" if loaded else "peft_config_empty"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--reload-adapter", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "smoke"
    errors: list[str] = []
    log = (output / "training.log").read_text(encoding="utf-8", errors="replace")
    losses = [float(value) for value in re.findall(r"Step \d+, Loss: ([0-9.eE+-]+)", log)]
    if not losses or not all(math.isfinite(value) for value in losses):
        errors.append("finite_step_losses_missing")
    elif len(losses) > 1 and min(losses[1:]) >= losses[0]:
        errors.append("loss_never_decreased_below_first_logged_value")
    if re.search(r"\b(?:OOM|out of memory|NaN|Inf)\b", log, flags=re.IGNORECASE):
        errors.append("fatal_numeric_or_memory_marker_in_log")
    resume_marker = re.search(
        r"Resumed(?: LoRA)? from epoch 5, step 65[^\n]*",
        log,
    )
    if resume_marker is None:
        errors.append("checkpoint_resume_marker_missing")
    else:
        marker_text = resume_marker.group(0)
        if "optimizer OK" not in marker_text or "scheduler OK" not in marker_text:
            errors.append(f"checkpoint_resume_state_incomplete:{marker_text}")

    validation_path = output / "validation_state.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    if int(validation.get("best_optimizer_step", 0)) <= 0:
        errors.append("best_optimizer_step_missing")
    for key in ("best_loss", "latest_loss"):
        try:
            if not math.isfinite(float(validation[key])):
                errors.append(f"validation_{key}_nonfinite")
        except (KeyError, TypeError, ValueError):
            errors.append(f"validation_{key}_missing")

    counts_path = output / "prompt_selection_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8")) if counts_path.is_file() else {}
    cumulative = counts.get("cumulative_counts", [])
    if len(cumulative) != 1 or int(cumulative[0]) <= 0:
        errors.append(f"single_fused_canonical_prompt_not_observed:{cumulative}")

    checkpoints = sorted((output / "checkpoints").glob("epoch_6_loss_*"))
    if len(checkpoints) != 1:
        errors.append(f"resume_checkpoint_count:{len(checkpoints)}")
        state = {}
    else:
        import torch

        state = torch.load(checkpoints[0] / "training_state.pt", map_location="cpu", weights_only=True)
        if int(state.get("global_step", 0)) != 66:
            errors.append(f"global_step_not_66:{state.get('global_step')}")
    adapter_summary, adapter_errors = inspect_adapter(output / "final")
    errors.extend(adapter_errors)
    gpu_summary, gpu_errors = inspect_gpu_metrics(output / "gpu_metrics.csv")
    errors.extend(gpu_errors)
    reload_ok = False
    reload_error = "not_requested"
    if args.reload_adapter and not adapter_errors:
        reload_ok, reload_error = reload_adapter(root, output / "final")
        if not reload_ok:
            errors.append(f"adapter_reload_failed:{reload_error}")
    tensor_report = json.loads((root / "data_v2" / "tensor_validation_report.json").read_text(encoding="utf-8"))
    if tensor_report.get("prompt_embeddings_per_record") != 1:
        errors.append("tensor_prompt_embedding_count_invalid")
    if tensor_report.get("validation_caption_index") != 0 or tensor_report.get("validation_cfg_dropout") != 0.0:
        errors.append("validation_prompt_or_cfg_semantics_invalid")

    report = {
        "status": "pass" if not errors else "failed",
        "configuration": "xl_base_lora_r32_alpha32_dropout0.1_lr5e-5_ddp2",
        "optimizer_steps": state.get("global_step"),
        "logged_losses": losses,
        "validation_state": validation,
        "prompt_selection_counts": cumulative,
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

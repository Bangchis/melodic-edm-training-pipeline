#!/usr/bin/env python3
"""Validate train/validation completion and expose the best optimizer step."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from v2_common import atomic_json
from validate_smoke_v2 import inspect_adapter, inspect_gpu_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "train-validation"
    errors: list[str] = []

    validation_path = output / "validation_state.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    for key in ("best_loss", "latest_loss"):
        try:
            if not math.isfinite(float(validation[key])):
                errors.append(f"{key}_nonfinite")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key}_missing")
    best_epoch = int(validation.get("best_epoch", 0))
    best_step = int(validation.get("best_optimizer_step", 0))
    if best_epoch <= 0 or best_step <= 0:
        errors.append("best_epoch_or_optimizer_step_missing")

    log_path = output / "training.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    validation_losses = [float(value) for value in re.findall(r"Validation epoch \d+: ([0-9.eE+-]+)", log)]
    if not validation_losses or not all(math.isfinite(value) for value in validation_losses):
        errors.append("finite_validation_loss_history_missing")
    validation_epochs = {
        int(value) for value in re.findall(r"Validation epoch (\d+):", log)
    }
    required_epochs = {5, 10, 15, 20, 25, 30}
    if validation_epochs != required_epochs:
        errors.append(f"validation_epochs_invalid:{sorted(validation_epochs)}")
    if re.search(r"\b(?:OOM|out of memory|NaN|Inf)\b", log, flags=re.IGNORECASE):
        errors.append("fatal_numeric_or_memory_marker_in_log")

    metric_path = output / "metrics_history.jsonl"
    metrics = [
        json.loads(line)
        for line in metric_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if metric_path.is_file() else []
    event_types = {item.get("event") for item in metrics}
    for required in ("train_step", "train_epoch", "validation", "gradient_norm", "checkpoint"):
        if required not in event_types:
            errors.append(f"metric_event_missing:{required}")

    prompt_path = output / "prompt_selection_counts.json"
    prompt_counts = json.loads(prompt_path.read_text(encoding="utf-8")) if prompt_path.is_file() else {}
    cumulative = prompt_counts.get("cumulative_counts", [])
    if len(cumulative) != 3 or any(int(value) <= 0 for value in cumulative):
        errors.append(f"three_fused_prompt_randomization_invalid:{cumulative}")

    adapter_summary, adapter_errors = inspect_adapter(output / "checkpoints" / "best_val")
    errors.extend(f"best_val:{value}" for value in adapter_errors)
    final_summary, final_errors = inspect_adapter(output / "final")
    errors.extend(f"last:{value}" for value in final_errors)
    gpu_summary, gpu_errors = inspect_gpu_metrics(output / "gpu_metrics.csv")
    errors.extend(gpu_errors)
    checkpoints = sorted(
        path for path in (output / "checkpoints").glob("epoch_*_loss_*")
        if re.match(r"epoch_\d+_loss_", path.name)
    )
    checkpoint_epochs: set[int] = set()
    for path in checkpoints:
        match = re.match(r"epoch_(\d+)_loss_", path.name)
        if not match or int(match.group(1)) % 5 != 0:
            errors.append(f"unexpected_checkpoint_epoch:{path.name}")
        else:
            checkpoint_epochs.add(int(match.group(1)))
    if checkpoint_epochs != required_epochs:
        errors.append(f"checkpoint_epochs_invalid:{sorted(checkpoint_epochs)}")

    stop_override_path = output / "training_stop_override.json"
    stop_override = (
        json.loads(stop_override_path.read_text(encoding="utf-8"))
        if stop_override_path.is_file()
        else None
    )
    if stop_override is not None:
        if stop_override.get("status") != "pass" or stop_override.get("termination") != "user_requested_cutoff":
            errors.append("invalid_user_stop_override")
        if int(stop_override.get("requested_cutoff_epoch") or 0) != best_epoch:
            errors.append("user_stop_cutoff_is_not_best_epoch")

    report = {
        "status": "pass" if not errors else "failed",
        "best_validation_epoch": best_epoch,
        "best_optimizer_step": best_step,
        "best_validation_loss": validation.get("best_loss"),
        "latest_validation_loss": validation.get("latest_loss"),
        "validation_losses": validation_losses,
        "validation_epochs": sorted(validation_epochs),
        "checkpoint_epochs": sorted(checkpoint_epochs),
        "checkpoint_count": len(checkpoints),
        "best_adapter": adapter_summary,
        "last_adapter": final_summary,
        "prompt_selection_counts": cumulative,
        "gpu_observation": gpu_summary,
        "metric_events": sorted(str(value) for value in event_types),
        "termination": stop_override or {"termination": "trainer_early_stop_or_max_epochs"},
        "errors": errors,
    }
    atomic_json(output / "training_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

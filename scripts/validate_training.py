#!/usr/bin/env python3
"""Validate the completed fixed main training run and select comparison checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from latest_checkpoint import epoch_checkpoints  # noqa: E402
from validate_smoke import inspect_adapter, inspect_gpu_metrics  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def select_checkpoints(output: Path) -> dict[str, str]:
    checkpoints = epoch_checkpoints(output / "checkpoints")
    if not checkpoints:
        return {}
    last_epoch = checkpoints[-1][0]
    middle = min(checkpoints, key=lambda item: (abs(item[0] - last_epoch / 2), item[0]))
    return {
        "middle": str(middle[1]),
        "best_val": str(output / "checkpoints" / "best_val"),
        "last": str(output / "final"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "training" / "melodic-edm-core-v1"
    errors: list[str] = []

    log_path = output / "training.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    epoch_losses = [float(value) for value in re.findall(r"Epoch \d+/150[^\n]*Loss:\s*([0-9.eE+-]+)", log)]
    val_losses = [float(value) for value in re.findall(r"Validation epoch \d+:\s*([0-9.eE+-]+)", log)]
    if not epoch_losses or not all(math.isfinite(value) for value in epoch_losses):
        errors.append("finite_epoch_losses_not_found")
    if not val_losses or not all(math.isfinite(value) for value in val_losses):
        errors.append("finite_validation_losses_not_found")

    validation_path = output / "validation_state.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    if not validation or any(not math.isfinite(float(validation.get(key, math.nan))) for key in ("best_loss", "latest_loss")):
        errors.append("validation_state_missing_or_nonfinite")

    selected = select_checkpoints(output)
    if set(selected) != {"middle", "best_val", "last"}:
        errors.append("comparison_checkpoints_missing")
    adapter_summaries = {}
    for label, raw_path in selected.items():
        summary, adapter_errors = inspect_adapter(Path(raw_path))
        adapter_summaries[label] = summary
        errors.extend(f"{label}:{value}" for value in adapter_errors)

    checkpoints = epoch_checkpoints(output / "checkpoints")
    last_epoch = checkpoints[-1][0] if checkpoints else 0
    global_step = 0
    if checkpoints:
        try:
            import torch

            state = torch.load(checkpoints[-1][1] / "training_state.pt", map_location="cpu", weights_only=True)
            global_step = int(state.get("global_step", 0))
            if int(state.get("epoch", 0)) != last_epoch or global_step <= 0:
                errors.append("last_training_state_invalid")
        except Exception as exc:
            errors.append(f"last_training_state_load_failed:{type(exc).__name__}:{exc}")

    gpu_summary, gpu_errors = inspect_gpu_metrics(output / "gpu_metrics.csv")
    errors.extend(gpu_errors)
    report = {
        "status": "pass" if not errors else "failed",
        "configuration": "xl_base_lora_r32_alpha64_dropout0.1_lr1e-4_ddp2",
        "completed_epoch": last_epoch,
        "global_step": global_step,
        "epoch_loss_count": len(epoch_losses),
        "validation_loss_count": len(val_losses),
        "validation_state": validation,
        "selected_checkpoints": selected,
        "adapter_summaries": adapter_summaries,
        "gpu_observation": gpu_summary,
        "errors": errors,
    }
    atomic_json(output / "training_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

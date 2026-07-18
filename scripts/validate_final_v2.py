#!/usr/bin/env python3
"""Validate the fresh all-data adapter reached the exact scaled step."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from v2_common import atomic_json
from validate_smoke_v2 import inspect_adapter, inspect_gpu_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "final-all-data"
    plan = json.loads((root / "outputs" / "v2" / "final_plan.json").read_text(encoding="utf-8"))
    expected_steps = int(plan["final_optimizer_steps"])
    errors: list[str] = []
    log = (output / "training.log").read_text(encoding="utf-8", errors="replace")
    if "Loading checkpoint" in log or "Resumed LoRA" in log or "[RESUME]" in log:
        errors.append("final_run_was_not_fresh")
    if f"Reached exact optimizer-step limit {expected_steps}" not in log:
        errors.append("exact_step_stop_marker_missing")

    checkpoint_candidates: list[tuple[int, Path]] = []
    for path in (output / "checkpoints").glob("epoch_*_loss_*"):
        state_path = path / "training_state.pt"
        if not state_path.is_file():
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        checkpoint_candidates.append((int(state.get("global_step", 0)), path))
    checkpoint_candidates.sort()
    observed_steps = checkpoint_candidates[-1][0] if checkpoint_candidates else 0
    if observed_steps != expected_steps:
        errors.append(f"final_step_mismatch:{observed_steps}:{expected_steps}")
    adapter_summary, adapter_errors = inspect_adapter(output / "final")
    errors.extend(adapter_errors)
    counts_path = output / "prompt_selection_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8")) if counts_path.is_file() else {}
    cumulative = counts.get("cumulative_counts", [])
    if len(cumulative) != 1 or int(cumulative[0]) <= 0:
        errors.append(f"single_fused_canonical_prompt_invalid:{cumulative}")
    tensor_report = json.loads((root / "data_v2" / "tensor_validation_report.json").read_text(encoding="utf-8"))
    if tensor_report.get("all_tensors") != 231:
        errors.append("final_dataset_is_not_231_records")
    gpu_summary, gpu_errors = inspect_gpu_metrics(output / "gpu_metrics.csv")
    errors.extend(gpu_errors)
    report = {
        "status": "pass" if not errors else "failed",
        "initialization": "fresh_xl_base_and_fresh_rank32_lora",
        "records": 231,
        "expected_optimizer_steps": expected_steps,
        "observed_optimizer_steps": observed_steps,
        "adapter": adapter_summary,
        "prompt_selection_counts": cumulative,
        "gpu_observation": gpu_summary,
        "errors": errors,
    }
    atomic_json(output / "final_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

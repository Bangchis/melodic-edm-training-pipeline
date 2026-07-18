#!/usr/bin/env python3
"""Scale the selected validation optimizer step to all 231 records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2_common import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--selection",
        default="outputs/v2/checkpoint-evaluation/selection.json",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    selection = json.loads((root / args.selection).read_text(encoding="utf-8"))
    best_steps = int(selection["best_optimizer_step"])
    if best_steps <= 0:
        raise ValueError("best_optimizer_step must be positive")
    final_steps = round(best_steps * 231 / 196)
    plan = {
        "status": "ready",
        "selected_checkpoint": selection["selected_checkpoint"],
        "best_optimizer_step": best_steps,
        "train_records": 196,
        "final_records": 231,
        "formula": "round(best_optimizer_step * 231 / 196)",
        "final_optimizer_steps": final_steps,
        "initialization": "fresh_xl_base_and_fresh_rank48_lora",
        "resume": False,
    }
    atomic_json(root / "outputs" / "v2" / "final_plan.json", plan)
    print(final_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

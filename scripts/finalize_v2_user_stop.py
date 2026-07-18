#!/usr/bin/env python3
"""Finalize a user-requested train/validation cutoff without deleting later evidence."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from v2_common import atomic_json


EPOCH_PATTERN = re.compile(r"epoch_(\d+)_loss_")


def checkpoint_state(path: Path) -> dict[str, int]:
    value = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    return {"epoch": int(value.get("epoch") or 0), "optimizer_step": int(value.get("global_step") or 0)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--cutoff-epoch", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "train-validation"
    validation = json.loads((output / "validation_state.json").read_text(encoding="utf-8"))
    cutoff = args.cutoff_epoch
    if cutoff <= 0:
        raise ValueError("cutoff epoch must be positive")
    if int(validation.get("best_epoch") or 0) != cutoff:
        raise RuntimeError(
            f"requested cutoff {cutoff} is not the current best epoch {validation.get('best_epoch')}"
        )

    checkpoints: list[tuple[int, Path, dict[str, int]]] = []
    for path in (output / "checkpoints").glob("epoch_*_loss_*"):
        match = EPOCH_PATTERN.match(path.name)
        if not match or not (path / "training_state.pt").is_file():
            continue
        checkpoints.append((int(match.group(1)), path, checkpoint_state(path)))
    cutoff_matches = [item for item in checkpoints if item[0] == cutoff]
    if len(cutoff_matches) != 1:
        raise RuntimeError(f"expected one complete epoch-{cutoff} checkpoint, found {len(cutoff_matches)}")
    _, cutoff_path, cutoff_state = cutoff_matches[0]
    best_step = int(validation.get("best_optimizer_step") or 0)
    if cutoff_state["optimizer_step"] != best_step:
        raise RuntimeError(
            f"cutoff checkpoint step {cutoff_state['optimizer_step']} != best step {best_step}"
        )

    best_source = output / "checkpoints" / "best_val" / "adapter"
    if not best_source.is_dir():
        raise FileNotFoundError(best_source)
    final_dir = output / "final"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing split-run final adapter: {final_dir}")
    shutil.copytree(best_source, final_dir)

    post_cutoff = sorted(epoch for epoch, _, _ in checkpoints if epoch > cutoff)
    latest_logged_epoch = 0
    log = (output / "training.log").read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"\[OK\] Epoch (\d+)/", log):
        latest_logged_epoch = max(latest_logged_epoch, int(match.group(1)))
    report = {
        "status": "pass",
        "termination": "user_requested_cutoff",
        "requested_cutoff_epoch": cutoff,
        "selected_best_epoch": int(validation["best_epoch"]),
        "selected_best_optimizer_step": best_step,
        "selected_best_validation_loss": validation["best_loss"],
        "cutoff_checkpoint": str(cutoff_path),
        "split_run_final_adapter_source": str(best_source),
        "post_cutoff_completed_checkpoints_preserved_but_excluded": post_cutoff,
        "latest_fully_logged_epoch_before_stop": latest_logged_epoch,
        "source_checkpoints_deleted": False,
    }
    atomic_json(output / "training_stop_override.json", report)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_training_v2.py"),
            "--project-root", str(root),
        ],
        check=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

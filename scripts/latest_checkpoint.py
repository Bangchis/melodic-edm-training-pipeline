#!/usr/bin/env python3
"""Print the newest resumable ACE-Step epoch checkpoint, if any."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(r"^epoch_(\d+)_loss_")


def epoch_checkpoints(root: Path) -> list[tuple[int, Path]]:
    found = []
    if not root.is_dir():
        return found
    for path in root.iterdir():
        match = EPOCH_PATTERN.match(path.name)
        if path.is_dir() and match and (path / "training_state.pt").is_file():
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root")
    args = parser.parse_args()
    found = epoch_checkpoints(Path(args.checkpoint_root).resolve())
    if found:
        print(found[-1][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

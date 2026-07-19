#!/usr/bin/env python3
"""Merge preprocessed v2 tensor shards using safe same-filesystem hard links."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from v2_common import atomic_json


def hardlink_tensor(source: Path, target: Path) -> None:
    """Link one tensor into a merged loader directory without duplicating data.

    ACE-Step resolves manifest entries against the dataset directory and rejects
    symlinks whose real target sits outside that directory. A hard link keeps the
    same inode and storage usage while remaining inside the loader's safe root.
    """
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f"refusing to overwrite unrelated tensor {target}")
    os.link(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    destination = (root / args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    selected: dict[str, Path] = {}
    for value in args.source:
        source = (root / value).resolve()
        for path in sorted(source.glob("*.pt")):
            if path.name in selected:
                raise ValueError(f"duplicate tensor filename {path.name}")
            selected[path.name] = path
    if len(selected) != args.expected:
        raise ValueError(f"expected {args.expected} tensors, found {len(selected)}")
    for name, source in selected.items():
        hardlink_tensor(source, destination / name)
    manifest = {"samples": sorted(selected)}
    atomic_json(destination / "manifest.json", manifest)
    report = {
        "status": "pass",
        "destination": str(destination),
        "sources": args.source,
        "tensors": len(selected),
        "storage": "hardlinks",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

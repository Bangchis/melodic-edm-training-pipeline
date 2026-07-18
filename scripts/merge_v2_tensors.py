#!/usr/bin/env python3
"""Merge preprocessed v2 tensor shards using relative symlinks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from v2_common import atomic_json


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
        target = destination / name
        if target.is_symlink():
            if target.resolve() == source:
                continue
            target.unlink()
        elif target.exists():
            raise FileExistsError(f"refusing to overwrite non-symlink {target}")
        target.symlink_to(os.path.relpath(source, destination))
    manifest = {"samples": sorted(selected)}
    atomic_json(destination / "manifest.json", manifest)
    report = {
        "status": "pass",
        "destination": str(destination),
        "sources": args.source,
        "tensors": len(selected),
        "storage": "relative_symlinks",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a classifier-ready manifest from completed separated stems."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--preset", choices=("instrumental_full", "instrumental_clean"), default="instrumental_full")
    parser.add_argument("--vocal-manifest", default="data/vocal_manifest.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / (args.output or f"data/separated_{args.preset}_audio_manifest.jsonl")
    vocals = read_jsonl(root / args.vocal_manifest)
    targets = {r["sample_id"]: r for r in vocals if r.get("vocal_status") == "lyrics"}
    states: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "data").glob(f"separation_{args.preset}_part*.jsonl")):
        for row in read_jsonl(path):
            if row.get("separation_status") == "complete":
                states[row["sample_id"]] = row

    rows = []
    missing = []
    for sid, source in sorted(targets.items()):
        state = states.get(sid)
        if not state or not Path(state["separated_path"]).is_file() or not Path(state["preview_path"]).is_file():
            missing.append(sid)
            continue
        rows.append({
            **source,
            "original_canonical_path": source["canonical_path"],
            "canonical_path": state["separated_path"],
            "preview_path": state["preview_path"],
            "duration": state["duration"],
            "sample_rate": state["sample_rate"],
            "channels": state["channels"],
            "audio_valid": True,
            "separation_preset": args.preset,
            "pre_separation_vocal_status": source["vocal_status"],
        })
    atomic_jsonl(output, rows)
    report = {"preset": args.preset, "targets": len(targets), "ready": len(rows), "missing": missing}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

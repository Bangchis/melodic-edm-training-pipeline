#!/usr/bin/env python3
"""Merge two preprocessed train parts by symlink and verify exact record counts."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def expected_stems(index_path: Path) -> set[str]:
    """Return sample stems declared by one aggregate ACE-Step dataset JSON."""
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return {Path(row["filename"]).stem for row in data["samples"]}


def main() -> int:
    parts = [ROOT / "data" / "tensors_part0", ROOT / "data" / "tensors_part1"]
    indexes = [ROOT / "data" / "dataset_train_part0.json", ROOT / "data" / "dataset_train_part1.json"]
    expected = [expected_stems(path) for path in indexes]
    errors = []
    for index, (directory, stems) in enumerate(zip(parts, expected)):
        actual = {path.stem for path in directory.glob("*.pt") if not path.name.endswith(".tmp.pt")}
        missing = sorted(stems - actual)
        extra = sorted(actual - stems)
        if missing or extra:
            errors.append({"part": index, "missing": missing, "extra": extra})

    output = ROOT / "data" / "tensors_all"
    output.mkdir(parents=True, exist_ok=True)
    wanted = expected[0] | expected[1]
    if expected[0] & expected[1]:
        errors.append({"reason": "sample_collision_between_parts"})
    if not errors:
        for directory in parts:
            for source in directory.glob("*.pt"):
                if source.name.endswith(".tmp.pt"):
                    continue
                target = output / source.name
                if target.is_symlink() and target.resolve() != source.resolve():
                    target.unlink()
                if not target.exists():
                    target.symlink_to(source)
        for path in output.glob("*.pt"):
            if path.stem not in wanted and path.is_symlink():
                path.unlink()

    report = {
        "status": "pass" if not errors else "failed",
        "part0_expected": len(expected[0]),
        "part1_expected": len(expected[1]),
        "train_expected": len(wanted),
        "merged_tensors": len(list(output.glob("*.pt"))),
        "deduplication_performed": False,
        "errors": errors,
    }
    report_path = ROOT / "data" / "tensor_merge_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

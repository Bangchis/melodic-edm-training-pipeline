#!/usr/bin/env python3
"""Apply the absolute V2 listening gate to one MOSS score report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2_common import atomic_json
from v2_listening_quality import summarize_quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"listening score report has not passed technical validation: {report_path}")
    quality = summarize_quality(report.get("results", []))
    quality["source_report"] = str(report_path.relative_to(root))
    atomic_json(output_path, quality)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0 if quality["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

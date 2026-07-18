#!/usr/bin/env python3
"""Validate complete 231-record multi-view audible-claim consensus."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from annotate_moss_music import MODEL_REVISION
from v2_common import atomic_json, file_sha256, object_sha256, read_jsonl
from verify_v2_audio_claims_moss import extract_instrument_claims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    claim_total = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            annotation = json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))
            claims = extract_instrument_claims(annotation)
            audio_hash = file_sha256(Path(row["final_audio_path"]))
            expected_input = object_sha256({"audio": audio_hash, "claims": claims})
            record = json.loads(
                (root / "data_v2" / "claim_consensus" / f"{sample_id}.json").read_text(encoding="utf-8")
            )
            if record.get("model_revision") != MODEL_REVISION:
                raise ValueError("model_revision_mismatch")
            if record.get("audio_sha256") != audio_hash:
                raise ValueError("audio_sha256_mismatch")
            if record.get("input_sha256") != expected_input:
                raise ValueError("input_sha256_mismatch")
            if record.get("claims") != claims:
                raise ValueError("claim_list_mismatch")
            decisions = record.get("decisions")
            if not isinstance(decisions, list) or [item.get("claim") for item in decisions] != claims:
                raise ValueError("decision_coverage_mismatch")
            for item in decisions:
                decision = str(item.get("decision") or "")
                if decision not in {"present", "absent", "uncertain"}:
                    raise ValueError(f"invalid_decision:{decision}")
                counts[decision] += 1
                claim_total += 1
            views = record.get("views")
            if claims and (not isinstance(views, dict) or set(views) != {"full_neutral", "full_challenge", "overview_montage"}):
                raise ValueError("view_coverage_mismatch")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}:{exc}"})
    report: dict[str, Any] = {
        "status": "pass" if not errors and len(rows) == 231 else "failed",
        "records": len(rows) - len(errors),
        "catalog_records": len(rows),
        "claim_total": claim_total,
        "decisions": dict(counts),
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "claim_consensus_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate complete MOSS caption repairs and apply only rows marked revise."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from annotate_moss_music import MODEL_REVISION
from repair_v2_annotations_moss import (
    CAPTION_COMPILER_REVISION,
    SCORE_FIELDS,
    exact_claim_asserted,
    parse_repair,
    qualified_claim_mentioned,
)
from v2_common import atomic_json, file_sha256, object_sha256, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data_v2" / "manifest.jsonl"), key=lambda row: row["sample_id"])
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    recommendations: Counter[str] = Counter()
    scores: dict[str, list[int]] = {field: [] for field in SCORE_FIELDS}
    staged_annotations: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        annotation_path = Path(row["v2_annotation_path"])
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            original = {
                str(item["type"]): str(item["text"])
                for item in annotation["caption_variants"]
            }
            record = json.loads(
                (root / "data_v2" / "caption_repairs" / f"{sample_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            repair, validation_errors = parse_repair(record.get("repair", {}))
            if validation_errors:
                raise ValueError(",".join(validation_errors))
            if record.get("model_revision") != MODEL_REVISION:
                raise ValueError("model_revision_mismatch")
            if record.get("caption_compiler_revision") != CAPTION_COMPILER_REVISION:
                raise ValueError("caption_compiler_revision_mismatch")
            if record.get("audio_sha256") != file_sha256(Path(row["final_audio_path"])):
                raise ValueError("audio_sha256_mismatch")
            if record.get("original_captions_sha256") != object_sha256(original):
                raise ValueError("original_captions_sha256_mismatch")
            decisions = record.get("claim_decisions")
            if not isinstance(decisions, list):
                raise ValueError("claim_decisions_missing")
            if record.get("claim_decisions_sha256") != object_sha256(decisions):
                raise ValueError("claim_decisions_sha256_mismatch")
            corrected_text = " ".join(repair["corrected_captions"].values()).casefold()
            for decision in decisions:
                claim = str(decision.get("claim") or "").casefold()
                resolution = str(decision.get("decision") or "")
                if not claim or resolution not in {"present", "absent", "uncertain"}:
                    raise ValueError("invalid_claim_decision")
                if resolution == "present" and not exact_claim_asserted(corrected_text, claim):
                    raise ValueError(f"verified_present_claim_missing:{claim}")
                if resolution == "absent" and claim in corrected_text:
                    raise ValueError(f"verified_absent_claim_retained:{claim}")
                if resolution == "uncertain" and exact_claim_asserted(corrected_text, claim):
                    raise ValueError(f"uncertain_claim_asserted_as_exact:{claim}")
                if resolution == "uncertain" and not qualified_claim_mentioned(corrected_text, claim):
                    raise ValueError(f"uncertain_claim_qualified_token_missing:{claim}")
            recommendations[repair["recommendation"]] += 1
            for field in SCORE_FIELDS:
                scores[field].append(repair["scores"][field])
            selected = repair["corrected_captions"]
            changed = object_sha256(selected) != object_sha256(original)
            if changed:
                annotation["caption"] = selected["canonical"]
                annotation["caption_variants"] = [
                    {"type": name, "text": selected[name]}
                    for name in ("canonical", "composition", "production")
                ]
                annotation["master_annotation"]["merged_captions"] = selected
            annotation["caption_repair"] = {
                "model": record["model_id"],
                "model_revision": record["model_revision"],
                "caption_compiler_revision": record["caption_compiler_revision"],
                "repaired_at": record["repaired_at"],
                "recommendation": repair["recommendation"],
                "original_scores": repair["scores"],
                "unsupported_claims": repair["unsupported_claims"],
                "claim_decisions": decisions,
                "claim_decisions_sha256": record["claim_decisions_sha256"],
                "original_captions_sha256": record["original_captions_sha256"],
                "applied": changed,
            }
            staged_annotations.append((sample_id, annotation))
            results.append({
                "sample_id": sample_id,
                "recommendation": repair["recommendation"],
                "applied": changed,
            })
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}:{exc}"})
    dimension_means = {
        field: round(mean(values), 3) if values else 0.0
        for field, values in scores.items()
    }
    status = "pass" if not errors and len(results) == len(rows) else "failed"
    backup_path: str | None = None
    if status == "pass":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging = root / "data_v2" / f"annotations_repaired_staging_{timestamp}"
        staging.mkdir(parents=True, exist_ok=False)
        for sample_id, annotation in staged_annotations:
            atomic_json(staging / f"{sample_id}.json", annotation)
        current = root / "data_v2" / "annotations"
        backup = root / "data_v2" / f"annotations_before_caption_repair_{timestamp}"
        os.replace(current, backup)
        os.replace(staging, current)
        backup_path = str(backup)
        stale = root / "data_v2" / f"pre_caption_repair_artifacts_{timestamp}"
        stale.mkdir(parents=True, exist_ok=False)
        for name in (
            "tensors_all",
            "tensors_train",
            "tensors_train_part0",
            "tensors_train_part1",
            "tensors_validation",
            "tensors_validation_raw",
            "tensor_validation_report.json",
            "metadata_upload_report.json",
            "annotation_quality_audit.json",
            "annotation_fidelity_audit.json",
        ):
            source = root / "data_v2" / name
            if source.exists():
                os.replace(source, stale / name)
        dataset_rebuild = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_v2_dataset.py"), "--project-root", str(root)],
            check=False,
        )
        if dataset_rebuild.returncode != 0:
            errors.append({"sample_id": "*", "reason": "post_repair_dataset_rebuild_failed"})
            status = "failed"
    report = {
        "status": status,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "records": len(results),
        "recommendations": dict(recommendations),
        "repairs_applied": sum(item["applied"] for item in results),
        "previous_annotations_backup": backup_path,
        "original_caption_dimension_means": dimension_means,
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "caption_repair_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

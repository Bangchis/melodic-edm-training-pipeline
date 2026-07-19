#!/usr/bin/env python3
"""Validate complete OpenRouter caption fusions and apply the corrected views."""
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

from repair_v2_annotations_moss import (
    CAPTION_COMPILER_REVISION,
    CAPTION_COMPILER_PROVIDER,
    DEFAULT_COMPILER_MODEL,
    SCORE_FIELDS,
    fusion_source_material,
    parse_repair,
    validate_claim_constraints,
)
from v2_common import (
    TRACK_STYLE_REFERENCE_REVISION,
    atomic_json,
    attach_track_style_reference,
    file_sha256,
    object_sha256,
    read_jsonl,
    track_style_reference,
    validate_track_style_caption_set,
)


def archive_stale_model_outputs(root: Path, timestamp: str) -> dict[str, Any]:
    """Archive every model artifact whose lineage predates the new captions."""
    archive = root / "outputs" / "archive" / f"v2-before-caption-fusion-{timestamp}"
    archived: list[dict[str, str]] = []
    for source in (
        root / "outputs" / "v2",
        root / "outputs" / "release" / "melodic-edm-core-v2-preview",
        root / "outputs" / "release" / "melodic-edm-core-v2",
    ):
        if not source.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / source.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite stale model archive {destination}")
        os.replace(source, destination)
        archived.append({"source": str(source), "archive": str(destination)})
    (root / "outputs" / "v2").mkdir(parents=True, exist_ok=True)
    report = {
        "status": "pass",
        "reason": f"caption_lineage_changed_to_{CAPTION_COMPILER_REVISION}",
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive) if archived else None,
        "archived": archived,
        "fresh_rank32_outputs_required": True,
    }
    atomic_json(root / "data_v2" / "downstream_reset_report.json", report)
    return report


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
            current_fusion_sources = fusion_source_material(annotation, original)
            current_fusion_hash = object_sha256(current_fusion_sources)
            record = json.loads(
                (root / "data_v2" / "caption_repairs" / f"{sample_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            repair, validation_errors = parse_repair(record.get("repair", {}))
            if validation_errors:
                raise ValueError(",".join(validation_errors))
            if record.get("caption_compiler_provider") != CAPTION_COMPILER_PROVIDER:
                raise ValueError("caption_compiler_provider_mismatch")
            if record.get("compiler_model_requested") != os.environ.get(
                "V2_CAPTION_COMPILER_MODEL", DEFAULT_COMPILER_MODEL
            ):
                raise ValueError("caption_compiler_model_mismatch")
            if record.get("caption_compiler_revision") != CAPTION_COMPILER_REVISION:
                raise ValueError("caption_compiler_revision_mismatch")
            if record.get("audio_sha256") != file_sha256(Path(row["final_audio_path"])):
                raise ValueError("audio_sha256_mismatch")
            if record.get("original_captions_sha256") != object_sha256(original):
                raise ValueError("original_captions_sha256_mismatch")
            if record.get("fusion_sources_sha256") != current_fusion_hash:
                raise ValueError("fusion_sources_sha256_mismatch")
            if record.get("fusion_sources") != current_fusion_sources:
                raise ValueError("fusion_sources_content_mismatch")
            style_reference = {
                "revision": TRACK_STYLE_REFERENCE_REVISION,
                "artist": " ".join(str(row.get("expected_artist") or "").split()),
                "title": " ".join(str(row.get("expected_title") or "").split()),
            }
            style_reference["prefix"] = track_style_reference(
                style_reference["artist"], style_reference["title"]
            )
            style_reference_hash = object_sha256(style_reference)
            if record.get("track_style_reference") != style_reference:
                raise ValueError("track_style_reference_content_mismatch")
            if record.get("track_style_reference_sha256") != style_reference_hash:
                raise ValueError("track_style_reference_sha256_mismatch")
            decisions = record.get("claim_decisions")
            if not isinstance(decisions, list):
                raise ValueError("claim_decisions_missing")
            if record.get("claim_decisions_sha256") != object_sha256(decisions):
                raise ValueError("claim_decisions_sha256_mismatch")
            corrected_text = " ".join(repair["corrected_captions"].values()).casefold()
            claim_errors = validate_claim_constraints(corrected_text, decisions)
            if claim_errors:
                raise ValueError(",".join(claim_errors))
            recommendations[repair["recommendation"]] += 1
            for field in SCORE_FIELDS:
                scores[field].append(repair["scores"][field])
            selected = attach_track_style_reference(
                repair["corrected_captions"],
                style_reference["artist"],
                style_reference["title"],
            )
            style_errors = validate_track_style_caption_set(
                selected, style_reference["artist"], style_reference["title"]
            )
            if style_errors:
                raise ValueError(",".join(style_errors))
            changed = object_sha256(selected) != object_sha256(original)
            annotation["caption"] = selected["canonical"]
            annotation["caption_variants"] = [
                {"type": name, "text": selected[name]}
                for name in ("canonical", "composition", "production")
            ]
            annotation["master_annotation"]["merged_captions"] = selected
            annotation["master_annotation"]["track_style_reference"] = style_reference
            annotation["master_annotation"]["caption_fusion"] = {
                "revision": record["caption_compiler_revision"],
                "sources_sha256": record["fusion_sources_sha256"],
                "uses_prior_per_track_annotation": True,
                "uses_independent_audio_analysis": True,
                "binding_multi_view_claim_decisions": True,
            }
            annotation["caption_repair"] = {
                "provider": record["caption_compiler_provider"],
                "model_requested": record["compiler_model_requested"],
                "model_resolved": record["compiler_model_resolved"],
                "caption_compiler_revision": record["caption_compiler_revision"],
                "repaired_at": record["repaired_at"],
                "recommendation": repair["recommendation"],
                "original_scores": repair["scores"],
                "unsupported_claims": repair["unsupported_claims"],
                "claim_decisions": decisions,
                "claim_decisions_sha256": record["claim_decisions_sha256"],
                "original_captions_sha256": record["original_captions_sha256"],
                "fusion_sources_sha256": record["fusion_sources_sha256"],
                "track_style_reference_revision": style_reference["revision"],
                "track_style_reference_sha256": style_reference_hash,
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
        model_reset = archive_stale_model_outputs(root, timestamp)
        stale = root / "data_v2" / f"pre_caption_repair_artifacts_{timestamp}"
        stale.mkdir(parents=True, exist_ok=False)
        for name in (
            "tensors_all",
            "tensors_train",
            "tensors_train_part0",
            "tensors_train_part1",
            "tensors_validation",
            "tensors_validation_raw",
            "tensors_all_unique",
            "tensors_train_unique",
            "tensors_validation_unique",
            "tensor_validation_report.json",
            "dedup_training_view_report.json",
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
        "caption_compiler_revision": CAPTION_COMPILER_REVISION,
        "caption_compiler_provider": CAPTION_COMPILER_PROVIDER,
        "caption_compiler_model": os.environ.get(
            "V2_CAPTION_COMPILER_MODEL", DEFAULT_COMPILER_MODEL
        ),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "records": len(results),
        "recommendations": dict(recommendations),
        "repairs_applied": sum(item["applied"] for item in results),
        "fusion_records": len(results),
        "fusion_sources": [
            "prior_per_track_annotation",
            "independent_audio_analysis",
            "binding_multi_view_claim_decisions",
        ],
        "track_style_reference_revision": TRACK_STYLE_REFERENCE_REVISION,
        "track_style_reference_records": len(results),
        "track_style_reference_scope": ["canonical", "composition", "production"],
        "previous_annotations_backup": backup_path,
        "stale_model_outputs_reset": model_reset if status == "pass" else None,
        "original_caption_dimension_means": dimension_means,
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "caption_repair_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Repair only captions rejected by the final audio-grounded fidelity gate."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repair_v2_annotations_moss import (
    DEFAULT_COMPILER_MODEL,
    openrouter_generate,
    parse_repair,
    validate_training_caption_policy,
)
from v2_common import (
    CAPTION_TYPES,
    TRACK_STYLE_REFERENCE_REVISION,
    atomic_json,
    attach_track_style_reference,
    detach_track_style_reference,
    extract_json_object,
    object_sha256,
    read_jsonl,
    track_style_reference,
    validate_track_style_caption_set,
)


FIDELITY_REPAIR_REVISION = "stratified-listening-correction-v1"


def needs_fidelity_repair(
    result: dict[str, Any], adjudication: dict[str, Any] | None = None
) -> bool:
    """Return whether one judged caption set violates the absolute score gate."""
    scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
    return (
        (adjudication or {}).get("force_caption_repair") is True
        or result.get("recommendation") == "reject"
        or any(int(scores.get(field, 0)) < 2 for field in (
            "audible_fidelity",
            "specificity",
            "melody_arrangement_accuracy",
            "production_accuracy",
        ))
    )


def validate_fidelity_caption_policy(captions: dict[str, str]) -> list[str]:
    """Reject wording that turns a real recurring motif into a static-loop instruction."""
    errors = validate_training_caption_policy(captions)
    for name, text in captions.items():
        if re.search(r"\bstandard\s+electronic\s+structure\b", text, re.IGNORECASE):
            errors.append(f"{name}_contains_generic_electronic_structure")
        if re.search(r"\brepeats?\s+throughout\b", text, re.IGNORECASE):
            errors.append(f"{name}_contains_unqualified_full_track_repetition")
    return errors


def validate_adjudication_caption_policy(
    captions: dict[str, str], adjudication: dict[str, Any] | None
) -> list[str]:
    """Enforce only the audible terms recorded by a targeted adjudication."""
    if not adjudication:
        return []
    text = " ".join(captions.values()).casefold()
    errors: list[str] = []
    for term in adjudication.get("required_caption_terms", []):
        if str(term).casefold() not in text:
            errors.append(f"adjudication_required_term_missing:{term}")
    for term in adjudication.get("forbidden_caption_terms", []):
        if str(term).casefold() in text:
            errors.append(f"adjudication_forbidden_term_present:{term}")
    return errors


def request_for(
    captions: dict[str, str],
    audit: dict[str, Any],
    adjudication: dict[str, Any] | None = None,
) -> str:
    """Build a text-only repair request grounded in the final listening evidence."""
    evidence_packet = {
        "scores": audit.get("scores", {}),
        "evidence": audit.get("evidence", {}),
        "unsupported_claims": audit.get("unsupported_claims", []),
        "recommendation": audit.get("recommendation"),
    }
    adjudication_text = ""
    if adjudication:
        adjudication_text = (
            "\nIndependent multi-source adjudication: "
            + json.dumps(adjudication, ensure_ascii=False)
            + "\nThe caption-conditioned listener contradicted itself on this record. Where its "
            "latest verdict conflicts with this independent adjudication, follow the adjudication "
            "guidance while keeping wording cautious and audible."
        )
    return (
        "Correct the three audio-only training captions for one instrumental track. The final "
        "waveform-listening audit below is authoritative over the current captions and over any "
        "older annotation. Preserve accurate song-specific properties, but remove or replace every "
        "unsupported claim using only the audible alternatives stated in the audit evidence. Do not "
        "infer title, artist, country, intended use, or named-artist style in these caption bodies. "
        "Do not invent exact acoustic instruments. Prefer concrete synth timbre, melody behavior, "
        "rhythm, sectional development and mix texture that the evidence actually supports. Avoid "
        "generic praise and the words clean, polished, masterpiece, professional, repetitive, "
        "cyclical, hypnotic, unchanging, or static loop. Canonical must contain 40-80 English words; "
        "Never write that a motif repeats throughout the track. When recurrence is audible, describe "
        "how it returns after a contrast, transition, changed ending, or layer change. "
        "composition and production must each contain 25-80 English words. The three views must be "
        "consistent but distinct. Return JSON only with integer scores 1-5 for audible_fidelity, "
        "specificity, melody_arrangement_accuracy and production_accuracy; an evidence object with "
        "those four keys; unsupported_claims as an array; recommendation keep or revise; and "
        "corrected_captions with exact keys canonical, composition and production. "
        "Required shape: "
        '{"audible_fidelity":1,"specificity":1,"melody_arrangement_accuracy":1,'
        '"production_accuracy":1,"evidence":{"audible_fidelity":"...","specificity":"...",'
        '"melody_arrangement_accuracy":"...","production_accuracy":"..."},'
        '"unsupported_claims":[],"recommendation":"revise","corrected_captions":'
        '{"canonical":"...","composition":"...","production":"..."}}.\nCurrent captions: '
        + json.dumps(captions, ensure_ascii=False)
        + "\nAuthoritative final listening audit: "
        + json.dumps(evidence_packet, ensure_ascii=False)
        + adjudication_text
    )


def archive_downstream(root: Path, timestamp: str) -> list[dict[str, str]]:
    """Archive caption-dependent tensors and reports while retaining audit cache."""
    destination = root / "data_v2" / f"pre_fidelity_repair_artifacts_{timestamp}"
    moved: list[dict[str, str]] = []
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
        "trainer_runtime_audit.json",
    ):
        source = root / "data_v2" / name
        if not source.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        os.replace(source, target)
        moved.append({"source": str(source), "archive": str(target)})
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument(
        "--model",
        default=os.environ.get("V2_CAPTION_COMPILER_MODEL", DEFAULT_COMPILER_MODEL),
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    audit_path = root / "data_v2" / "annotation_fidelity_audit.json"
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    adjudication_path = root / "configs" / "v2" / "audio_adjudication_overrides.json"
    adjudication_config = json.loads(adjudication_path.read_text(encoding="utf-8"))
    adjudications = adjudication_config.get("samples", {})
    audit_results = {
        str(item["sample_id"]): item
        for item in audit_report.get("results", [])
        if isinstance(item, dict) and item.get("sample_id")
    }
    selected = {
        sample_id: result
        for sample_id, result in audit_results.items()
        if needs_fidelity_repair(result, adjudications.get(sample_id))
    }
    if not selected:
        raise RuntimeError("fidelity audit contains no below-threshold caption set to repair")
    rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "data_v2" / "manifest.jsonl")
    }
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for fidelity caption repair")

    staged: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    repair_dir = root / "data_v2" / "fidelity_repairs"
    for index, sample_id in enumerate(sorted(selected), 1):
        row = rows[sample_id]
        audit = selected[sample_id]
        annotation_path = Path(row["v2_annotation_path"])
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        attached = {
            str(item["type"]): str(item["text"])
            for item in annotation["caption_variants"]
        }
        bodies = {
            name: detach_track_style_reference(
                attached[name],
                str(row.get("expected_artist") or ""),
                str(row.get("expected_title") or ""),
            )
            for name in CAPTION_TYPES
        }
        adjudication = adjudications.get(sample_id)
        request = request_for(bodies, audit, adjudication)
        last_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            try:
                response, metadata = openrouter_generate(
                    request,
                    api_key,
                    args.model,
                    args.max_new_tokens,
                    args.timeout,
                )
                repair, errors = parse_repair(extract_json_object(response))
                errors.extend(validate_fidelity_caption_policy(repair["corrected_captions"]))
                errors.extend(
                    validate_adjudication_caption_policy(
                        repair["corrected_captions"], adjudication
                    )
                )
                if errors:
                    raise ValueError(",".join(errors))
                selected_captions = attach_track_style_reference(
                    repair["corrected_captions"],
                    str(row.get("expected_artist") or ""),
                    str(row.get("expected_title") or ""),
                )
                style_errors = validate_track_style_caption_set(
                    selected_captions,
                    str(row.get("expected_artist") or ""),
                    str(row.get("expected_title") or ""),
                )
                if style_errors:
                    raise ValueError(",".join(style_errors))
                lineage = {
                    "revision": FIDELITY_REPAIR_REVISION,
                    "audit_result_sha256": object_sha256(audit),
                    "previous_captions_sha256": object_sha256(attached),
                    "corrected_audio_only_captions_sha256": object_sha256(
                        repair["corrected_captions"]
                    ),
                    "track_style_reference_revision": TRACK_STYLE_REFERENCE_REVISION,
                    "adjudication_revision": (
                        adjudication_config.get("revision") if adjudication else None
                    ),
                    "adjudication_sha256": object_sha256(adjudication) if adjudication else None,
                    "model_requested": metadata["requested_model"],
                    "model_resolved": metadata["resolved_model"],
                    "usage": metadata["usage"],
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                }
                annotation["caption"] = selected_captions["canonical"]
                annotation["caption_variants"] = [
                    {"type": name, "text": selected_captions[name]}
                    for name in CAPTION_TYPES
                ]
                annotation["master_annotation"]["merged_captions"] = selected_captions
                annotation["master_annotation"]["caption_fidelity_repair"] = lineage
                annotation["caption_fidelity_repair"] = {
                    **lineage,
                    "previous_scores": audit.get("scores", {}),
                    "previous_unsupported_claims": audit.get("unsupported_claims", []),
                }
                staged[sample_id] = annotation
                record = {
                    "schema_version": "2.0-fidelity-repair",
                    "sample_id": sample_id,
                    "lineage": lineage,
                    "audit_evidence": audit,
                    "repair": repair,
                }
                atomic_json(repair_dir / f"{sample_id}.json", record)
                records.append({"sample_id": sample_id, "attempt": attempt, **lineage})
                print(f"[{index}/{len(selected)}] {sample_id} PASS", flush=True)
                break
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                request += (
                    "\nPrevious output failed validation: "
                    + last_error
                    + ". Correct every error and return the complete JSON object only."
                )
        else:
            raise RuntimeError(f"{sample_id} fidelity repair failed: {last_error}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    annotations_dir = root / "data_v2" / "annotations"
    backup = root / "data_v2" / f"annotations_before_fidelity_repair_{timestamp}"
    shutil.copytree(annotations_dir, backup)
    for sample_id, annotation in staged.items():
        atomic_json(annotations_dir / f"{sample_id}.json", annotation)
    archived = archive_downstream(root, timestamp)
    rebuild = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_v2_dataset.py"), "--project-root", str(root)],
        check=False,
    )
    status = "pass" if rebuild.returncode == 0 else "failed"
    report = {
        "status": status,
        "revision": FIDELITY_REPAIR_REVISION,
        "source_audit": str(audit_path),
        "source_audit_sha256": object_sha256(audit_report),
        "records_repaired": len(records),
        "sample_ids": sorted(staged),
        "records": records,
        "annotations_backup": str(backup),
        "archived_caption_dependent_artifacts": archived,
        "dataset_rebuild_returncode": rebuild.returncode,
    }
    atomic_json(root / "data_v2" / "fidelity_repair_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

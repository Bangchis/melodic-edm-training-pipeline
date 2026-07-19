#!/usr/bin/env python3
"""Listen to a stratified V2 sample and score caption fidelity before training."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from annotate_moss_music import MODEL_REVISION, generate, load_runtime
from v2_common import (
    atomic_json,
    detach_track_style_reference,
    extract_json_object,
    object_sha256,
    read_jsonl,
)


SCORE_FIELDS = (
    "audible_fidelity",
    "specificity",
    "melody_arrangement_accuracy",
    "production_accuracy",
)
DEFAULT_MAX_TOKENS = 3600


def stratified_rows(rows: list[dict[str, Any]], per_group: int) -> list[dict[str, Any]]:
    """Select evenly spaced deterministic records from every catalog family."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        groups[str(row["sample_id"]).split("__", 1)[0]].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for values in groups.values():
        count = min(per_group, len(values))
        for index in range(count):
            offset = round(index * (len(values) - 1) / max(1, count - 1))
            selected[str(values[offset]["sample_id"])] = values[offset]
    # Always cover the only cross-parent near-duplicate caption pair found by the text audit.
    for row in rows:
        if row["sample_id"] in {"diversity__024", "thefatrat__028"}:
            selected[str(row["sample_id"])] = row
    return [selected[key] for key in sorted(selected)]


def request_for(captions: dict[str, str]) -> str:
    """Build a conservative audio-grounded annotation review prompt."""
    return (
        "Listen to the complete supplied instrumental audio and audit the three proposed training "
        "captions. Judge audible facts only. Do not use or infer title, artist, filename, popularity, "
        "or hidden metadata. A high fidelity score requires that instrument, rhythm, melody and "
        "arrangement claims are actually audible. A high specificity score requires useful facts "
        "that distinguish this track; generic praise such as clean, polished, cinematic, wide or "
        "classic EDM structure is not enough. Penalize invented traditional instruments, exact "
        "melodic claims that are unsupported, and copy-like boilerplate. Return JSON only with "
        "integer scores from 1 to 5 for audible_fidelity, specificity, "
        "melody_arrangement_accuracy and production_accuracy; an evidence object using the same "
        "four keys; unsupported_claims as a JSON array; and recommendation as one of keep, revise, "
        "or reject. Required shape: "
        '{"audible_fidelity":1,"specificity":1,"melody_arrangement_accuracy":1,'
        '"production_accuracy":1,"evidence":{"audible_fidelity":"...","specificity":"...",'
        '"melody_arrangement_accuracy":"...","production_accuracy":"..."},'
        '"unsupported_claims":[],"recommendation":"revise"}.\nCaptions: '
        + json.dumps(captions, ensure_ascii=False)
    )


def parse_review(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize one MOSS annotation review."""
    errors: list[str] = []
    scores: dict[str, int] = {}
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    normalized_evidence: dict[str, str] = {}
    for field in SCORE_FIELDS:
        try:
            score = int(value.get(field))
        except (TypeError, ValueError):
            score = 0
        if not 1 <= score <= 5:
            errors.append(f"{field}_outside_1_5")
        scores[field] = score
        detail = str(evidence.get(field) or "").strip()
        if not detail:
            errors.append(f"evidence_{field}_missing")
        normalized_evidence[field] = detail
    unsupported = value.get("unsupported_claims")
    if not isinstance(unsupported, list):
        errors.append("unsupported_claims_not_list")
        unsupported = []
    recommendation = str(value.get("recommendation") or "").strip().lower()
    if recommendation not in {"keep", "revise", "reject"}:
        errors.append("recommendation_invalid")
    return {
        "scores": scores,
        "evidence": normalized_evidence,
        "unsupported_claims": [str(item).strip() for item in unsupported if str(item).strip()],
        "recommendation": recommendation,
    }, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    sample = stratified_rows(rows, max(1, args.per_group))
    output = root / "data_v2" / "annotation_fidelity_audit.json"
    previous: dict[str, Any] = {}
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    cached = {
        (str(item.get("sample_id")), str(item.get("captions_sha256"))): item
        for item in previous.get("results", [])
        if isinstance(item, dict) and item.get("sample_id")
    }
    model, processor = load_runtime(root / "checkpoints" / "MOSS-Music-8B-Thinking")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(sample, 1):
        sample_id = str(row["sample_id"])
        annotation = json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))
        train_captions = {
            str(item["type"]): str(item["text"])
            for item in annotation["caption_variants"]
        }
        captions_sha256 = object_sha256(train_captions)
        captions = {
            name: detach_track_style_reference(
                text,
                str(row.get("expected_artist") or ""),
                str(row.get("expected_title") or ""),
            )
            for name, text in train_captions.items()
        }
        cache_key = (sample_id, captions_sha256)
        if cache_key in cached:
            results.append(cached[cache_key])
            print(f"[{index}/{len(sample)}] {sample_id} CACHED", flush=True)
            continue
        request = request_for(captions)
        last_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            response = generate(
                model,
                processor,
                Path(row["final_audio_path"]),
                request,
                max(900, args.max_tokens),
            )
            try:
                review, validation_errors = parse_review(extract_json_object(response))
                if validation_errors:
                    raise ValueError(",".join(validation_errors))
                results.append({
                    "sample_id": sample_id,
                    "parent_song_id": row["parent_song_id"],
                    "split": row["split"],
                    "captions_sha256": captions_sha256,
                    "judge_model": "OpenMOSS-Team/MOSS-Music-8B-Thinking",
                    "judge_model_revision": MODEL_REVISION,
                    **review,
                })
                atomic_json(output, {
                    "status": "in_progress",
                    "sample_size": len(sample),
                    "results": results,
                    "errors": [],
                })
                print(f"[{index}/{len(sample)}] {sample_id} PASS", flush=True)
                break
            except (KeyError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                request += "\nPrevious response failed validation: " + last_error + ". Return corrected JSON only."
        else:
            errors.append({"sample_id": sample_id, "reason": last_error})
    means = {
        field: round(mean(item["scores"][field] for item in results), 3)
        for field in SCORE_FIELDS
    } if results else {field: 0.0 for field in SCORE_FIELDS}
    recommendations = {
        name: sum(item["recommendation"] == name for item in results)
        for name in ("keep", "revise", "reject")
    }
    gate_pass = (
        len(results) == len(sample)
        and not errors
        and all(value >= 3.0 for value in means.values())
        and all(score >= 2 for item in results for score in item["scores"].values())
        and recommendations["reject"] == 0
    )
    report = {
        "status": "pass" if gate_pass else "failed",
        "scope": "deterministic stratified audio-grounded audit",
        "sample_size": len(sample),
        "catalog_records": len(rows),
        "score_scale": [1, 5],
        "dimensions": list(SCORE_FIELDS),
        "dimension_means": means,
        "recommendations": recommendations,
        "results": results,
        "errors": errors,
    }
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

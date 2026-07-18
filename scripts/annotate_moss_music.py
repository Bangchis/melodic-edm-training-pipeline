#!/usr/bin/env python3
"""Supplement existing annotations with pinned MOSS-Music-8B-Thinking."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_common import (
    atomic_json,
    atomic_jsonl,
    caption_map,
    extract_json_object,
    file_sha256,
    object_sha256,
    parent_song_id,
    read_jsonl,
    validate_caption_set,
)


MODEL_ID = "OpenMOSS-Team/MOSS-Music-8B-Thinking"
MODEL_REVISION = "2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3"
SOURCE_REVISION = "ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e"


def build_prompt(
    row: dict[str, Any], existing: dict[str, Any], mir: dict[str, Any], schema: dict[str, Any]
) -> str:
    """Build an audio-grounded correction and caption-merging request."""
    identity = {
        "title_for_identity_check_only": row.get("expected_title", ""),
        "artist_for_identity_check_only": row.get("expected_artist", ""),
        "version_for_identity_check_only": row.get("expected_version", ""),
        "instrumental": True,
    }
    mir_context = {
        field: mir.get(field)
        for field in ("bpm", "keyscale", "timesignature", "sections")
    }
    old_annotation = {
        field: existing.get(field)
        for field in (
            "primary_genre",
            "secondary_genres",
            "style_families",
            "moods",
            "main_instruments",
            "melody",
            "arrangement",
            "production",
            "canonical_caption",
            "caption_variants",
        )
    }
    return (
        "Listen to the complete supplied training audio. Improve the existing Gemini/Qwen "
        "annotation by correcting unsupported claims and adding stable audible musical facts. "
        "Return only one JSON object matching the supplied schema. The result is a supplement, "
        "not a replacement for the audio evidence. Do not mention title, artist, channel, model, "
        "BPM, exact key, time signature, media use cases, quality hype, or named-artist style. "
        "Do not invent a traditional instrument when uncertain. Treat vocal chops as production "
        "texture and do not describe lyrics. Captions must be English, distinct, grounded, and use "
        "the exact keys canonical, composition and production. Canonical must contain 40-80 words; "
        "composition and production must each contain 25-80 words. Canonical summarizes the whole "
        "track. Composition prioritizes melody, motif, harmony and arrangement. Production "
        "prioritizes audible instruments, synths, bass, drums, texture and space. The captions "
        "should merge facts that remain correct in the existing annotation with your own listening. "
        "The top-level confidence field is mandatory and must be a numeric overall confidence "
        "strictly greater than 0 and at most 1. Every audible_facts field in the schema is mandatory. "
        "Do not wrap JSON in Markdown.\n"
        f"Identity metadata: {json.dumps(identity, ensure_ascii=False)}\n"
        f"Existing annotation: {json.dumps(old_annotation, ensure_ascii=False)}\n"
        f"MIR context (keep out of prose captions): {json.dumps(mir_context, ensure_ascii=False)}\n"
        f"Required schema: {json.dumps(schema, ensure_ascii=False)}"
    )


def validate_supplement(
    value: dict[str, Any], row: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Normalize and validate one generated MOSS supplement."""
    errors: list[str] = []
    if "confidence" not in value:
        errors.append("confidence_missing")
    try:
        confidence = float(value["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0
    except KeyError:
        confidence = 0.0
    if not 0.0 < confidence <= 1.0:
        errors.append("confidence_outside_open_0_1")
    facts = value.get("audible_facts")
    if not isinstance(facts, dict) or not facts:
        errors.append("audible_facts_missing")
        facts = {}
    required_facts = {
        "genre_and_style", "moods", "instruments_and_roles", "melody_and_motifs",
        "harmony", "rhythm", "arrangement_and_sections", "production",
        "uncertain_or_conflicting_facts",
    }
    missing_facts = sorted(required_facts - set(facts))
    if missing_facts:
        errors.append(f"audible_facts_fields_missing:{missing_facts}")
    normalized_facts: dict[str, Any] = dict(facts)
    for field in ("genre_and_style", "moods", "uncertain_or_conflicting_facts"):
        raw = facts.get(field)
        if isinstance(raw, str):
            lowered = raw.strip().casefold()
            if field == "uncertain_or_conflicting_facts" and lowered in (
                "", "none", "no conflicts", "not applicable", "n/a",
            ):
                items: list[str] = []
            else:
                items = [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]
        elif isinstance(raw, list):
            items = [str(item).strip() for item in raw if str(item).strip()]
        else:
            items = []
            errors.append(f"{field}_must_be_list_or_string")
        if field != "uncertain_or_conflicting_facts" and not items:
            errors.append(f"{field}_empty")
        normalized_facts[field] = items

    instruments = facts.get("instruments_and_roles")
    normalized_instruments = []
    if not isinstance(instruments, list):
        errors.append("instruments_and_roles_must_be_list")
        instruments = []
    for index, instrument in enumerate(instruments):
        if not isinstance(instrument, dict):
            errors.append(f"instrument_{index}_must_be_object")
            continue
        name = str(instrument.get("name") or "").strip()
        role = str(instrument.get("role") or "").strip()
        try:
            instrument_confidence = float(instrument["confidence"])
        except (KeyError, TypeError, ValueError):
            instrument_confidence = -1.0
        if not name:
            errors.append(f"instrument_{index}_name_missing")
        if not role:
            errors.append(f"instrument_{index}_role_missing")
        if not 0.0 <= instrument_confidence <= 1.0:
            errors.append(f"instrument_{index}_confidence_outside_0_1")
        normalized_instruments.append({
            "name": name,
            "role": role,
            "confidence": instrument_confidence,
        })
    if not normalized_instruments:
        errors.append("instruments_and_roles_empty")
    normalized_facts["instruments_and_roles"] = normalized_instruments

    for field in (
        "melody_and_motifs", "harmony", "rhythm",
        "arrangement_and_sections", "production",
    ):
        text = str(facts.get(field) or "").strip()
        if not text:
            errors.append(f"{field}_empty")
        normalized_facts[field] = text
    captions = caption_map(value.get("captions"))
    errors.extend(
        validate_caption_set(
            captions,
            artist=str(row.get("expected_artist") or ""),
            title=str(row.get("expected_title") or ""),
        )
    )
    return {
        "confidence": confidence,
        "audible_facts": normalized_facts,
        "captions": captions,
    }, errors


def load_runtime(model_path: Path):
    """Load the official MOSS-Music model and processor on one visible GPU."""
    import torch
    from src.modeling_moss_music import MossMusicModel
    from src.processing_moss_music import MossMusicProcessor

    model = MossMusicModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = MossMusicProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        enable_time_marker=True,
    )
    return model, processor


def generate(model: Any, processor: Any, audio_path: Path, prompt: str, max_tokens: int) -> str:
    """Run one deterministic MOSS-Music generation request."""
    import torch
    from src.audio_io import load_audio

    raw_audio = load_audio(str(audio_path), sample_rate=processor.config.mel_sr)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    input_length = inputs["input_ids"].shape[1]
    return processor.decode(generated[0, input_length:], skip_special_tokens=True)


def valid_existing(path: Path, row: dict[str, Any]) -> bool:
    """Return whether an existing output is complete and still valid."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        _, errors = validate_supplement(record["supplement"], row)
        return not errors and record.get("model_revision") == MODEL_REVISION
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def normalize_existing(path: Path, row: dict[str, Any]) -> bool:
    """Atomically normalize a valid stored supplement without rerunning MOSS."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        supplement, errors = validate_supplement(record["supplement"], row)
        if errors or record.get("model_revision") != MODEL_REVISION:
            return False
        if record["supplement"] != supplement:
            record["supplement"] = supplement
            record["schema_normalized_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, record)
        return True
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-path", default="checkpoints/MOSS-Music-8B-Thinking")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    root = Path(args.project_root).resolve()
    model_path = (root / args.model_path).resolve()
    pinned = (model_path / "PINNED_REVISION").read_text(encoding="utf-8").strip()
    if pinned != MODEL_REVISION:
        raise RuntimeError(f"MOSS model revision mismatch: {pinned}")

    rows = sorted(read_jsonl(root / "data" / "final_manifest.jsonl"), key=lambda row: row["sample_id"])
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    schema = json.loads(
        (root / "configs" / "v2" / "moss_annotation_schema.json").read_text(encoding="utf-8")
    )
    output_dir = root / "data_v2" / "moss_annotations"
    failure_dir = root / "data_v2" / "moss_failures"
    for row in rows:
        normalize_existing(output_dir / f"{row['sample_id']}.json", row)
    pending = [row for row in rows if not valid_existing(output_dir / f"{row['sample_id']}.json", row)]
    print(
        json.dumps({
            "shard": args.shard_index,
            "assigned": len(rows),
            "pending": len(pending),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
        }),
        flush=True,
    )
    if not pending:
        return 0

    model, processor = load_runtime(model_path)
    manifest: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(pending, 1):
        sample_id = str(row["sample_id"])
        audio_path = Path(row["final_audio_path"])
        failure_path = failure_dir / f"{sample_id}.json"
        repair_round = 0
        prior_failure_error = ""
        if failure_path.is_file():
            try:
                prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
                repair_round = int(prior_failure.get("failure_count", 1))
                prior_failure_error = str(prior_failure.get("error") or "")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                repair_round = 1
        old_record = json.loads(
            (root / "data" / "annotations" / f"{sample_id}.json").read_text(encoding="utf-8")
        )
        old_annotation = old_record["annotation"]
        mir = json.loads((root / "data" / "mir" / f"{sample_id}.json").read_text(encoding="utf-8"))
        prompt = build_prompt(row, old_annotation, mir, schema)
        if repair_round:
            prompt += (
                f"\nRepair round {repair_round}: a prior complete run exhausted retries with "
                f"{prior_failure_error}. Do not reuse that response. Keep canonical between "
                "50 and 70 words and both composition and production between 30 and 60 words. "
                "Count conservatively and return a newly worded complete JSON object."
            )
        last_error = "unknown"
        last_response_hash = ""
        for attempt in range(1, args.max_attempts + 1):
            response = generate(model, processor, audio_path, prompt, args.max_new_tokens)
            last_response_hash = object_sha256(response)
            try:
                parsed = extract_json_object(response)
                supplement, errors = validate_supplement(parsed, row)
                if errors:
                    raise ValueError(",".join(errors))
                record = {
                    "schema_version": "2.0",
                    "sample_id": sample_id,
                    "parent_song_id": parent_song_id(row),
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "source_revision": SOURCE_REVISION,
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                    "audio_sha256": file_sha256(audio_path),
                    "existing_annotation_sha256": object_sha256(old_annotation),
                    "raw_response_sha256": last_response_hash,
                    "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                    "supplement": supplement,
                }
                atomic_json(output_dir / f"{sample_id}.json", record)
                failure_path.unlink(missing_ok=True)
                manifest.append({
                    "sample_id": sample_id,
                    "status": "accepted",
                    "attempt": attempt,
                    "path": str(output_dir / f"{sample_id}.json"),
                })
                print(f"[{index}/{len(pending)}] {sample_id} PASS attempt={attempt}", flush=True)
                break
            except (ValueError, KeyError, TypeError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                prompt += (
                    "\nYour previous answer failed validation: " + last_error +
                    ". Return a corrected JSON object only."
                )
        else:
            failures += 1
            failure = {
                "sample_id": sample_id,
                "status": "failed",
                "failure_count": repair_round + 1,
                "error": last_error,
                "raw_response_sha256": last_response_hash,
            }
            atomic_json(failure_path, failure)
            manifest.append(failure)
            print(f"[{index}/{len(pending)}] {sample_id} FAILED {last_error}", flush=True)

    atomic_jsonl(
        root / "data_v2" / f"moss_manifest_part{args.shard_index}.jsonl",
        manifest,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    raise SystemExit(main())

#!/usr/bin/env python3
"""Repair V2 captions using audio-grounded claim consensus."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_moss_music import MODEL_ID, MODEL_REVISION, generate, load_runtime
from v2_common import (
    atomic_json,
    atomic_jsonl,
    caption_map,
    extract_json_object,
    file_sha256,
    object_sha256,
    read_jsonl,
    validate_caption_set,
)
from verify_v2_audio_claims_moss import CLAIM_DISCOVERY_TERMS


SCORE_FIELDS = (
    "audible_fidelity",
    "specificity",
    "melody_arrangement_accuracy",
    "production_accuracy",
)
CAPTION_COMPILER_REVISION = "audio-grounded-caption-compiler-v2.6"
FORBIDDEN_TRAINING_CAPTION_PATTERNS = {
    "embedded_bpm": re.compile(r"\b\d{2,3}\s*bpm\b", re.IGNORECASE),
    "embedded_time_signature": re.compile(r"\b[2-7]\s*/\s*(?:2|4|8|16)\b"),
    "embedded_exact_key": re.compile(
        r"\b(?:in|key(?:\s+is|\s+of)?)\s+[A-G](?:#|b)?\s+(?:major|minor)\b",
        re.IGNORECASE,
    ),
    "quality_hype": re.compile(
        r"\b(?:masterpiece|polished|professional|extremely beautiful|best song ever)\b",
        re.IGNORECASE,
    ),
    "generic_edm_structure": re.compile(
        r"\b(?:classic|standard|typical)\s+EDM\s+structure\b",
        re.IGNORECASE,
    ),
}


def exact_claim_asserted(text: str, claim: str) -> bool:
    """Return True for an exact-name assertion, excluding an explicit ``-like`` qualifier."""
    pattern = re.compile(rf"(?<![a-z]){re.escape(claim.casefold())}(?![a-z])")
    for match in pattern.finditer(text.casefold()):
        suffix = text.casefold()[match.end():]
        if suffix.startswith("-like") or suffix.startswith(" like"):
            continue
        return True
    return False


def qualified_claim_mentioned(text: str, claim: str) -> bool:
    """Keep an unresolved vocabulary token, but only with an explicit ``-like`` qualifier."""
    pattern = re.compile(
        rf"(?<![a-z]){re.escape(claim.casefold())}(?:-like|\s+like)(?![a-z])"
    )
    return bool(pattern.search(text.casefold()))


def validate_training_caption_policy(captions: dict[str, str]) -> list[str]:
    """Reject metadata leakage and boilerplate from final trainable captions."""
    errors: list[str] = []
    for caption_type, text in captions.items():
        for label, pattern in FORBIDDEN_TRAINING_CAPTION_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{caption_type}_contains_{label}")
    return errors


def unverified_new_claims(text: str, decisions: list[dict[str, Any]]) -> list[str]:
    """Find exact controlled names introduced after the audible claim-verification stage."""
    allowed = {str(item.get("claim") or "").casefold() for item in decisions}
    for item in decisions:
        claim = str(item.get("claim") or "")
        alternative = str(item.get("audible_alternative") or "")
        allowed.update(
            term for term in CLAIM_DISCOVERY_TERMS
            if exact_claim_asserted(claim, term) or exact_claim_asserted(alternative, term)
        )
    return [
        term for term in CLAIM_DISCOVERY_TERMS
        if term not in allowed and exact_claim_asserted(text, term)
    ]


def request_for(captions: dict[str, str], decisions: list[dict[str, Any]]) -> str:
    """Ask MOSS to compile captions while obeying multi-view audible decisions."""
    return (
        "Listen to the complete supplied instrumental audio and compile accurate, prompt-useful "
        "training captions. Exact named instruments are valuable and must remain specific when the "
        "multi-view verifier marks them present. Remove claims marked absent. For claims marked "
        "uncertain, do not assert the physical instrument as fact, but preserve the exact vocabulary "
        "token at least once with a -like qualifier, followed by its audible alternative; for example, "
        "pipa-like plucked lead or dizi-like airy flute lead. Never replace an unresolved named claim "
        "with only a generic phrase, because the qualified vocabulary remains useful for prompt "
        "conditioning. Do not introduce any new exact named instrument that is absent from the "
        "binding decisions or their audible alternatives; use a non-physical timbre description "
        "instead. Never infer title, "
        "artist, country, channel or intended media use. Avoid quality hype and "
        "boilerplate such as clean, polished, masterpiece, classic EDM structure, or professional. "
        "Describe stable audible genre/style, mood, concrete melody or motif behavior, rhythm, "
        "arrangement development and production texture. Do not include BPM, exact key, time "
        "signature, title, artist, or named-artist style. Return English captions with exact keys "
        "canonical, composition and production. Canonical must be 40-80 words; composition and "
        "production must each be 25-80 words. They must be distinct and grounded. Return JSON only. "
        "Score the original proposals independently from 1 to 5 for audible_fidelity, specificity, "
        "melody_arrangement_accuracy and production_accuracy. Include short evidence for each, an "
        "unsupported_claims array, recommendation keep or revise, and corrected_captions. Even when "
        "recommendation is keep, corrected_captions is mandatory and may equal the original. "
        "Required shape: "
        '{"audible_fidelity":1,"specificity":1,"melody_arrangement_accuracy":1,'
        '"production_accuracy":1,"evidence":{"audible_fidelity":"...","specificity":"...",'
        '"melody_arrangement_accuracy":"...","production_accuracy":"..."},'
        '"unsupported_claims":[],"recommendation":"revise","corrected_captions":'
        '{"canonical":"...","composition":"...","production":"..."}}.\nUntrusted proposals: '
        + json.dumps(captions, ensure_ascii=False)
        + "\nBinding multi-view claim decisions: "
        + json.dumps(decisions, ensure_ascii=False)
    )


def parse_repair(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate one repair response and its corrected captions."""
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
    if recommendation not in {"keep", "revise"}:
        errors.append("recommendation_invalid")
    captions = caption_map(value.get("corrected_captions"))
    errors.extend(validate_caption_set(captions))
    errors.extend(validate_training_caption_policy(captions))
    return {
        "scores": scores,
        "evidence": normalized_evidence,
        "unsupported_claims": [str(item).strip() for item in unsupported if str(item).strip()],
        "recommendation": recommendation,
        "corrected_captions": captions,
    }, errors


def existing_valid(
    path: Path,
    audio_hash: str,
    caption_hash: str,
    decisions_hash: str,
) -> bool:
    """Return whether a resumable repair still matches exact input audio and captions."""
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        base_valid = (
            value.get("model_revision") == MODEL_REVISION
            and value.get("caption_compiler_revision") == CAPTION_COMPILER_REVISION
            and value.get("audio_sha256") == audio_hash
            and value.get("original_captions_sha256") == caption_hash
            and not parse_repair(value.get("repair", {}))[1]
        )
        return base_valid and value.get("claim_decisions_sha256") == decisions_hash
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data_v2" / "manifest.jsonl"), key=lambda row: row["sample_id"])
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    output_dir = root / "data_v2" / "caption_repairs"
    pending: list[tuple[dict[str, Any], dict[str, str], str, str, list[dict[str, Any]], str]] = []
    for row in rows:
        annotation = json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))
        captions = {
            str(item["type"]): str(item["text"])
            for item in annotation["caption_variants"]
        }
        audio_hash = file_sha256(Path(row["final_audio_path"]))
        caption_hash = object_sha256(captions)
        consensus_path = root / "data_v2" / "claim_consensus" / f"{row['sample_id']}.json"
        if not consensus_path.is_file():
            raise RuntimeError(f"missing multi-view claim consensus: {consensus_path}")
        consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
        decisions = [
            {
                "claim": item["claim"],
                "decision": item["decision"],
                "audible_alternative": item.get("audible_alternative", ""),
            }
            for item in consensus["decisions"]
        ]
        decisions_hash = object_sha256(decisions)
        path = output_dir / f"{row['sample_id']}.json"
        if not existing_valid(path, audio_hash, caption_hash, decisions_hash):
            pending.append((row, captions, audio_hash, caption_hash, decisions, decisions_hash))
    print(json.dumps({
        "shard": args.shard_index,
        "assigned": len(rows),
        "pending": len(pending),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }), flush=True)
    if not pending:
        return 0
    model, processor = load_runtime(root / "checkpoints" / "MOSS-Music-8B-Thinking")
    manifest: list[dict[str, Any]] = []
    errors = 0
    for index, (row, captions, audio_hash, caption_hash, decisions, decisions_hash) in enumerate(pending, 1):
        sample_id = str(row["sample_id"])
        request = request_for(captions, decisions)
        last_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            response = generate(
                model, processor, Path(row["final_audio_path"]), request, args.max_new_tokens
            )
            try:
                repair, validation_errors = parse_repair(extract_json_object(response))
                corrected_text = " ".join(repair.get("corrected_captions", {}).values()).casefold()
                if any(item["decision"] != "present" for item in decisions):
                    if repair.get("recommendation") != "revise":
                        validation_errors.append("non_present_claims_require_revision")
                for decision in decisions:
                    claim = str(decision["claim"]).casefold()
                    if decision["decision"] == "present" and not exact_claim_asserted(corrected_text, claim):
                        validation_errors.append(f"verified_present_claim_missing:{claim}")
                    if decision["decision"] == "absent" and claim in corrected_text:
                        validation_errors.append(f"verified_absent_claim_retained:{claim}")
                    if decision["decision"] == "uncertain" and exact_claim_asserted(corrected_text, claim):
                        validation_errors.append(f"uncertain_claim_asserted_as_exact:{claim}")
                    if decision["decision"] == "uncertain" and not qualified_claim_mentioned(corrected_text, claim):
                        validation_errors.append(f"uncertain_claim_qualified_token_missing:{claim}")
                for claim in unverified_new_claims(corrected_text, decisions):
                    validation_errors.append(f"unverified_new_claim_introduced:{claim}")
                if validation_errors:
                    raise ValueError(",".join(validation_errors))
                record = {
                    "schema_version": "2.1-caption-repair",
                    "sample_id": sample_id,
                    "parent_song_id": row["parent_song_id"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "caption_compiler_revision": CAPTION_COMPILER_REVISION,
                    "repaired_at": datetime.now(timezone.utc).isoformat(),
                    "audio_sha256": audio_hash,
                    "original_captions_sha256": caption_hash,
                    "claim_decisions_sha256": decisions_hash,
                    "claim_decisions": decisions,
                    "repair": repair,
                }
                atomic_json(output_dir / f"{sample_id}.json", record)
                manifest.append({"sample_id": sample_id, "status": "pass", "attempt": attempt})
                print(f"[{index}/{len(pending)}] {sample_id} PASS {repair['recommendation']}", flush=True)
                break
            except (KeyError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                request += "\nPrevious response failed validation: " + last_error + ". Return corrected JSON only."
        else:
            errors += 1
            manifest.append({"sample_id": sample_id, "status": "failed", "reason": last_error})
            print(f"[{index}/{len(pending)}] {sample_id} FAILED {last_error}", flush=True)
    atomic_jsonl(root / "data_v2" / f"caption_repair_manifest_part{args.shard_index}.jsonl", manifest)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

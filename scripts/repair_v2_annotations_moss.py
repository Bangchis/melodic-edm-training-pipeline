#!/usr/bin/env python3
"""Compile V2 captions through OpenRouter from audio-grounded claim consensus."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_common import (
    CAPTION_TYPES,
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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_COMPILER_MODEL = "google/gemini-3.1-flash-lite"
CAPTION_COMPILER_PROVIDER = "openrouter"
CAPTION_COMPILER_REVISION = "openrouter-per-track-prior-audio-fusion-v2.8"
REPAIR_RESPONSE_SCHEMA = {
    "name": "per_track_caption_fusion",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            *SCORE_FIELDS,
            "evidence",
            "unsupported_claims",
            "recommendation",
            "corrected_captions",
        ],
        "properties": {
            **{
                field: {"type": "integer", "minimum": 1, "maximum": 5}
                for field in SCORE_FIELDS
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SCORE_FIELDS),
                "properties": {
                    field: {"type": "string", "minLength": 1}
                    for field in SCORE_FIELDS
                },
            },
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "recommendation": {"type": "string", "enum": ["keep", "revise"]},
            "corrected_captions": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CAPTION_TYPES),
                "properties": {
                    name: {"type": "string", "minLength": 1}
                    for name in CAPTION_TYPES
                },
            },
        },
    },
}
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


def prior_prompt_material(annotation: dict[str, Any]) -> dict[str, Any]:
    """Extract the old, song-specific prompt material without identity metadata.

    The base annotation is useful conditioning evidence, but it is not trusted as
    audible truth.  Keeping this packet separate lets MOSS fuse it with an
    independent audio reading and lets downstream code prove that the prompt for
    one song was never substituted for another.
    """
    master = annotation.get("master_annotation")
    if not isinstance(master, dict):
        master = {}
    base = master.get("base_annotation")
    if not isinstance(base, dict):
        base = {}
    variants: dict[str, str] = {}
    for item in base.get("caption_variants", []):
        if not isinstance(item, dict):
            continue
        caption_type = str(item.get("type") or "").strip().casefold()
        if caption_type == "full":
            caption_type = "canonical"
        text = str(item.get("text") or "").strip()
        if caption_type in {*CAPTION_TYPES, "tags"} and text:
            variants[caption_type] = text
    packet = {
        "canonical_caption": str(base.get("canonical_caption") or "").strip(),
        "caption_variants": variants,
        "primary_genre": base.get("primary_genre"),
        "secondary_genres": base.get("secondary_genres"),
        "style_families": base.get("style_families"),
        "moods": base.get("moods"),
        "main_instruments": base.get("main_instruments"),
        "melody": base.get("melody"),
        "arrangement": base.get("arrangement"),
        "production": base.get("production"),
    }
    return {
        key: value
        for key, value in packet.items()
        if value not in (None, "", [], {})
    }


def fusion_source_material(
    annotation: dict[str, Any], moss_captions: dict[str, str]
) -> dict[str, Any]:
    """Build the exact two-source packet used to compile one track's captions."""
    master = annotation.get("master_annotation")
    if not isinstance(master, dict):
        master = {}
    audible_facts = master.get("moss_music_supplement")
    if not isinstance(audible_facts, dict):
        audible_facts = {}
    return {
        "prior_per_track_annotation": prior_prompt_material(annotation),
        "independent_audio_analysis": {
            "audible_facts": audible_facts,
            "caption_proposals": {name: moss_captions[name] for name in CAPTION_TYPES},
        },
    }


def request_for(fusion_sources: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    """Ask a text LLM to fuse old per-track prompts with verified audio evidence."""
    return (
        "Compile three accurate, prompt-useful descriptions for this exact instrumental track. "
        "The independent audio analysis and multi-view claim decisions were produced by an audio "
        "listener before this text-only compilation step; treat them as the audible evidence and "
        "do not invent facts beyond them. Fuse both supplied source packets: "
        "(1) the old per-track annotation and prompt, which contains useful song-specific intent "
        "but may contain mistakes, and (2) the independent waveform-only MOSS analysis. Do not "
        "discard the old per-track prompt, and do not copy it blindly. Preserve its distinctive "
        "genre, mood, melody, arrangement and production properties when they are audible or not "
        "contradicted by the waveform. Prefer the independent audio evidence when the sources "
        "conflict. Never average the track into generic EDM boilerplate. Each corrected caption "
        "must describe this song rather than the dataset as a whole; composition and production "
        "must emphasize different concrete properties. Exact named instruments are valuable and "
        "must remain specific when the "
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
        '{"canonical":"...","composition":"...","production":"..."}}.\nFusion sources: '
        + json.dumps(fusion_sources, ensure_ascii=False)
        + "\nBinding multi-view claim decisions: "
        + json.dumps(decisions, ensure_ascii=False)
    )


def openrouter_generate(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    """Generate one strict JSON repair through OpenRouter without sending audio bytes."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative music-dataset caption compiler. Follow the binding "
                    "audio evidence and return only the requested JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "minimal", "exclude": True},
        "response_format": {"type": "json_schema", "json_schema": REPAIR_RESPONSE_SCHEMA},
        "provider": {"require_parameters": True},
    }
    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "melodic-edm-training-pipeline/2.0",
            "X-Title": "Melodic EDM Core V2 Caption Fusion",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"OpenRouter HTTP {error.code}: {detail}") from error
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response has no choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter response content is empty")
    return content, {
        "requested_model": model,
        "resolved_model": body.get("model") or model,
        "usage": body.get("usage") or {},
    }


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
    fusion_sources_hash: str,
    compiler_model: str,
) -> bool:
    """Return whether a resumable repair still matches exact input audio and captions."""
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        base_valid = (
            value.get("caption_compiler_provider") == CAPTION_COMPILER_PROVIDER
            and value.get("compiler_model_requested") == compiler_model
            and value.get("caption_compiler_revision") == CAPTION_COMPILER_REVISION
            and value.get("audio_sha256") == audio_hash
            and value.get("original_captions_sha256") == caption_hash
            and value.get("fusion_sources_sha256") == fusion_sources_hash
            and object_sha256(value.get("fusion_sources")) == fusion_sources_hash
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
    parser.add_argument("--model", default=os.environ.get("V2_CAPTION_COMPILER_MODEL", DEFAULT_COMPILER_MODEL))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data_v2" / "manifest.jsonl"), key=lambda row: row["sample_id"])
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    output_dir = root / "data_v2" / "caption_repairs"
    pending: list[
        tuple[
            dict[str, Any], dict[str, str], str, str, list[dict[str, Any]], str,
            dict[str, Any], str,
        ]
    ] = []
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
        fusion_sources = fusion_source_material(annotation, captions)
        fusion_sources_hash = object_sha256(fusion_sources)
        path = output_dir / f"{row['sample_id']}.json"
        if not existing_valid(
            path, audio_hash, caption_hash, decisions_hash, fusion_sources_hash, args.model
        ):
            pending.append((
                row, captions, audio_hash, caption_hash, decisions, decisions_hash,
                fusion_sources, fusion_sources_hash,
            ))
    print(json.dumps({
        "shard": args.shard_index,
        "assigned": len(rows),
        "pending": len(pending),
        "provider": CAPTION_COMPILER_PROVIDER,
        "model": args.model,
        "compiler_revision": CAPTION_COMPILER_REVISION,
    }), flush=True)
    if not pending:
        return 0
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for caption fusion")
    manifest: list[dict[str, Any]] = []
    errors = 0
    for index, (
        row, captions, audio_hash, caption_hash, decisions, decisions_hash,
        fusion_sources, fusion_sources_hash,
    ) in enumerate(pending, 1):
        sample_id = str(row["sample_id"])
        request = request_for(fusion_sources, decisions)
        last_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            response = ""
            call_metadata: dict[str, Any] = {}
            try:
                response, call_metadata = openrouter_generate(
                    request, api_key, args.model, args.max_new_tokens, args.timeout
                )
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
                    "caption_compiler_provider": CAPTION_COMPILER_PROVIDER,
                    "compiler_model_requested": call_metadata["requested_model"],
                    "compiler_model_resolved": call_metadata["resolved_model"],
                    "caption_compiler_revision": CAPTION_COMPILER_REVISION,
                    "repaired_at": datetime.now(timezone.utc).isoformat(),
                    "audio_sha256": audio_hash,
                    "original_captions_sha256": caption_hash,
                    "fusion_sources": fusion_sources,
                    "fusion_sources_sha256": fusion_sources_hash,
                    "claim_decisions_sha256": decisions_hash,
                    "claim_decisions": decisions,
                    "raw_response_sha256": object_sha256(response),
                    "usage": call_metadata["usage"],
                    "repair": repair,
                }
                atomic_json(output_dir / f"{sample_id}.json", record)
                manifest.append({"sample_id": sample_id, "status": "pass", "attempt": attempt})
                print(f"[{index}/{len(pending)}] {sample_id} PASS {repair['recommendation']}", flush=True)
                break
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                if isinstance(exc, RuntimeError):
                    if attempt < max(1, args.attempts):
                        time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                request += (
                    "\nPrevious response failed validation: "
                    + last_error
                    + ". Return corrected JSON only."
                )
        else:
            errors += 1
            manifest.append({"sample_id": sample_id, "status": "failed", "reason": last_error})
            print(f"[{index}/{len(pending)}] {sample_id} FAILED {last_error}", flush=True)
    atomic_jsonl(root / "data_v2" / f"caption_repair_manifest_part{args.shard_index}.jsonl", manifest)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify audible instrument claims with multi-view MOSS consensus.

This is deliberately a claim verifier, not an instrument blacklist. Exact names
are preserved when independent full-track prompts and excerpt evidence agree.
Conflicting evidence is recorded as uncertain for the caption compiler.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_moss_music import MODEL_ID, MODEL_REVISION, generate, load_runtime
from v2_common import atomic_json, extract_json_object, file_sha256, object_sha256, read_jsonl


# Used only to discover free-text claims that are not present in the structured
# instrument lists. Membership here never means that a claim is rejected.
CLAIM_DISCOVERY_TERMS = (
    "pipa", "guzheng", "dizi", "erhu", "piano", "guitar", "bass guitar",
    "strings", "orchestral strings", "brass", "choir", "flute", "violin",
    "cello", "saxophone", "trumpet", "harp", "marimba", "xylophone",
    "synth lead", "synth pluck", "supersaw", "sub bass", "synth bass",
    "electronic drums", "drum machine", "vocal chops", "choir texture",
    "bells", "taiko", "xiao", "acoustic guitar", "electric guitar", "percussion",
    "synthesizer", "drums", "bass", "woodwinds", "synth texture", "synth pad",
    "recorder", "music box", "glockenspiel", "orchestral hits", "keyboard",
)
GENERIC_NAMES = {
    "", "unknown", "instrument", "instruments", "traditional instruments",
    "traditional chinese instruments", "electronic elements",
}
VIEW_NAMES = ("full_neutral", "full_challenge", "overview_montage")
CLAIM_VERIFIER_REVISION = "multi-view-audio-claims-v2.7"
MIGRATABLE_CLAIM_VERIFIER_REVISIONS = {
    "multi-view-audio-claims-v2.5",
    "multi-view-audio-claims-v2.6",
}


def evidence_explicitly_denies_presence(text: str) -> bool:
    """Detect a narrow set of contradictions in an asserted-present review.

    MOSS occasionally emits ``verdict=present, confidence=0`` while its prose
    says that no such sound is audible.  Confidence is confidence in the chosen
    verdict, not a probability that the claim is present, so those objects must
    be retried instead of silently entering consensus as ambiguous evidence.
    This intentionally recognizes only explicit negation phrases; nuanced prose
    remains ``uncertain`` rather than being reinterpreted here.
    """
    normalized = re.sub(r"\s+", " ", str(text or "").strip().casefold())
    return bool(
        re.search(
            r"\b(?:there (?:is|are) no|contains? no|has no|without any|"
            r"no (?:clear |audible |distinct )?|not (?:clearly )?audible|"
            r"cannot be heard|can't be heard|absent|inaudible)\b",
            normalized,
        )
    )


def normalize_claim(value: Any) -> str:
    """Normalize a structured instrument label without broadening its meaning."""
    text = re.sub(r"[_-]+", " ", str(value or "").strip().casefold())
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bsynthesizers\b", "synthesizer", text)
    text = re.sub(r"\bsynth textures\b", "synth texture", text)
    text = re.sub(r"\bsynth(?:esizer)? pads?\b", "synth pad", text)
    text = re.sub(r"\bviolins\b", "violin", text)
    text = re.sub(r"\bcellos\b", "cello", text)
    text = re.sub(r"\bstring sections?\b", "orchestral strings", text)
    text = re.sub(r"\bdrum kits?\b", "drums", text)
    if re.fullmatch(r"bass\s*\(\s*synthesizer\s*\)?", text):
        text = "synth bass"
    if "vocal" in text and "chop" in text:
        text = "vocal chops"
    if text == "synth":
        text = "synthesizer"
    if text == "string":
        text = "strings"
    if text == "bell":
        text = "bells"
    return text


def extract_instrument_claims(annotation: dict[str, Any]) -> list[str]:
    """Collect exact audible instrument/timbre claims from structured and prose fields."""
    master = annotation.get("master_annotation") if isinstance(annotation.get("master_annotation"), dict) else {}
    candidates: list[str] = []
    for container, key in (
        (master.get("base_annotation", {}), "main_instruments"),
        (master.get("moss_music_supplement", {}), "instruments_and_roles"),
    ):
        if not isinstance(container, dict):
            continue
        values = container.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                raw_name = str(item.get("name") or "")
                claim = normalize_claim(raw_name)
                qualified = bool(re.search(r"(?:-like|\blike\b)", raw_name, re.IGNORECASE))
                composite = (
                    "/" in raw_name
                    or "," in raw_name
                    or "(" in raw_name
                    or ")" in raw_name
                    or len(claim.split()) > 4
                )
                if qualified:
                    # Keep only an unqualified source outside parenthetical ``...-like`` text.
                    unqualified = re.sub(
                        r"\([^)]*(?:-like|\blike\b)[^)]*\)", "", raw_name,
                        flags=re.IGNORECASE,
                    )
                    base = normalize_claim(unqualified)
                    if base and base != claim:
                        candidates.append(base)
                elif composite:
                    discovered = [
                        term for term in sorted(CLAIM_DISCOVERY_TERMS, key=len, reverse=True)
                        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", claim)
                    ]
                    if discovered:
                        candidates.extend(discovered)
                    else:
                        candidates.extend(
                            normalize_claim(re.sub(r"[()]", " ", part))
                            for part in re.split(r"[/,]", raw_name)
                            if normalize_claim(re.sub(r"[()]", " ", part))
                        )
                else:
                    candidates.append(claim)

    captions = annotation.get("caption_variants")
    prose = " ".join(
        str(item.get("text") or "")
        for item in captions if isinstance(item, dict)
    ).casefold() if isinstance(captions, list) else ""
    # Prefer the most specific phrase when one term contains another.
    for term in sorted(CLAIM_DISCOVERY_TERMS, key=len, reverse=True):
        matches = list(re.finditer(rf"(?<![a-z]){re.escape(term)}(?![a-z])", prose))
        if any(not prose[match.end():].startswith("-like") for match in matches):
            candidates.append(term)

    output: list[str] = []
    for claim in candidates:
        if not claim or claim in GENERIC_NAMES or claim in output:
            continue
        # Do not keep both "strings" and the more specific "orchestral strings".
        if claim == "strings" and "orchestral strings" in output:
            continue
        if claim == "orchestral strings" and "strings" in output:
            output.remove("strings")
        output.append(claim)
    # Prefer exact subtypes over broad overlapping labels. This only avoids
    # duplicate questions; it does not determine the verdict of any remaining claim.
    supersedes = {
        "guitar": {"acoustic guitar", "electric guitar", "bass guitar"},
        "bass": {"sub bass", "synth bass", "bass guitar"},
        "drums": {"electronic drums", "acoustic drums", "taiko"},
        "synthesizer": {"synth lead", "synth pluck", "synth bass", "supersaw"},
        "choir": {"choir texture"},
    }
    return [
        claim for claim in output
        if not (claim in supersedes and supersedes[claim].intersection(output))
    ]


def parse_claim_review(value: dict[str, Any], expected: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Validate one exact-coverage MOSS claim review."""
    errors: list[str] = []
    raw = value.get("claims", value.get("results"))
    if isinstance(raw, dict):
        raw = [
            {"claim": claim, **(item if isinstance(item, dict) else {})}
            for claim, item in raw.items()
        ]
    if raw is None and all(claim in value for claim in expected):
        raw = [
            {"claim": claim, **(value[claim] if isinstance(value[claim], dict) else {})}
            for claim in expected
        ]
    if not isinstance(raw, list):
        return {"claims": []}, ["claims_not_list"]
    by_claim: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"claim_{index}_not_object")
            continue
        claim = normalize_claim(item.get("claim"))
        verdict = str(item.get("verdict") or "").strip().casefold()
        original_verdict = verdict
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = -1.0
        evidence = str(item.get("evidence") or "").strip()
        alternative = normalize_claim(item.get("audible_alternative"))
        if claim in by_claim:
            errors.append(f"duplicate_claim:{claim}")
            continue
        if verdict not in {"present", "absent", "uncertain"}:
            errors.append(f"invalid_verdict:{claim}")
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"invalid_confidence:{claim}")
        evidence_contradiction = (
            original_verdict == "present"
            and evidence_explicitly_denies_presence(evidence)
        )
        if evidence_contradiction:
            errors.append(f"present_verdict_contradicts_evidence:{claim}")
        normalization = ""
        if (
            original_verdict in {"present", "absent"}
            and 0.0 <= confidence < 0.5
            and not evidence_contradiction
        ):
            # A low-confidence binary verdict cannot support either side of the
            # consensus. Conservatively preserve the evidence as ``uncertain``
            # instead of spending repeated GPU calls asking for the same semantic fix.
            verdict = "uncertain"
            normalization = "low_confidence_binary_to_uncertain"
        if not evidence:
            errors.append(f"missing_evidence:{claim}")
        normalized_item = {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "audible_alternative": alternative,
        }
        if normalization:
            normalized_item.update({
                "original_verdict": original_verdict,
                "normalization": normalization,
            })
        by_claim[claim] = normalized_item
    expected_set = set(expected)
    if set(by_claim) != expected_set:
        errors.append(
            "claim_coverage_mismatch:missing=" + repr(sorted(expected_set - set(by_claim)))
            + ":extra=" + repr(sorted(set(by_claim) - expected_set))
        )
    return {"claims": [by_claim[name] for name in expected if name in by_claim]}, errors


def consensus_for(claim: str, reviews: dict[str, dict[str, Any]], threshold: float = 0.65) -> dict[str, Any]:
    """Resolve one claim; disagreement stays uncertain instead of being hard-deleted."""
    evidence = [
        {"view": view, **next(item for item in review["claims"] if item["claim"] == claim)}
        for view, review in reviews.items()
    ]
    positives = [item for item in evidence if item["verdict"] == "present" and item["confidence"] >= threshold]
    full_negatives = [
        item for item in evidence
        if item["view"] in {"full_neutral", "full_challenge"}
        and item["verdict"] == "absent" and item["confidence"] >= threshold
    ]
    if len(positives) >= 2 and not full_negatives:
        decision = "present"
    elif len(full_negatives) == 2 and not positives:
        decision = "absent"
    else:
        decision = "uncertain"
    alternatives = [item["audible_alternative"] for item in evidence if item["audible_alternative"]]
    alternative = max(
        sorted(set(alternatives)),
        key=lambda value: (alternatives.count(value), len(value)),
        default="",
    )
    return {
        "claim": claim,
        "decision": decision,
        "consensus_rule": "2-positive-no-full-negative_or_2-full-negative-no-positive",
        "audible_alternative": alternative,
        "evidence": evidence,
    }


def migrate_cached_consensus(
    cached: dict[str, Any],
    claims: list[str],
    audio_hash: str,
    input_hash: str,
) -> dict[str, Any] | None:
    """Upgrade old evidence only when it passes the current strict parser.

    The model output itself is immutable.  A cache with low-confidence binary
    verdicts or prose/verdict contradictions returns ``None`` and is listened to
    again; clean multi-view evidence can be re-resolved without spending another
    GPU pass over the same audio.
    """
    previous_revision = str(cached.get("claim_verifier_revision") or "")
    if previous_revision not in MIGRATABLE_CLAIM_VERIFIER_REVISIONS:
        return None
    if cached.get("model_revision") != MODEL_REVISION:
        return None
    if cached.get("audio_sha256") != audio_hash or cached.get("claims") != claims:
        return None
    raw_views = cached.get("views")
    if not isinstance(raw_views, dict) or set(raw_views) != set(VIEW_NAMES):
        return None
    reviews: dict[str, dict[str, Any]] = {}
    for view in VIEW_NAMES:
        review, errors = parse_claim_review(raw_views[view], claims)
        if errors:
            return None
        reviews[view] = review
    migrated = dict(cached)
    migrated.update({
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "claim_verifier_revision": CLAIM_VERIFIER_REVISION,
        "input_sha256": input_hash,
        "views": reviews,
        "decisions": [consensus_for(claim, reviews) for claim in claims],
        "migration": {
            "from_revision": previous_revision,
            "policy": "raw_multi_view_evidence_passed_current_strict_parser",
        },
    })
    return migrated


def request_for(claims: list[str], view: str) -> str:
    """Build independent prompts whose outputs can be compared claim by claim."""
    if view == "full_neutral":
        instruction = (
            "Listen to the complete instrumental track neutrally. Decide whether each named sound "
            "source is actually audible, not merely stylistically plausible."
        )
    elif view == "full_challenge":
        instruction = (
            "Listen to the complete instrumental track as a skeptical fact checker. Some proposed "
            "instrument names may be wrong, but do not reject a name that has unmistakable timbral "
            "evidence. Challenge every claim independently."
        )
    else:
        instruction = (
            "This audio is a chronological montage of excerpts from the beginning, middle and late "
            "track. Mark present only when the timbre is audible in these excerpts. Because this is "
            "not the complete track, use uncertain rather than absent when evidence is missing."
        )
    return (
        instruction
        + " Do not infer from title, artist, genre stereotypes or filenames. For every claim return "
        "verdict present, absent or uncertain; confidence from 0 to 1; short audible evidence; and "
        "an audible_alternative when the exact name is unsupported. Confidence measures confidence "
        "in the verdict you selected, not the probability that the sound is present. If you are not "
        "confident in present or absent, choose uncertain. Never encode absence as present with "
        "confidence zero, and never contradict the verdict in the evidence text. Preserve exact specific names "
        "when heard. Return JSON only with exactly this shape. Do not emit analysis, reasoning, "
        "markdown, or a thinking block before the object. The first output character must be { "
        "and the final output character must be }: "
        '{"claims":[{"claim":"pipa","verdict":"present","confidence":0.9,'
        '"evidence":"...","audible_alternative":""}]}. Return exactly one item for every '
        "input claim and copy each claim string exactly. Claims: "
        + json.dumps(claims, ensure_ascii=False)
    )


def duration_seconds(path: Path) -> float:
    """Read audio duration with ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def build_montage(audio: Path, output: Path) -> None:
    """Build a deterministic 60-second overview from early/middle/late excerpts."""
    audio_hash = file_sha256(audio)
    stamp = output.with_suffix(".source.json")
    if output.is_file() and stamp.is_file():
        try:
            if json.loads(stamp.read_text(encoding="utf-8")).get("audio_sha256") == audio_hash:
                return
        except (OSError, json.JSONDecodeError):
            pass
    length = duration_seconds(audio)
    excerpt = min(20.0, max(8.0, length / 4.0))
    starts = [
        max(0.0, min(length - excerpt, length * ratio - excerpt / 2.0))
        for ratio in (0.12, 0.50, 0.84)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.wav")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for start in starts:
        command += ["-ss", f"{start:.3f}", "-t", f"{excerpt:.3f}", "-i", str(audio)]
    command += [
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output)
    atomic_json(stamp, {"audio_sha256": audio_hash, "starts": starts, "excerpt_seconds": excerpt})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data_v2" / "manifest.jsonl"), key=lambda row: row["sample_id"])
    if args.sample_id:
        selected = set(args.sample_id)
        rows = [row for row in rows if row["sample_id"] in selected]
        missing = selected - {row["sample_id"] for row in rows}
        if missing:
            raise ValueError(f"unknown sample ids: {sorted(missing)}")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    work: list[tuple[dict[str, Any], dict[str, Any], list[str], str]] = []
    output_dir = root / "data_v2" / "claim_consensus"
    for row in rows:
        annotation = json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))
        claims = extract_instrument_claims(annotation)
        input_hash = object_sha256({
            "audio": file_sha256(Path(row["final_audio_path"])),
            "claims": claims,
            "claim_verifier_revision": CLAIM_VERIFIER_REVISION,
        })
        output = output_dir / f"{row['sample_id']}.json"
        if output.is_file():
            try:
                cached = json.loads(output.read_text(encoding="utf-8"))
                if (
                    cached.get("input_sha256") == input_hash
                    and cached.get("model_revision") == MODEL_REVISION
                    and cached.get("claim_verifier_revision") == CLAIM_VERIFIER_REVISION
                ):
                    continue
                migrated = migrate_cached_consensus(
                    cached,
                    claims,
                    file_sha256(Path(row["final_audio_path"])),
                    input_hash,
                )
                if migrated is not None:
                    atomic_json(output, migrated)
                    print(f"[CACHE] {row['sample_id']} migrated to {CLAIM_VERIFIER_REVISION}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if not claims:
            atomic_json(output, {
                "schema_version": "2.2-claim-consensus",
                "sample_id": row["sample_id"],
                "parent_song_id": row["parent_song_id"],
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "claim_verifier_revision": CLAIM_VERIFIER_REVISION,
                "audio_sha256": file_sha256(Path(row["final_audio_path"])),
                "input_sha256": input_hash,
                "claims": [],
                "views": {},
                "decisions": [],
            })
        else:
            work.append((row, annotation, claims, input_hash))
    print(json.dumps({"assigned": len(rows), "pending": len(work), "shard": args.shard_index}), flush=True)
    if not work:
        return 0
    model, processor = load_runtime(root / "checkpoints" / "MOSS-Music-8B-Thinking")
    failures = 0
    for index, (row, _annotation, claims, input_hash) in enumerate(work, 1):
        sample_id = str(row["sample_id"])
        audio = Path(row["final_audio_path"])
        montage = root / "data_v2" / "claim_montages" / f"{sample_id}.wav"
        build_montage(audio, montage)
        reviews: dict[str, dict[str, Any]] = {}
        failed = False
        for view in VIEW_NAMES:
            prompt = request_for(claims, view)
            source = montage if view == "overview_montage" else audio
            last_error = ""
            for attempt in range(1, max(1, args.attempts) + 1):
                response = generate(model, processor, source, prompt, 1600)
                try:
                    review, errors = parse_claim_review(extract_json_object(response), claims)
                    if errors:
                        raise ValueError(",".join(errors))
                    reviews[view] = review
                    break
                except (KeyError, TypeError, ValueError) as exc:
                    last_error = f"{type(exc).__name__}:{exc}"
                    prompt += (
                        "\nPrevious response failed exact validation: " + last_error
                        + ". Return corrected JSON only. Start immediately with {, end with }, "
                        "and emit no reasoning or markdown outside the object. For every present or "
                        "absent verdict, confidence must be at least 0.5. If confidence would be below "
                        "0.5, change that verdict to uncertain and describe the audible alternative."
                    )
            else:
                failed = True
                print(f"[{index}/{len(work)}] {sample_id} {view} FAILED {last_error}", flush=True)
                break
        if failed:
            failures += 1
            continue
        decisions = [consensus_for(claim, reviews) for claim in claims]
        record = {
            "schema_version": "2.2-claim-consensus",
            "sample_id": sample_id,
            "parent_song_id": row["parent_song_id"],
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "claim_verifier_revision": CLAIM_VERIFIER_REVISION,
            "audio_sha256": file_sha256(audio),
            "input_sha256": input_hash,
            "claims": claims,
            "views": reviews,
            "decisions": decisions,
        }
        atomic_json(output_dir / f"{sample_id}.json", record)
        counts = {name: sum(item["decision"] == name for item in decisions) for name in ("present", "absent", "uncertain")}
        print(f"[{index}/{len(work)}] {sample_id} PASS {counts}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

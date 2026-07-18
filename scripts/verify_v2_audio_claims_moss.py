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
)
GENERIC_NAMES = {
    "", "unknown", "instrument", "instruments", "traditional instruments",
    "traditional chinese instruments", "electronic elements",
}
VIEW_NAMES = ("full_neutral", "full_challenge", "overview_montage")
CLAIM_VERIFIER_REVISION = "multi-view-audio-claims-v2.3"


def normalize_claim(value: Any) -> str:
    """Normalize a structured instrument label without broadening its meaning."""
    text = re.sub(r"[_-]+", " ", str(value or "").strip().casefold())
    return re.sub(r"\s+", " ", text)


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
                claim = normalize_claim(item.get("name"))
                if "/" in str(item.get("name") or ""):
                    candidates.extend(normalize_claim(part) for part in str(item["name"]).split("/"))
                elif "," in str(item.get("name") or "") or len(claim.split()) > 4:
                    for term in sorted(CLAIM_DISCOVERY_TERMS, key=len, reverse=True):
                        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", claim):
                            candidates.append(term)
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
        if not evidence:
            errors.append(f"missing_evidence:{claim}")
        by_claim[claim] = {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "audible_alternative": alternative,
        }
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
        "an audible_alternative when the exact name is unsupported. Preserve exact specific names "
        "when heard. Return JSON only with exactly this shape: "
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
                    prompt += "\nPrevious response failed exact validation: " + last_error + ". Return corrected JSON only."
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

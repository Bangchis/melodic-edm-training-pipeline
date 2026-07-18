#!/usr/bin/env python3
"""Shared deterministic helpers for the second training release."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable


CAPTION_TYPES = ("canonical", "composition", "production")
CAPTION_COMPILER_REVISION = "openrouter-per-track-salient-audio-fusion-v3.1"
STRUCTURE_NORMALIZATION_REVISION = "sequence-safe-instrumental-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from *path*."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically so interrupted jobs keep the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a sequence of dictionaries as JSONL atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    """Count whitespace-separated words after normalizing spaces."""
    return len(re.findall(r"\S+", " ".join(str(text).split())))


def clean_caption(text: str) -> str:
    """Normalize a generated caption without changing its meaning."""
    value = re.sub(r"\s+", " ", str(text)).strip().strip('"`')
    return value[0].upper() + value[1:] if value else value


def normalize_instrumental_structure(lyrics: str) -> tuple[str, list[dict[str, Any]]]:
    """Repair only positionally impossible instrumental section labels.

    All-In-One can label a recurring low-density passage as ``intro`` or
    ``outro`` based on local acoustics even when it occurs in the middle of a
    track.  ACE-Step treats lyrics tags as a temporal script, so retaining those
    labels teaches contradictory ordering.  This function deliberately leaves
    every ordinary Theme/Drop/Break/Final Drop label unchanged and applies only
    four unambiguous sequence normalizations.
    """
    lines = str(lyrics).splitlines()
    section_lines: list[tuple[int, str]] = []
    for line_index, line in enumerate(lines):
        match = re.fullmatch(r"\s*\[([^\]]+)\]\s*", line)
        if not match:
            continue
        label = match.group(1).strip()
        if label.casefold() == "instrumental":
            continue
        section_lines.append((line_index, label))

    changes: list[dict[str, Any]] = []
    last_index = len(section_lines) - 1
    for section_index, (line_index, label) in enumerate(section_lines):
        normalized = label
        folded = label.casefold()
        reason = ""
        if folded == "break" and section_index == 0:
            normalized, reason = "Intro", "opening_break_to_intro"
        elif folded == "intro" and section_index > 0:
            normalized, reason = "Build", "mid_track_intro_to_build"
        elif folded == "outro" and section_index < last_index:
            normalized, reason = "Break", "mid_track_outro_to_break"
        elif folded == "break" and section_index == last_index:
            normalized, reason = "Outro", "closing_break_to_outro"
        if normalized != label:
            lines[line_index] = f"[{normalized}]"
            changes.append({
                "section_index": section_index,
                "from": label,
                "to": normalized,
                "reason": reason,
            })
    suffix = "\n" if str(lyrics).endswith("\n") else ""
    return "\n".join(lines) + suffix, changes


def parent_song_id(row: dict[str, Any]) -> str:
    """Return a stable parent ID shared by duplicate catalog records.

    Each accepted catalog row represents one complete source song. Sections,
    stems, captions and other derived artifacts inherit this identifier. Exact
    duplicate catalog rows share a YouTube video ID and therefore cannot cross
    the train/validation boundary.
    """
    video_id = str(row.get("video_id") or "").strip()
    if video_id:
        return f"youtube:{video_id}"
    identity = "|".join(
        str(row.get(field) or "").strip().casefold()
        for field in ("expected_artist", "expected_title", "expected_version")
    )
    return "identity:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def grouped_split(
    rows: list[dict[str, Any]], validation_count: int, seed: int
) -> dict[str, str]:
    """Choose an exact grouped split with no parent song crossing boundaries."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        parent = parent_song_id(row)
        groups.setdefault(parent, []).append(str(row["sample_id"]))

    ordered = sorted(groups.items())
    random.Random(seed).shuffle(ordered)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, sample_ids) in enumerate(ordered):
        size = len(sample_ids)
        for total, selected in sorted(list(reachable.items()), reverse=True):
            candidate = total + size
            if candidate <= validation_count and candidate not in reachable:
                reachable[candidate] = selected + (index,)
    if validation_count not in reachable:
        sizes = sorted(len(sample_ids) for _, sample_ids in ordered)
        raise ValueError(
            f"cannot form exact validation size {validation_count} from parent groups {sizes}"
        )

    validation_groups = {ordered[index][0] for index in reachable[validation_count]}
    return {
        str(row["sample_id"]): (
            "validation" if parent_song_id(row) in validation_groups else "train"
        )
        for row in rows
    }


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first decodable JSON object from model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "```").replace("```JSON", "```")
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response contains no decodable JSON object")


def caption_map(value: Any) -> dict[str, str]:
    """Normalize supported caption containers into the three required types."""
    if isinstance(value, dict):
        return {
            name: clean_caption(value.get(name, ""))
            for name in CAPTION_TYPES
        }
    if isinstance(value, list):
        output: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("type") or "").strip().lower()
            if name in CAPTION_TYPES:
                output[name] = clean_caption(item.get("text", ""))
        return {name: output.get(name, "") for name in CAPTION_TYPES}
    return {name: "" for name in CAPTION_TYPES}


def validate_caption_set(
    captions: dict[str, str], artist: str = "", title: str = ""
) -> list[str]:
    """Validate exact caption coverage, length and identity-name leakage."""
    errors: list[str] = []
    if set(captions) != set(CAPTION_TYPES):
        errors.append("caption_types_must_be_exactly_canonical_composition_production")
    for name in CAPTION_TYPES:
        text = clean_caption(captions.get(name, ""))
        words = word_count(text)
        lower = text.casefold()
        minimum = 40 if name == "canonical" else 25
        if not minimum <= words <= 80:
            errors.append(f"{name}_word_count_{words}_outside_{minimum}_80")
        for label, identity in (("artist", artist), ("title", title)):
            normalized = " ".join(str(identity).casefold().split())
            if len(normalized) >= 4 and normalized in lower:
                errors.append(f"{name}_contains_{label}_identity")
    if len({clean_caption(value).casefold() for value in captions.values()}) != 3:
        errors.append("captions_are_not_distinct")
    return errors

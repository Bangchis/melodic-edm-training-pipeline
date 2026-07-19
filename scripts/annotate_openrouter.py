#!/usr/bin/env python3
"""Create detailed, resumable audio annotations with strict OpenRouter JSON."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
HYPE_PHRASES = (
    "masterpiece", "best song ever", "professional quality", "extremely beautiful",
    "exactly like", "in the style of", "style of", "polished", "suitable for",
    "ideal for", "perfect for", "well-produced", "well produced", "high-quality",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise ValueError("unsupported message content")


def parse_json_content(content: Any) -> dict[str, Any]:
    """Parse strict JSON, tolerating a provider-added Markdown code fence."""
    text = content_text(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        candidates = []
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
        if not candidates:
            raise original_error
        def annotation_score(candidate: dict[str, Any]) -> int:
            return sum((
                isinstance(candidate.get("primary_genre"), str),
                isinstance(candidate.get("canonical_caption"), str),
                isinstance(candidate.get("caption_variants"), list),
                isinstance(candidate.get("main_instruments"), list),
            ))

        value = max(candidates, key=annotation_score)
    if not isinstance(value, dict):
        raise ValueError("annotation JSON root must be an object")
    return value


def ensure_preview(audio: Path, preview: Path, start: float, end: float) -> None:
    if preview.is_file():
        return
    preview.parent.mkdir(parents=True, exist_ok=True)
    tmp = preview.with_suffix(".tmp.mp3")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}",
            "-i", str(audio), "-t", f"{end - start:.3f}",
            "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(tmp),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("preview_failed:" + result.stderr[-500:].replace("\n", " "))
    os.replace(tmp, preview)


def artist_names(value: str) -> list[str]:
    parts = re.split(r"\b(?:feat\.?|ft\.?|and)\b|[&,/+|]", value, flags=re.IGNORECASE)
    return [re.sub(r"\s+", " ", part).strip().lower() for part in parts if len(part.strip()) >= 4]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_caption_text(text: str) -> str:
    """Remove vague quality/use-case phrases while preserving musical content."""
    value = str(text)
    value = re.sub(
        r",?\s+(?:making|rendering)\s+it\s+(?:ideal|suitable|perfect)\s+for[^.]*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\s+(?:ideal|suitable|perfect)\s+for[^.]*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bpolished\s+and\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+and\s+polished\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bpolished,\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bpolished\b", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b(?:a|an)\s+well[- ]produced\b",
        lambda match: "An" if match.group(0)[0].isupper() else "an",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bwell[- ]produced\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b[A-G](?:[#♯b♭])?\s+(?P<mode>major|minor)(?:\s+(?:key|scale|tonality))?\b",
        lambda match: f"{match.group('mode').lower()} tonality",
        value,
        flags=re.IGNORECASE,
    )
    # Normalize text produced by the earlier sanitizer revision.
    value = re.sub(r"\bthe\s+a\s+tonal center\b", "the tonal center", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(major|minor)\s+tonality\s+tonality\b", r"\1 tonality", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r",\s*,", ",", value)
    return value.strip()


def sanitize_annotation(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove low-confidence instrument guesses without rejecting good captions."""
    instruments = result.get("main_instruments", [])
    kept = []
    removed = []
    instrument_aliases = {"bass": "sub_bass", "drums": "electronic_drums"}
    role_aliases = {
        "harmony": "chordal_texture", "lead": "main_melody", "melody": "main_melody",
        "pad": "atmosphere", "percussion": "drums", "rhythm": "rhythmic_texture",
        "rhythmic_support": "rhythmic_texture", "unknown": "atmosphere",
    }
    for instrument in instruments if isinstance(instruments, list) else []:
        if not isinstance(instrument, dict):
            removed.append({"action": "removed_malformed_instrument"})
            continue
        original_name = str(instrument.get("name", ""))
        if original_name in instrument_aliases:
            instrument["name"] = instrument_aliases[original_name]
            removed.append({
                "action": "normalized_instrument_name",
                "from": original_name,
                "to": instrument["name"],
            })
        original_role = str(instrument.get("role", ""))
        if original_role in role_aliases:
            instrument["role"] = role_aliases[original_role]
            removed.append({
                "action": "normalized_instrument_role",
                "from": original_role,
                "to": instrument["role"],
                "instrument": instrument.get("name"),
            })
        if instrument.get("name") != "unknown" and safe_float(instrument.get("confidence"), -1.0) < 0.55:
            removed.append({
                "action": "removed_low_confidence_instrument",
                "name": instrument.get("name"),
                "confidence": instrument.get("confidence"),
            })
        else:
            kept.append(instrument)
    if isinstance(instruments, list):
        result["main_instruments"] = kept
    fields = [(result, "canonical_caption", "canonical_caption")]
    variants = result.get("caption_variants", [])
    sections = result.get("section_captions", [])
    fields.extend(
        (variant, "text", f"caption_variant:{variant.get('type', '')}")
        for variant in variants if isinstance(variant, dict)
    )
    fields.extend(
        (section, "caption", f"section_caption:{section.get('label', '')}")
        for section in sections if isinstance(section, dict)
    )
    for owner, key, field in fields:
        if key not in owner:
            continue
        original = str(owner[key])
        cleaned = sanitize_caption_text(original)
        if cleaned != original:
            owner[key] = cleaned
            removed.append({"action": "sanitized_non_audio_caption_phrase", "field": field})
    return removed


def validate_annotation(
    result: dict[str, Any], row: dict[str, Any], taxonomy: dict[str, Any], mir: dict[str, Any]
) -> list[str]:
    errors = []
    if result.get("primary_genre") not in taxonomy["primary_genres"]:
        errors.append("invalid_primary_genre")
    for field in ("secondary_genres", "style_families", "moods"):
        allowed = taxonomy["primary_genres"] if field == "secondary_genres" else taxonomy[field]
        values = result.get(field, [])
        if not isinstance(values, list) or any(value not in allowed for value in values):
            errors.append(f"invalid_{field}")
    instruments = result.get("main_instruments", [])
    if not isinstance(instruments, list) or not instruments:
        errors.append("invalid_main_instruments")
        instruments = []
    for instrument in instruments:
        if not isinstance(instrument, dict):
            errors.append("invalid_main_instruments")
            continue
        if instrument.get("name") not in taxonomy["instruments"]:
            errors.append("invalid_instrument")
        if instrument.get("role") not in taxonomy["instrument_roles"]:
            errors.append("invalid_instrument_role")
        if instrument.get("name") != "unknown" and safe_float(instrument.get("confidence"), -1.0) < 0.55:
            errors.append("low_confidence_instrument")

    canonical_value = result.get("canonical_caption", "")
    if not isinstance(canonical_value, str):
        errors.append("invalid_canonical_caption_type")
        canonical_value = ""
    canonical = canonical_value.strip()
    count = word_count(canonical)
    if not 40 <= count <= 80:
        errors.append(f"canonical_caption_word_count:{count}")
    variants = result.get("caption_variants", [])
    if not isinstance(variants, list):
        errors.append("caption_variant_types")
        variants = []
    if any(not isinstance(variant, dict) for variant in variants):
        errors.append("caption_variant_types")
        variants = [variant for variant in variants if isinstance(variant, dict)]
    types = [variant.get("type") for variant in variants]
    if sorted(types) != ["composition", "full", "production", "tags"]:
        errors.append("caption_variant_types")
    full_variant = next((variant.get("text", "").strip() for variant in variants if variant.get("type") == "full"), "")
    if full_variant != canonical:
        errors.append("full_variant_must_equal_canonical")
    if any(not str(variant.get("text", "")).strip() for variant in variants):
        errors.append("empty_caption_variant")
    variant_texts = [str(variant.get("text", "")).strip().lower() for variant in variants]
    if len(set(variant_texts)) != len(variant_texts):
        errors.append("caption_variants_must_differ")
    section_captions = result.get("section_captions", [])
    if not isinstance(section_captions, list):
        errors.append("section_captions_missing_or_duplicate_mir_labels")
        section_captions = []
    if any(not isinstance(item, dict) for item in section_captions):
        errors.append("section_captions_missing_or_duplicate_mir_labels")
        section_captions = [item for item in section_captions if isinstance(item, dict)]
    section_labels = [str(item.get("label", "")) for item in section_captions]
    supported_labels = {
        str(section.get("label", "")) for section in mir.get("sections", []) if section.get("label")
    }
    if not supported_labels.issubset(set(section_labels)) or len(section_labels) != len(set(section_labels)):
        errors.append("section_captions_missing_or_duplicate_mir_labels")
    section_texts = [str(item.get("caption", "")).strip() for item in section_captions]
    if any(not text for text in section_texts):
        errors.append("empty_section_caption")
    for item, text in zip(section_captions, section_texts):
        count = word_count(text)
        if text and not 5 <= count <= 40:
            errors.append(f"section_captions_word_count:{item.get('label', '')}:{count}")
    if len({text.lower() for text in section_texts}) != len(section_texts):
        errors.append("section_captions_must_differ")
    semantic_cores = []
    boilerplate = {
        "the", "section", "intro", "outro", "main", "theme", "build", "drop", "break",
        "final", "features",
    }
    for text in section_texts:
        words = [
            word.lower() for word in re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
            if word.lower() not in boilerplate
        ]
        semantic_cores.append(" ".join(words))
    nonempty_cores = [core for core in semantic_cores if core]
    if len(set(nonempty_cores)) != len(nonempty_cores):
        errors.append("section_captions_semantic_duplicates")
    texts = [canonical] + [str(variant.get("text", "")) for variant in variants] + section_texts
    combined = "\n".join(texts).lower()
    if any(phrase in combined for phrase in HYPE_PHRASES):
        errors.append("banned_hype_or_style_phrase")
    if re.search(r"\b\d{2,3}\s*bpm\b", combined):
        errors.append("bpm_leaked_into_caption")
    for artist in artist_names(str(row.get("expected_artist", ""))):
        if artist in combined:
            errors.append("artist_name_in_caption")
            break
    if re.search(
        r"\b[A-G](?:[#♯b♭])?\s+(?:major|minor)(?:\s+(?:key|scale|tonality))?\b",
        combined,
        flags=re.IGNORECASE,
    ):
        errors.append("keyscale_in_caption")
    if not canonical.lower().startswith("instrumental"):
        errors.append("canonical_caption_not_instrumental")
    confidence = safe_float(result.get("annotation_confidence"), -1.0)
    if not 0 <= confidence <= 1:
        errors.append("invalid_annotation_confidence")
    return sorted(set(errors))


def request_annotation(
    row: dict[str, Any], mir: dict[str, Any], preview: Path, taxonomy: dict[str, Any],
    schema: dict[str, Any], api_key: str, model: str, effort: str, timeout: int,
    structured_output: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_mir = {
        "bpm": mir.get("bpm"),
        "keyscale": mir.get("keyscale"),
        "key_confidence": mir.get("key_confidence"),
        "timesignature": mir.get("timesignature"),
        "sections": mir.get("sections", []),
        "required_section_caption_labels": sorted({
            str(section.get("label")) for section in mir.get("sections", []) if section.get("label")
        }),
    }
    metadata = {
        "title_for_identity_only": row.get("expected_title", ""),
        "artist_for_identity_only_do_not_copy_to_captions": row.get("expected_artist", ""),
        "version": row.get("expected_version", ""),
        "audio_source": row.get("audio_source", ""),
        "vocal_status": row.get("vocal_status_after_processing", "instrumental"),
        "annotation_window_seconds": [
            row.get("annotation_window_start", 0),
            row.get("annotation_window_end", row.get("duration")),
        ],
    }
    prompt = (
        "Listen carefully to the complete final training audio and return the strict JSON master annotation. "
        "Describe only audible, stable musical evidence. Treat the track as instrumental: permitted vocal chops are "
        "production texture, never lyrics. Use only taxonomy values for categorical fields. Do not guess a traditional "
        "instrument when uncertain; use unknown. The canonical caption must be English, start with 'Instrumental', contain "
        "40-80 words, and keep only the most important genre, mood, melody, arrangement, instrumentation and production "
        "facts. Never put artist/channel/title names, BPM, key, time signature, hype, quality claims, or 'in the style of' "
        "language in any caption. Return exactly four caption variants with unique types: full, composition, production, "
        "and tags. The full variant text must exactly equal canonical_caption; composition and production should emphasize their own "
        "audible aspects; tags should be a concise comma-separated prompt. Return one distinct section caption for every "
        "label in required_section_caption_labels. An extra taxonomy label is allowed only when it is clearly audible in the audio. "
        "Keep BPM/key/time signature as metadata, not caption text. "
        "Do not include use cases, audiences, content/media suitability, or vague quality words such as polished. "
        "Use 'unclear' for detailed free-text attributes that cannot be heard confidently.\n"
        "Metadata: " + json.dumps(metadata, ensure_ascii=False) + "\n"
        "MIR: " + json.dumps(compact_mir, ensure_ascii=False) + "\n"
        "Taxonomy: " + json.dumps(taxonomy, ensure_ascii=False)
    )
    audio_b64 = base64.b64encode(preview.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 3000,
        "reasoning": {"effort": effort, "exclude": True},
    }
    if structured_output:
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        payload["provider"] = {"require_parameters": True}
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "melodic-edm-training-pipeline/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    if not body.get("choices"):
        raise ValueError("OpenRouter response has no choices")
    result = parse_json_content(body["choices"][0]["message"]["content"])
    return result, body.get("usage") or {}


def write_manual_review(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "record_key", "expected_artist", "expected_title", "review_reason"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    parser.add_argument("--env-file", default="/workspace/.env")
    parser.add_argument("--taxonomy", default="configs/taxonomy.json")
    parser.add_argument("--schema", default="configs/annotation_schema.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--unstructured-json", action="store_true",
        help="Rely on the JSON-only prompt when the provider does not support response_format.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    load_env_file(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    taxonomy = json.loads((root / args.taxonomy).read_text(encoding="utf-8"))
    schema = json.loads((root / args.schema).read_text(encoding="utf-8"))
    source = [r for r in read_jsonl(root / args.manifest) if r.get("quality_status") == "accepted"]
    state_path = root / "data" / "annotation_manifest.jsonl"
    existing = read_jsonl(state_path)
    by_id = {r["sample_id"]: r for r in existing}
    reconciled = False
    for row in source:
        sid = row["sample_id"]
        record = by_id.get(sid)
        if not record or not isinstance(record.get("annotation"), dict):
            continue
        mir_path = root / "data" / "mir" / f"{sid}.json"
        if not mir_path.is_file():
            continue
        mir = json.loads(mir_path.read_text(encoding="utf-8"))
        actions = sanitize_annotation(record["annotation"])
        validation_errors = validate_annotation(record["annotation"], row, taxonomy, mir)
        accepted = bool(
            safe_float(record["annotation"].get("annotation_confidence")) >= 0.70
            and not validation_errors
        )
        if actions:
            record["annotation_sanitization"] = (record.get("annotation_sanitization") or []) + actions
        record["annotation_status"] = "accepted" if accepted else "manual_review"
        record["annotation_error"] = None if accepted else "revalidation:" + ",".join(validation_errors)
        if accepted:
            atomic_json(root / "data" / "annotations" / f"{sid}.json", record)
        else:
            (root / "data" / "annotations" / f"{sid}.json").unlink(missing_ok=True)
        by_id[sid] = record
        reconciled = reconciled or bool(actions) or not accepted
    if reconciled:
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda item: item["sample_id"]))
    pending = [r for r in source if by_id.get(r["sample_id"], {}).get("annotation_status") != "accepted"]
    if args.limit is not None:
        pending = pending[:max(0, args.limit)]

    for index, row in enumerate(pending, 1):
        sid = row["sample_id"]
        audio = Path(row["training_audio_path"])
        mir_path = root / "data" / "mir" / f"{sid}.json"
        last_error = ""
        result: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        validation_errors: list[str] = []
        effort_used = "minimal"
        sanitization: list[dict[str, Any]] = []
        annotation_start = 0.0
        annotation_end = float(row.get("duration") or 0)
        try:
            if not audio.is_file():
                raise FileNotFoundError(audio)
            if not mir_path.is_file():
                raise FileNotFoundError(mir_path)
            mir = json.loads(mir_path.read_text(encoding="utf-8"))
            from build_acestep_dataset import choose_window

            annotation_start, annotation_end = choose_window(float(row["duration"]), mir, 240.0)
            cached = by_id.get(sid, {})
            if isinstance(cached.get("annotation"), dict):
                result = cached["annotation"]
                sanitization = sanitize_annotation(result)
                validation_errors = validate_annotation(result, row, taxonomy, mir)
                if safe_float(result.get("annotation_confidence")) >= 0.70 and not validation_errors:
                    usage = cached.get("annotation_usage", {})
                    effort_used = cached.get("annotation_reasoning_effort", "cached_revalidation")
                    last_error = ""
                else:
                    result = None

            if result is None:
                preview = root / "data" / "training_preview" / f"{sid}_{int(annotation_start * 1000)}_{int(annotation_end * 1000)}.mp3"
                ensure_preview(audio, preview, annotation_start, annotation_end)
                request_row = {
                    **row,
                    "annotation_window_start": annotation_start,
                    "annotation_window_end": annotation_end,
                }
                for effort in ("minimal", "low"):
                    effort_used = effort
                    for attempt in range(1, args.retries + 1):
                        try:
                            result, usage = request_annotation(
                                request_row, mir, preview, taxonomy, schema, api_key, args.model, effort,
                                args.timeout, structured_output=not args.unstructured_json,
                            )
                            sanitization = sanitize_annotation(result)
                            validation_errors = validate_annotation(result, row, taxonomy, mir)
                            if safe_float(result.get("annotation_confidence")) >= 0.70 and not validation_errors:
                                last_error = ""
                                break
                            last_error = "validation:" + ",".join(validation_errors or ["low_confidence"])
                            # Temperature is zero, so repeating the same semantic
                            # request at the same reasoning level usually returns the
                            # same validation failure. Escalate directly to `low`;
                            # reserve retries for transport/provider exceptions.
                            break
                        except urllib.error.HTTPError as exc:
                            last_error = f"http_{exc.code}"
                            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                                break
                        except Exception as exc:
                            last_error = f"{type(exc).__name__}:{exc}"
                        if attempt < args.retries:
                            time.sleep(min(60.0, 2 ** attempt + random.random()))
                    if result is not None and safe_float(result.get("annotation_confidence")) >= 0.70 and not validation_errors:
                        break
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"

        accepted = bool(
            result is not None
            and safe_float(result.get("annotation_confidence")) >= 0.70
            and not validation_errors
        )
        record = {
            **row,
            "annotation_status": "accepted" if accepted else "manual_review",
            "annotation_model": args.model,
            "annotation_reasoning_effort": effort_used,
            "annotation_usage": usage,
            "annotation_sanitization": sanitization,
            "annotation_window_start": annotation_start,
            "annotation_window_end": annotation_end,
            "annotation_error": None if accepted else last_error,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        if result is not None:
            record["annotation"] = result
        by_id[sid] = record
        if accepted:
            atomic_json(root / "data" / "annotations" / f"{sid}.json", record)
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda item: item["sample_id"]))
        print(
            f"[{index}/{len(pending)}] {sid} {'PASS' if accepted else 'REVIEW'} "
            f"confidence={safe_float((result or {}).get('annotation_confidence')):.2f} {last_error}",
            flush=True,
        )

    final = [by_id[r["sample_id"]] for r in source if r["sample_id"] in by_id]
    review = [
        {
            **row,
            "review_reason": row.get("annotation_error", ""),
        }
        for row in final if row.get("annotation_status") != "accepted"
    ]
    write_manual_review(root / "data" / "manual_review.csv", review)
    accepted_count = sum(r.get("annotation_status") == "accepted" for r in final)
    print(json.dumps({"source": len(source), "accepted": accepted_count, "manual_review": len(review)}))
    return 0 if accepted_count == len(source) else 1


if __name__ == "__main__":
    raise SystemExit(main())

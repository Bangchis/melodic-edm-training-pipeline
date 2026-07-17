#!/usr/bin/env python3
"""Resume strict audio annotations with a pinned local Qwen2.5-Omni model."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_openrouter import (
    atomic_json,
    atomic_jsonl,
    ensure_preview,
    parse_json_content,
    read_jsonl,
    sanitize_annotation,
    sanitize_caption_text,
    safe_float,
    validate_annotation,
    word_count,
    write_manual_review,
)


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
CAPTION_COMPILER_VERSION = 9


def _sentence(value: str) -> str:
    text = " ".join(str(value).strip().split()).strip(" .")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def _without_explicit_key(value: str) -> str:
    import re

    cleaned = re.sub(
        r"\b[A-G](?:[#♯b♭])?\s+(?P<mode>major|minor)(?:\s+(?:key|scale))?\b",
        lambda match: f"{match.group('mode').lower()} tonality",
        str(value),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bthe\s+a\s+tonal center\b", "the tonal center", cleaned, flags=re.IGNORECASE)
    return sanitize_caption_text(cleaned)


SECTION_DISPLAY = {
    "Intro": "intro", "Theme": "main theme", "Build": "build", "Drop": "drop",
    "Break": "break", "Final Drop": "final drop", "Outro": "outro",
}
FINITE_SECTION_VERBS = {
    "adds", "begins", "build", "builds", "combines", "develops", "diminishes", "fades",
    "features", "introduce", "introduces", "intensifies", "leads", "maintains", "moves",
    "opens", "peaks", "presents", "provide", "provides", "reaches", "reduces", "returns",
    "shifts", "starts", "strips", "take", "takes", "transitions", "winds",
}


def _section_sentence(label: str, value: str) -> str:
    import re

    phrase = _without_explicit_key(value).strip(" .")
    section_name = SECTION_DISPLAY.get(label, label.lower())
    phrase = re.sub(r"^during\s+the\s+.+?\s+section,\s*", "", phrase, flags=re.IGNORECASE)
    lowered = phrase.lower()
    if lowered.startswith(f"the {section_name} section ") or lowered.startswith(f"the {section_name} "):
        return _sentence(phrase)
    phrase_words = phrase.split()
    first_word = phrase_words[0].lower() if phrase_words else "develops"
    later_finite_verb = any(
        word.lower().strip(",.;:") in FINITE_SECTION_VERBS for word in phrase_words[1:]
    )
    if first_word in FINITE_SECTION_VERBS:
        return _sentence(f"The {section_name} section {phrase[0].lower() + phrase[1:]}")
    if later_finite_verb:
        return _sentence(f"{phrase} in the {section_name} section")
    return _sentence(f"The {section_name} section features {phrase[0].lower() + phrase[1:]}")


def compile_canonical_caption(annotation: dict[str, Any]) -> str:
    """Compile a 40-80 word caption using only master-annotation evidence."""
    genre_key = str(annotation.get("primary_genre", "melodic_edm"))
    genre = {
        "melodic_edm": "melodic EDM",
        "progressive_house": "progressive house",
        "electro_house": "electro house",
        "glitch_hop": "glitch hop",
        "future_bass": "future bass",
        "drumstep": "drumstep",
        "cinematic_electronic": "cinematic electronic music",
    }.get(genre_key, genre_key.replace("_", " "))
    moods = [str(value).replace("_", " ") for value in annotation.get("moods", [])[:4]]
    mood_text = ", ".join(moods[:-1]) + (" and " + moods[-1] if len(moods) > 1 else (moods[0] if moods else ""))
    base = f"Instrumental {genre}" + (f" with an {mood_text} mood." if mood_text else ".")

    instruments = []
    role_phrases = {
        "main_melody": "leading the main melody",
        "main_hook": "carrying the main hook",
        "counter_melody": "adding a countermelody",
        "call_and_response": "providing call-and-response phrases",
        "melody_doubling": "doubling the melody",
        "chordal_texture": "shaping the chord texture",
        "rhythmic_texture": "adding rhythmic texture",
        "bass": "supporting the bass",
        "drums": "driving the rhythm",
        "accents": "adding accents",
        "atmosphere": "creating atmosphere",
    }
    for item in annotation.get("main_instruments", [])[:5]:
        name = str(item.get("name", "")).replace("_", " ")
        role_key = str(item.get("role", ""))
        role = role_phrases.get(role_key, role_key.replace("_", " "))
        if name and name != "unknown":
            instruments.append(f"{name} {role}" if role else name)
    candidates = []
    if instruments:
        candidates.append(_sentence("The instrumentation uses " + ", ".join(instruments)))
    melody = annotation.get("melody") or {}
    if melody.get("description"):
        candidates.append(_sentence(_without_explicit_key(str(melody["description"]))))
    sections = {item.get("label"): item.get("caption") for item in annotation.get("section_captions", [])}
    for label in ("Intro", "Drop", "Final Drop", "Build"):
        if sections.get(label):
            candidates.append(_section_sentence(label, str(sections[label])))
    production = annotation.get("production") or {}
    production_parts = []
    production_nouns = {
        "bass": ("bass",), "chords": ("chord", "chords"),
        "drums": ("drum", "drums", "percussion"),
        "space": ("space", "stereo", "reverb", "ambience"),
    }
    production_suffix = {"bass": "bass", "chords": "chords", "drums": "drums", "space": "stereo space"}
    for key in ("bass", "chords", "drums", "space"):
        if not production.get(key):
            continue
        phrase = _without_explicit_key(str(production[key])).strip(" .")
        if not any(noun in phrase.lower() for noun in production_nouns[key]):
            phrase += " " + production_suffix[key]
        production_parts.append(phrase)
    if production_parts:
        joined = ", ".join(production_parts[:-1])
        if len(production_parts) > 1:
            joined += ", and " + production_parts[-1]
        else:
            joined = production_parts[0]
        candidates.append(_sentence("The production uses " + joined))
    if production.get("description"):
        candidates.append(_sentence(_without_explicit_key(str(production["description"]))))

    result = base
    for candidate in candidates:
        if not candidate:
            continue
        proposed = result + " " + candidate
        if word_count(proposed) <= 65 or (word_count(result) < 50 and word_count(proposed) <= 80):
            result = proposed
        if word_count(result) >= 55:
            break
    # Continue filling to the hard minimum if earlier clauses were unusually short.
    if word_count(result) < 40:
        for candidate in candidates:
            if candidate and candidate not in result and word_count(result + " " + candidate) <= 80:
                result += " " + candidate
            if word_count(result) >= 40:
                break
    return result


def compile_section_captions(annotation: dict[str, Any], mir: dict[str, Any]) -> list[dict[str, str]]:
    """Create one grounded, distinct caption for every MIR-required section label."""
    mapping = {
        "Intro": "intro", "Theme": "theme", "Build": "buildup", "Drop": "drop",
        "Break": "break", "Final Drop": "final_drop", "Outro": "outro",
    }
    terse_defaults = {
        "Intro": "develops the opening texture", "Theme": "presents the main melodic motif",
        "Build": "increases tension toward the drop", "Drop": "intensifies the rhythmic and melodic drive",
        "Break": "reduces the arrangement density", "Final Drop": "returns at peak melodic intensity",
        "Outro": "winds down the arrangement",
    }
    required = []
    for section in mir.get("sections", []):
        label = str(section.get("label", ""))
        if label and label not in required:
            required.append(label)
    arrangement = annotation.get("arrangement") or {}
    instruments = [
        str(item.get("name", "")).replace("_", " ")
        for item in annotation.get("main_instruments", [])
        if item.get("name") and item.get("name") != "unknown"
    ][:3]
    if len(instruments) > 1:
        instrument_text = ", ".join(instruments[:-1]) + " and " + instruments[-1]
    elif instruments:
        instrument_text = instruments[0]
    else:
        instrument_text = "audible electronic layers"
    output = []
    for label in required:
        phrase = str(arrangement.get(mapping.get(label, ""), "develops")).strip()
        phrase_words = [word.lower().strip(",.;:") for word in phrase.split()]
        has_finite_verb = any(word in FINITE_SECTION_VERBS for word in phrase_words)
        if word_count(phrase) < 3 or (word_count(phrase) == 3 and not has_finite_verb):
            phrase = f"{terse_defaults.get(label, 'develops the section')} with {instrument_text}"
        caption = _section_sentence(label, phrase)
        output.append({"label": label, "caption": caption})
    return output


def build_prompt(
    row: dict[str, Any], mir: dict[str, Any], taxonomy: dict[str, Any], schema: dict[str, Any]
) -> str:
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
    return (
        "Listen carefully to the complete final training audio and return only one JSON object matching the schema below. "
        "Describe only audible, stable musical evidence. Treat the track as instrumental: permitted vocal chops are "
        "production texture, never lyrics. Use only taxonomy values for categorical fields. Do not guess a traditional "
        "instrument when uncertain; use unknown. The canonical caption must be English, start with 'Instrumental', contain "
        "50-65 words (count the words before returning), and keep the most important genre, mood, melody, arrangement, instrumentation and production "
        "facts. Never put artist/channel/title names, BPM, key, time signature, hype, quality claims, or 'in the style of' "
        "language in any caption. Return exactly four caption variants with unique types: full, composition, production, "
        "and tags. The full variant text must exactly equal canonical_caption. Return one distinct section caption of 6-30 "
        "words for every label in required_section_caption_labels. Extra section labels are allowed only when clearly audible. Do not include "
        "use cases, audiences, content/media suitability, or vague quality words such as polished. Use 'unclear' for detailed "
        "free-text attributes that cannot be heard confidently. Do not wrap the JSON in Markdown.\n"
        "Metadata: " + json.dumps(metadata, ensure_ascii=False) + "\n"
        "MIR: " + json.dumps(compact_mir, ensure_ascii=False) + "\n"
        "Taxonomy: " + json.dumps(taxonomy, ensure_ascii=False) + "\n"
        "Required JSON schema: " + json.dumps(schema.get("schema", schema), ensure_ascii=False)
    )


def load_model(model_path: Path):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    max_memory = {0: "20GiB", 1: "20GiB", "cpu": "40GiB"}
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto", max_memory=max_memory,
        attn_implementation="sdpa", local_files_only=True,
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), local_files_only=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, processor


def generate_annotation(model: Any, processor: Any, preview: Path, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch
    from qwen_omni_utils import process_mm_info

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": str(preview)},
            {"type": "text", "text": prompt},
        ],
    }]
    rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=rendered, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device).to(model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, return_audio=False, do_sample=False, max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )
    # Some Transformers versions return full input+output IDs; others return only
    # generated IDs. Decode both ways and prefer the suffix when it parses.
    candidates = []
    if hasattr(inputs, "input_ids") and output_ids.shape[-1] > inputs.input_ids.shape[-1]:
        candidates.append(processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0])
    candidates.append(processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0])
    last_error: Exception | None = None
    for value in candidates:
        try:
            parsed = parse_json_content(value)
            if not {"primary_genre", "canonical_caption"}.issubset(parsed):
                raise ValueError("parsed JSON is not a complete master annotation")
            return parsed
        except Exception as exc:
            last_error = exc
    raise ValueError(f"local model did not return parseable JSON: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    parser.add_argument("--taxonomy", default="configs/taxonomy.json")
    parser.add_argument("--schema", default="configs/annotation_schema.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2200)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    revision_path = model_path / "PINNED_REVISION"
    if not revision_path.is_file():
        raise SystemExit("local annotation model is not pinned/complete")
    model_revision = revision_path.read_text(encoding="utf-8").strip()
    taxonomy = json.loads((root / args.taxonomy).read_text(encoding="utf-8"))
    schema = json.loads((root / args.schema).read_text(encoding="utf-8"))
    source = [row for row in read_jsonl(root / args.manifest) if row.get("quality_status") == "accepted"]
    state_path = root / "data" / "annotation_manifest.jsonl"
    by_id = {row["sample_id"]: row for row in read_jsonl(state_path)}
    migrated_existing = False
    for row in source:
        sid = row["sample_id"]
        record = by_id.get(sid)
        if not record or record.get("annotation_status") != "accepted" or not record.get("annotation"):
            continue
        annotation = record["annotation"]
        actions = sanitize_annotation(annotation)
        mir = json.loads((root / "data" / "mir" / f"{sid}.json").read_text(encoding="utf-8"))
        errors = validate_annotation(annotation, row, taxonomy, mir)
        if any(
            error.startswith("section_captions_word_count:")
            or error == "section_captions_semantic_duplicates"
            for error in errors
        ):
            annotation["section_captions"] = compile_section_captions(annotation, mir)
            actions.append({
                "action": "compiled_section_captions_from_master_arrangement",
                "section_count": len(annotation["section_captions"]),
                "reason": "section_caption_too_terse_or_long",
            })
            record["annotation_caption_compiler_version"] = CAPTION_COMPILER_VERSION
            errors = validate_annotation(annotation, row, taxonomy, mir)
        if errors:
            raise RuntimeError(f"existing annotation migration failed for {sid}: {errors}")
        if not actions:
            continue
        record["annotation"] = annotation
        record["annotation_sanitization"] = (record.get("annotation_sanitization") or []) + actions
        by_id[sid] = record
        atomic_json(root / "data" / "annotations" / f"{sid}.json", record)
        migrated_existing = True
    if migrated_existing:
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda item: item["sample_id"]))
    reconciled = False
    for row in source:
        sid = row["sample_id"]
        record = by_id.get(sid)
        if (
            not record
            or record.get("annotation_status") != "accepted"
            or not str(record.get("annotation_model", "")).startswith(MODEL_ID + "@")
        ):
            continue
        actions = record.get("annotation_sanitization") or []
        was_compiled = any(action.get("action") in {
            "compiled_canonical_caption_from_master_fields",
            "compiled_section_captions_from_master_arrangement",
        } for action in actions)
        if not was_compiled or record.get("annotation_caption_compiler_version") == CAPTION_COMPILER_VERSION:
            continue
        annotation = record.get("annotation") or {}
        previous = str(annotation.get("canonical_caption", ""))
        reconciliation_sanitization = sanitize_annotation(annotation)
        if any(action.get("action") == "compiled_section_captions_from_master_arrangement" for action in actions):
            mir = json.loads((root / "data" / "mir" / f"{sid}.json").read_text(encoding="utf-8"))
            annotation["section_captions"] = compile_section_captions(annotation, mir)
            reconciliation_sanitization.append({
                "action": "recompiled_section_captions_from_master_arrangement",
                "section_count": len(annotation["section_captions"]),
            })
        compiled = compile_canonical_caption(annotation)
        annotation["canonical_caption"] = compiled
        for variant in annotation.get("caption_variants", []):
            if variant.get("type") == "full":
                variant["text"] = compiled
        reconciliation_sanitization += sanitize_annotation(annotation)
        mir = json.loads((root / "data" / "mir" / f"{sid}.json").read_text(encoding="utf-8"))
        errors = validate_annotation(annotation, row, taxonomy, mir)
        if errors:
            raise RuntimeError(f"caption compiler reconciliation failed for {sid}: {errors}")
        record["annotation"] = annotation
        record["annotation_caption_compiler_version"] = CAPTION_COMPILER_VERSION
        record["annotation_sanitization"] = actions + reconciliation_sanitization + [{
            "action": f"recompiled_canonical_caption_v{CAPTION_COMPILER_VERSION}",
            "previous_word_count": word_count(previous),
            "compiled_word_count": word_count(compiled),
        }]
        by_id[sid] = record
        atomic_json(root / "data" / "annotations" / f"{sid}.json", record)
        reconciled = True
    if reconciled:
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda item: item["sample_id"]))
    pending = [row for row in source if by_id.get(row["sample_id"], {}).get("annotation_status") != "accepted"]
    if args.limit is not None:
        pending = pending[:max(0, args.limit)]
    if not pending:
        print("No pending annotation records", flush=True)
        return 0

    model, processor = load_model(model_path)
    for index, row in enumerate(pending, 1):
        sid = row["sample_id"]
        mir_path = root / "data" / "mir" / f"{sid}.json"
        result = None
        errors: list[str] = []
        sanitization: list[dict[str, Any]] = []
        last_error = ""
        start, end = 0.0, float(row.get("duration") or 0)
        try:
            from build_acestep_dataset import choose_window

            mir = json.loads(mir_path.read_text(encoding="utf-8"))
            start, end = choose_window(float(row["duration"]), mir, 240.0)
            request_row = {**row, "annotation_window_start": start, "annotation_window_end": end}
            preview = root / "data" / "training_preview" / f"{sid}_{int(start * 1000)}_{int(end * 1000)}.mp3"
            ensure_preview(Path(row["training_audio_path"]), preview, start, end)
            base_prompt = build_prompt(request_row, mir, taxonomy, schema)
            correction = ""
            for attempt in range(2):
                try:
                    result = generate_annotation(
                        model, processor, preview, base_prompt + correction, args.max_new_tokens
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}:{exc}"
                    if attempt == 0:
                        correction = (
                            "\nYour previous response was not one complete JSON master annotation. Return exactly one "
                            "complete JSON object with every required schema field; do not emit separate objects, notes, "
                            "or Markdown."
                        )
                        continue
                    raise
                sanitization = sanitize_annotation(result)
                errors = validate_annotation(result, row, taxonomy, mir)
                if safe_float(result.get("annotation_confidence")) < 0.70:
                    errors.append("low_confidence")
                errors = sorted(set(errors))
                if any(
                    error.startswith("section_captions_") or error == "empty_section_caption"
                    for error in errors
                ):
                    result["section_captions"] = compile_section_captions(result, mir)
                    sanitization.append({
                        "action": "compiled_section_captions_from_master_arrangement",
                        "section_count": len(result["section_captions"]),
                    })
                    sanitization += sanitize_annotation(result)
                    errors = validate_annotation(result, row, taxonomy, mir)
                if errors and all(
                    error.startswith("canonical_caption_word_count:") or error == "keyscale_in_caption"
                    for error in errors
                ):
                    previous = str(result.get("canonical_caption", ""))
                    compiled = compile_canonical_caption(result)
                    result["canonical_caption"] = compiled
                    for variant in result.get("caption_variants", []):
                        if variant.get("type") == "full":
                            variant["text"] = compiled
                    sanitization.append({
                        "action": "compiled_canonical_caption_from_master_fields",
                        "previous_word_count": word_count(previous),
                        "compiled_word_count": word_count(compiled),
                    })
                    sanitization += sanitize_annotation(result)
                    errors = validate_annotation(result, row, taxonomy, mir)
                if not errors:
                    break
                last_error = "validation:" + ",".join(errors)
                correction = (
                    "\nYour previous JSON failed these deterministic validator checks: "
                    + json.dumps(errors, ensure_ascii=False)
                    + ". Return a corrected complete JSON object. In particular, canonical_caption and the identical full "
                    "variant must contain 50-65 English words while remaining grounded in the audio. Previous JSON: "
                    + json.dumps(result, ensure_ascii=False)
                )
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"

        accepted = result is not None and not errors and safe_float(result.get("annotation_confidence")) >= 0.70
        record = {
            **row,
            "annotation_status": "accepted" if accepted else "manual_review",
            "annotation_model": f"{MODEL_ID}@{model_revision}",
            "annotation_reasoning_effort": "local_deterministic",
            "annotation_usage": {"cost": 0, "provider": "local_vast"},
            "annotation_sanitization": sanitization,
            "annotation_window_start": start,
            "annotation_window_end": end,
            "annotation_error": None if accepted else last_error,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        if any(action.get("action") in {
            "compiled_canonical_caption_from_master_fields",
            "compiled_section_captions_from_master_arrangement",
        } for action in sanitization):
            record["annotation_caption_compiler_version"] = CAPTION_COMPILER_VERSION
        if result is not None:
            record["annotation"] = result
        by_id[sid] = record
        if accepted:
            atomic_json(root / "data" / "annotations" / f"{sid}.json", record)
        else:
            (root / "data" / "annotations" / f"{sid}.json").unlink(missing_ok=True)
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda item: item["sample_id"]))
        print(f"[{index}/{len(pending)}] {sid} {'PASS' if accepted else 'REVIEW'} {last_error}", flush=True)

    reviews = [
        {**row, "review_reason": row.get("annotation_error", "manual_review")}
        for row in by_id.values() if row.get("annotation_status") != "accepted"
    ]
    write_manual_review(root / "data" / "manual_review.csv", sorted(reviews, key=lambda item: item["sample_id"]))
    print(json.dumps({
        "source": len(source),
        "accepted": sum(row.get("annotation_status") == "accepted" for row in by_id.values()),
        "manual_review": len(reviews),
        "local_model_revision": model_revision,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

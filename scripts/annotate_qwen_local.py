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
    validate_annotation,
    write_manual_review,
)


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


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
        "40-80 words, and keep only the most important genre, mood, melody, arrangement, instrumentation and production "
        "facts. Never put artist/channel/title names, BPM, key, time signature, hype, quality claims, or 'in the style of' "
        "language in any caption. Return exactly four caption variants with unique types: full, composition, production, "
        "and tags. The full variant text must exactly equal canonical_caption. Return one distinct section caption for every "
        "label in required_section_caption_labels. Extra section labels are allowed only when clearly audible. Do not include "
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
            return parse_json_content(value)
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
            result = generate_annotation(model, processor, preview, build_prompt(request_row, mir, taxonomy, schema), args.max_new_tokens)
            sanitization = sanitize_annotation(result)
            errors = validate_annotation(result, row, taxonomy, mir)
            if float(result.get("annotation_confidence", 0)) < 0.70:
                errors.append("low_confidence")
            if errors:
                last_error = "validation:" + ",".join(sorted(set(errors)))
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"

        accepted = result is not None and not errors and float(result.get("annotation_confidence", 0)) >= 0.70
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

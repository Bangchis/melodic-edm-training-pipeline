#!/usr/bin/env python3
"""Classify vocals with OpenRouter while preserving every catalog record."""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def content_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return "".join(
            item.get("text", "") for item in message_content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise ValueError("unsupported message content shape")


def validate_result(result: dict[str, Any]) -> None:
    status = result.get("vocal_status")
    action = result.get("recommended_action")
    expected = {
        "instrumental": "keep_original",
        "vocal_chops": "keep_original",
        "lyrics": "separate_vocals",
        "uncertain": "manual_check",
    }
    if status not in expected:
        raise ValueError(f"invalid vocal_status: {status!r}")
    if action != expected[status]:
        raise ValueError(f"inconsistent action for {status}: {action!r}")
    prominence = float(result.get("lyrics_prominence"))
    if not 0.0 <= prominence <= 1.0:
        raise ValueError("lyrics_prominence outside 0..1")
    if not str(result.get("reason", "")).strip():
        raise ValueError("empty reason")


def classify(
    row: dict[str, Any], api_key: str, schema: dict[str, Any], model: str,
    reasoning_effort: str, timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preview = Path(row["preview_path"])
    if not preview.is_file():
        raise FileNotFoundError(preview)
    audio_b64 = base64.b64encode(preview.read_bytes()).decode("ascii")
    metadata = {
        "sample_id": row["sample_id"],
        "expected_title": row.get("expected_title", ""),
        "expected_artist": row.get("expected_artist", ""),
        "expected_version": row.get("expected_version", ""),
    }
    prompt = (
        "Listen to the complete track and classify only the presence of human vocals. "
        "Policy: no_lyrics_allow_vocal_chops. Sung or spoken intelligible lyrics count as lyrics, "
        "in any language. Short chopped, pitched, reversed, or syllabic vocal samples functioning "
        "as an instrument/effect count as vocal_chops and may be kept. If there is no human voice, "
        "return instrumental. If audio evidence is ambiguous, return uncertain. "
        "Actions are fixed: instrumental/vocal_chops=keep_original, lyrics=separate_vocals, "
        "uncertain=manual_check. Do not judge catalog identity. Metadata: "
        + json.dumps(metadata, ensure_ascii=False)
    )
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
        "max_tokens": 512,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        "response_format": {"type": "json_schema", "json_schema": schema},
        "provider": {"require_parameters": True},
    }
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
    result = json.loads(content_text(body["choices"][0]["message"]["content"]))
    validate_result(result)
    return result, body.get("usage") or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/audio_manifest.jsonl")
    parser.add_argument("--output", default="data/vocal_manifest.jsonl")
    parser.add_argument("--schema", default="configs/vocal_schema.json")
    parser.add_argument("--env-file", default="/workspace/.env")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low"), default="minimal")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    source = [r for r in read_jsonl(Path(args.manifest)) if r.get("audio_valid")]
    output_path = Path(args.output)
    existing = read_jsonl(output_path)
    completed = {r["sample_id"]: r for r in existing if r.get("classification_status") == "complete"}
    pending = [r for r in source if r["sample_id"] not in completed]
    if args.limit is not None:
        pending = pending[:max(0, args.limit)]

    ordered = list(existing)
    for index, row in enumerate(pending, 1):
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                result, usage = classify(
                    row, api_key, schema, args.model, args.reasoning_effort, args.timeout
                )
                record = {
                    **row,
                    **result,
                    "classification_status": "complete",
                    "classification_model": args.model,
                    "classification_reasoning_effort": args.reasoning_effort,
                    "classification_usage": usage,
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                ordered = [r for r in ordered if r.get("sample_id") != row["sample_id"]]
                ordered.append(record)
                atomic_jsonl(output_path, ordered)
                print(
                    f"[{index}/{len(pending)}] {row['sample_id']} "
                    f"{result['vocal_status']} -> {result['recommended_action']}",
                    flush=True,
                )
                break
            except urllib.error.HTTPError as exc:
                last_error = f"http_{exc.code}"
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"
            if attempt < args.retries:
                time.sleep(min(60.0, (2 ** attempt) + random.random()))
        else:
            attempt = args.retries
        if row["sample_id"] not in {r.get("sample_id") for r in ordered if r.get("classification_status") == "complete"}:
            failure = {
                **row,
                "classification_status": "failed",
                "classification_model": args.model,
                "classification_error": last_error,
                "classification_attempts": attempt,
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            ordered = [r for r in ordered if r.get("sample_id") != row["sample_id"]]
            ordered.append(failure)
            atomic_jsonl(output_path, ordered)
            print(f"[{index}/{len(pending)}] {row['sample_id']} FAILED {last_error}", flush=True)

    final = read_jsonl(output_path)
    complete = sum(r.get("classification_status") == "complete" for r in final)
    failed = sum(r.get("classification_status") == "failed" for r in final)
    print(json.dumps({"source": len(source), "complete": complete, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

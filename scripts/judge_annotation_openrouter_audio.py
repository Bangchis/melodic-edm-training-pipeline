#!/usr/bin/env python3
"""Use a second audio-capable model to adjudicate disputed caption claims."""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from v2_common import atomic_json, extract_json_object


API_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_mp3(audio: Path) -> str:
    """Encode complete audio to a compact stereo MP3 accepted by OpenRouter audio input."""
    with tempfile.TemporaryDirectory(prefix="v2-audio-judge-") as temporary:
        target = Path(temporary) / "audio.mp3"
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
                "-vn", "-ar", "44100", "-ac", "2", "-b:a", "96k", str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if completed.returncode or not target.is_file():
            raise RuntimeError(f"ffmpeg audio conversion failed: {completed.stderr[-1000:]}")
        return base64.b64encode(target.read_bytes()).decode("ascii")


def request_payload(model: str, audio_b64: str, captions: dict[str, str]) -> dict[str, Any]:
    """Build a strict independent caption audit request."""
    prompt = (
        "Listen to the entire instrumental track and independently audit these proposed training "
        "captions. Treat every caption claim as untrusted. Do not infer title, artist, country or "
        "metadata. In particular, only accept named instruments such as pipa, guzheng, dizi, erhu, "
        "strings, brass, piano or guitar when their timbre is unmistakably audible; otherwise list "
        "the claim as unsupported. Judge audible fidelity, specificity, melody/arrangement accuracy "
        "and production accuracy from 1 to 5. Return JSON only with those four integer scores, short "
        "evidence strings, unsupported_claims as an array, and recommendation keep or revise. "
        "Required shape: {\"audible_fidelity\":1,\"specificity\":1,"
        "\"melody_arrangement_accuracy\":1,\"production_accuracy\":1,\"evidence\":{"
        "\"audible_fidelity\":\"...\",\"specificity\":\"...\","
        "\"melody_arrangement_accuracy\":\"...\",\"production_accuracy\":\"...\"},"
        "\"unsupported_claims\":[],\"recommendation\":\"revise\"}.\nCaptions: "
        + json.dumps(captions, ensure_ascii=False)
    )
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model", default="openai/gpt-audio-mini")
    args = parser.parse_args()
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    annotation = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    captions = {
        str(item["type"]): str(item["text"])
        for item in annotation["caption_variants"]
    }
    payload = request_payload(args.model, encode_mp3(Path(args.audio)), captions)
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Bangchis/melodic-edm-training-pipeline",
            "X-Title": "Melodic EDM caption audit",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"OpenRouter audio audit failed HTTP {exc.code}: {detail}") from exc
    content = raw["choices"][0]["message"]["content"]
    result = {
        "status": "pass",
        "sample_id": args.sample_id,
        "judge_model": args.model,
        "audit": extract_json_object(content),
        "usage": raw.get("usage", {}),
    }
    atomic_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

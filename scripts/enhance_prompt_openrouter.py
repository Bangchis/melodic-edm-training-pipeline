#!/usr/bin/env python3
"""Turn a free-form music idea into validated ACE-Step conditions via OpenRouter."""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

from prompt_enhancer import compile_caption


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "~google/gemini-flash-latest"
DEFAULT_SECTIONS = ["Intro", "Theme", "Build", "Drop", "Break", "Final Drop", "Outro"]
MUSIC_FIELDS = ("genre", "mood", "melody", "arrangement", "production")

OUTPUT_SCHEMA = {
    "name": "ace_step_music_conditions",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["genre", "mood", "melody", "arrangement", "production"],
        "properties": {
            "genre": {
                "type": "string",
                "description": "Two to twenty words describing audible genre and style families.",
            },
            "mood": {
                "type": "string",
                "description": "Two to twenty audible mood words without quality hype.",
            },
            "melody": {
                "type": "string",
                "description": "Detailed audible description of motif, contour, repetition, register, harmony and melodic instruments.",
            },
            "arrangement": {
                "type": "string",
                "description": "Detailed audible description of how sections develop, transition, contrast and return.",
            },
            "production": {
                "type": "string",
                "description": "Detailed audible description of instruments, synths, bass, drums, texture, dynamics and space.",
            },
        },
    },
}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise ValueError("OpenRouter returned unsupported message content")


def _parse_json_content(content: Any) -> dict[str, Any]:
    text = _content_text(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("OpenRouter structured response must be an object")
    return value


def sections_to_lyrics(sections: list[str]) -> str:
    """Compile user-defined safe section labels into instrumental ACE text."""
    if not sections:
        raise ValueError("sections must be a non-empty list")
    normalized: list[str] = []
    for raw_section in sections:
        section = " ".join(str(raw_section).split())
        if not section:
            raise ValueError("section labels cannot be empty")
        if len(section) > 80:
            raise ValueError(f"section label is longer than 80 characters: {section!r}")
        if "[" in section or "]" in section:
            raise ValueError(f"section labels cannot contain '[' or ']': {section!r}")
        if any(ord(character) < 32 for character in section):
            raise ValueError(f"section labels cannot contain control characters: {section!r}")
        normalized.append(section)
    folded = [section.casefold() for section in normalized]
    if len(folded) != len(set(folded)):
        raise ValueError("sections must not contain duplicate labels")
    return "\n\n".join(f"[{section}]\n[Instrumental]" for section in normalized) + "\n"


def contains_required_term(caption: str, term: str) -> bool:
    """Require the exact phrase, not a weakened ``term-like`` substitution."""
    normalized = " ".join(term.casefold().split())
    phrase = re.escape(normalized).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![\w]){phrase}(?![\w-])", caption.casefold()))


def validate_conditions(
    value: dict[str, Any],
    explicit_conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provider output without weakening user-authoritative prompt terms."""
    if set(value) != set(MUSIC_FIELDS):
        missing = sorted(set(MUSIC_FIELDS) - set(value))
        extra = sorted(set(value) - set(MUSIC_FIELDS))
        raise ValueError(f"expected exactly five music fields; missing={missing}, extra={extra}")
    conditions = {field: str(value.get(field, "")).strip() for field in MUSIC_FIELDS}
    caption = compile_caption(conditions, min_words=40, max_words=300)
    explicit = dict(explicit_conditions or {})
    required_terms = explicit.pop("required_terms", [])
    if isinstance(required_terms, str) or not isinstance(required_terms, list):
        raise ValueError("required_terms must be a list of exact user-authoritative phrases")
    required_terms = [str(term).strip() for term in required_terms if str(term).strip()]
    missing_terms = [
        term for term in required_terms
        if not contains_required_term(caption, term)
    ]
    if missing_terms:
        raise ValueError(
            "enhanced caption weakened or omitted required user terms: "
            + ", ".join(missing_terms)
        )
    metadata = {
        "bpm": 128,
        "keyscale": "C minor",
        "timesignature": "4",
        "sections": list(DEFAULT_SECTIONS),
    }
    metadata.update(explicit)
    bpm = int(metadata["bpm"])
    if not 60 <= bpm <= 200:
        raise ValueError(f"bpm must be in 60..200, got {bpm}")
    keyscale = str(metadata["keyscale"]).strip()
    if not re.fullmatch(r"[A-G](?:#|b)?\s+(?:major|minor)", keyscale):
        raise ValueError(f"keyscale must look like 'F# minor', got {keyscale!r}")
    timesignature = str(metadata["timesignature"]).strip()
    if timesignature not in {"2", "3", "4", "5", "6", "7"}:
        raise ValueError(f"unsupported time signature: {timesignature!r}")
    sections = metadata["sections"]
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")
    lyrics = sections_to_lyrics([str(section) for section in sections])
    return {
        **conditions,
        "caption": caption,
        "word_count": len(caption.split()),
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": timesignature,
        "sections": sections,
        "lyrics": lyrics,
        "required_terms": required_terms,
    }


def _request_once(
    *,
    idea: str,
    api_key: str,
    model: str,
    explicit_conditions: dict[str, Any],
    correction: str,
    timeout: int,
    opener: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = (
        "You are a music prompt enhancer for an instrumental ACE-Step model. Convert the user's idea "
        "into concrete audible musical attributes. The user's idea is authoritative: preserve every "
        "explicit condition, exact named instrument, genre, mood, arrangement request, production "
        "request and negative condition. Never weaken or generalize a named instrument; for example, "
        "do not replace pipa with plucked-string-like or dizi with flute-like. Add detail only when it "
        "is compatible with the request. This is inference conditioning, not uncertain audio annotation. Do not use "
        "artist names, 'in the style of', quality hype, use cases, or claims that cannot be heard. Treat BPM, "
        "key, time signature and sections as external fixed conditions; do not repeat them as JSON fields. "
        "The five prose fields "
        "must compile into one detailed 40-300 word English caption, with 300 words as a hard maximum. "
        "Return exactly the five requested prose fields and no metadata fields."
    )
    user = {
        "idea": idea,
        "explicit_conditions_that_must_be_preserved": explicit_conditions,
        "validator_feedback_from_previous_attempt": correction or None,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 2400,
        "reasoning": {"effort": "minimal", "exclude": True},
        "response_format": {"type": "json_schema", "json_schema": OUTPUT_SCHEMA},
        "provider": {"require_parameters": True},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "melodic-edm-training-pipeline/2.0",
            "X-Title": "Melodic EDM Core V2 Colab Inference",
        },
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        body = json.load(response)
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response has no choices")
    content = choices[0].get("message", {}).get("content")
    return _parse_json_content(content), body


def enhance_prompt(
    idea: str,
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    explicit_conditions: dict[str, Any] | None = None,
    timeout: int = 120,
    max_attempts: int = 2,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Enhance one idea, retrying only when the deterministic gate rejects output."""
    idea = idea.strip()
    if not idea:
        raise ValueError("idea is required")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    explicit = dict(explicit_conditions or {})
    correction = ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw, body = _request_once(
                idea=idea,
                api_key=api_key,
                model=model,
                explicit_conditions=explicit,
                correction=correction,
                timeout=timeout,
                opener=opener,
            )
            validated = validate_conditions(raw, explicit)
        except (TypeError, ValueError) as error:
            correction = str(error)
            if attempt == max_attempts:
                raise ValueError(f"OpenRouter output failed the deterministic gate: {error}") from error
            continue
        return {
            "source_idea": idea,
            "requested_model": model,
            "resolved_model": body.get("model") or model,
            "conditions": validated,
            "usage": body.get("usage") or {},
        }
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("idea")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--keyscale")
    parser.add_argument("--timesignature")
    parser.add_argument(
        "--require-term",
        action="append",
        default=[],
        help="Exact user-authoritative term that the enhanced caption must retain; repeatable.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    explicit = {
        key: value for key, value in {
            "bpm": args.bpm,
            "keyscale": args.keyscale,
            "timesignature": args.timesignature,
        }.items() if value is not None
    }
    explicit["required_terms"] = args.require_term
    result = enhance_prompt(
        args.idea,
        os.environ.get("OPENROUTER_API_KEY", ""),
        model=args.model,
        explicit_conditions=explicit,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compile user-authoritative inference conditions without audio-annotation softening."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARTIST_SHORTCUTS = ("xomu", "thefatrat", "xu mengyuan", "徐梦圆")
QUALITY_SHORTCUTS = (
    "masterpiece", "best song ever", "professional quality", "extremely beautiful",
    "exactly like", "in the style of", "style of",
)


def compile_caption(
    spec: dict[str, str],
    *,
    min_words: int = 40,
    max_words: int = 80,
) -> str:
    if min_words < 1 or max_words < min_words:
        raise ValueError("invalid caption word range")
    required = ("genre", "mood", "melody", "arrangement", "production")
    missing = [key for key in required if not str(spec.get(key, "")).strip()]
    if missing:
        raise ValueError(f"missing musical fields: {', '.join(missing)}")
    source = " ".join(str(spec[key]) for key in required).casefold()
    if any(name in source for name in ARTIST_SHORTCUTS):
        raise ValueError("artist-name shortcuts are not allowed; describe audible traits")
    if any(phrase in source for phrase in QUALITY_SHORTCUTS):
        raise ValueError("quality/style shortcuts are not allowed; describe audible traits")
    mood = spec["mood"].strip()
    article = "an" if mood[:1].casefold() in "aeiou" else "a"
    caption = (
        f"Instrumental {spec['genre'].strip()} with {article} {mood} mood. "
        f"{spec['melody'].strip().rstrip('.')} . "
        f"{spec['arrangement'].strip().rstrip('.')} . "
        f"{spec['production'].strip().rstrip('.')} ."
    )
    caption = re.sub(r"\s+([.])", r"\1", re.sub(r"\s+", " ", caption)).strip()
    words = caption.split()
    if not min_words <= len(words) <= max_words:
        raise ValueError(
            f"compiled caption must contain {min_words}-{max_words} words, got {len(words)}"
        )
    return caption


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="JSON file containing genre, mood, melody, arrangement and production")
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    caption = compile_caption(spec)
    payload = json.dumps({"caption": caption, "word_count": len(caption.split())}, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

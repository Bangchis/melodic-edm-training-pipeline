#!/usr/bin/env python3
"""Compile only user-supplied musical facts into a 40-80 word training-style prompt."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARTIST_SHORTCUTS = ("xomu", "thefatrat", "xu mengyuan", "徐梦圆")


def compile_caption(spec: dict[str, str]) -> str:
    required = ("genre", "mood", "melody", "arrangement", "production")
    missing = [key for key in required if not str(spec.get(key, "")).strip()]
    if missing:
        raise ValueError(f"missing musical fields: {', '.join(missing)}")
    source = " ".join(str(spec[key]) for key in required).casefold()
    if any(name in source for name in ARTIST_SHORTCUTS):
        raise ValueError("artist-name shortcuts are not allowed; describe audible traits")
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
    if not 40 <= len(words) <= 80:
        raise ValueError(f"compiled caption must contain 40-80 words, got {len(words)}")
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

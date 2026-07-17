#!/usr/bin/env python3
"""Separate lyric-bearing samples on one GPU with resumable per-part state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def probe(path: Path) -> tuple[dict[str, Any] | None, str]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,sample_rate,channels", "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if result.returncode:
        return None, (result.stderr or result.stdout)[-600:].replace("\n", " ")
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"ffprobe_json_error:{exc}"


def validate_output(path: Path, expected_duration: float) -> tuple[bool, str, dict[str, Any]]:
    data, error = probe(path)
    if data is None:
        return False, "probe_failed:" + error, {}
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    details = {
        "duration": round(duration, 3),
        "sample_rate": int(stream.get("sample_rate") or 0) if stream else 0,
        "channels": int(stream.get("channels") or 0) if stream else 0,
    }
    if not stream:
        return False, "no_audio_stream", details
    if details["sample_rate"] != 48000 or details["channels"] != 2:
        return False, "format_mismatch", details
    if abs(duration - expected_duration) > 1.0:
        return False, "duration_mismatch", details
    decode = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    if decode.returncode:
        return False, "full_decode_failed:" + decode.stderr[-600:].replace("\n", " "), details
    return True, "", details


def make_preview(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp3")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(source),
            "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(tmp),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    if result.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("preview_failed:" + result.stderr[-600:].replace("\n", " "))
    os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/vocal_manifest.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--preset", choices=("instrumental_full", "instrumental_clean"), default="instrumental_full")
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.part_index < args.num_parts:
        raise SystemExit("part-index must be in [0, num-parts)")

    root = Path(args.project_root).resolve()
    manifest = read_jsonl(root / args.manifest)
    all_targets = sorted(
        (r for r in manifest if r.get("classification_status") == "complete" and r.get("vocal_status") == "lyrics"),
        key=lambda r: r["sample_id"],
    )
    targets = [
        row for row in all_targets
        if int(hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(), 16) % args.num_parts
        == args.part_index
    ]
    if args.limit is not None:
        targets = targets[:max(0, args.limit)]
    state_path = root / "data" / f"separation_{args.preset}_part{args.part_index}.jsonl"
    state = read_jsonl(state_path)
    by_id = {r["sample_id"]: r for r in state}

    from audio_separator.separator import Separator

    model_dir = root / "models" / "audio-separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(root / "data" / "separated"),
        output_format="FLAC",
        output_single_stem="Instrumental",
        sample_rate=48000,
        use_soundfile=True,
        use_autocast=True,
        ensemble_preset=args.preset,
    )
    separator.load_model()

    failures = 0
    for index, row in enumerate(targets, 1):
        sid = row["sample_id"]
        dest = root / "data" / "separated" / sid / args.preset
        output = dest / "instrumental.flac"
        preview = dest / "instrumental.mp3"
        prior = by_id.get(sid)
        if output.is_file() and preview.is_file():
            ok, error, details = validate_output(output, float(row["duration"]))
            if ok:
                if not prior or prior.get("separation_status") != "complete":
                    by_id[sid] = {
                        "sample_id": sid,
                        "record_key": row["record_key"],
                        "preset": args.preset,
                        "source_path": row["canonical_path"],
                        "separated_path": str(output),
                        "preview_path": str(preview),
                        "separation_status": "complete",
                        "elapsed_seconds": 0.0,
                        "separated_at": datetime.now(timezone.utc).isoformat(),
                        **details,
                    }
                    atomic_jsonl(state_path, sorted(by_id.values(), key=lambda r: r["sample_id"]))
                print(f"[{index}/{len(targets)}] {sid} SKIP complete", flush=True)
                continue
            print(f"[{index}/{len(targets)}] {sid} REBUILD {error}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            separator.output_dir = str(dest)
            produced = separator.separate(str(row["canonical_path"]), {"Instrumental": "instrumental"})
            if not output.is_file():
                candidates = [Path(p) for p in produced if "instrumental" in Path(p).name.lower()]
                if len(candidates) == 1 and candidates[0].is_file():
                    os.replace(candidates[0], output)
            ok, error, details = validate_output(output, float(row["duration"]))
            if not ok:
                raise RuntimeError(error)
            make_preview(output, preview)
            record = {
                "sample_id": sid,
                "record_key": row["record_key"],
                "preset": args.preset,
                "source_path": row["canonical_path"],
                "separated_path": str(output),
                "preview_path": str(preview),
                "separation_status": "complete",
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "separated_at": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(f"[{index}/{len(targets)}] {sid} PASS {record['elapsed_seconds']}s", flush=True)
        except Exception as exc:
            failures += 1
            record = {
                "sample_id": sid,
                "record_key": row["record_key"],
                "preset": args.preset,
                "source_path": row["canonical_path"],
                "separated_path": str(output),
                "preview_path": str(preview),
                "separation_status": "failed",
                "separation_error": f"{type(exc).__name__}:{exc}",
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "separated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"[{index}/{len(targets)}] {sid} FAILED {record['separation_error']}", flush=True)
        by_id[sid] = record
        atomic_jsonl(state_path, sorted(by_id.values(), key=lambda r: r["sample_id"]))

    complete = sum(r.get("separation_status") == "complete" for r in by_id.values())
    print(json.dumps({"part": args.part_index, "targets": len(targets), "complete": complete, "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

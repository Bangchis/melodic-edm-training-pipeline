#!/usr/bin/env python3
"""Recover lyric-free sections from separated stems using checked audio chunks."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any

from classify_vocals import classify, load_env_file, read_jsonl


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def render_preview(source: Path, target: Path, start: float, duration: float) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp3")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}",
            "-i", str(source), "-t", f"{duration:.3f}", "-ac", "2", "-ar", "44100",
            "-c:a", "libmp3lame", "-b:a", "128k", str(tmp),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("chunk_preview_failed:" + result.stderr[-400:].replace("\n", " "))
    os.replace(tmp, target)


def call_classifier(
    row: dict[str, Any], api_key: str, schema: dict[str, Any], model: str,
    retries: int, timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            return classify(row, api_key, schema, model, "minimal", timeout)
        except urllib.error.HTTPError as exc:
            last_error = f"http_{exc.code}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        if attempt < retries:
            time.sleep(min(60.0, 2 ** attempt + random.random()))
    raise RuntimeError(last_error)


def clean_runs(chunks: list[dict[str, Any]], stride: float, minimum: float) -> list[tuple[float, float]]:
    del stride  # intervals, rather than start order, define safe continuity

    def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        output: list[list[float]] = []
        for start, end in sorted(intervals):
            if output and start <= output[-1][1] + 0.1:
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        return [(start, end) for start, end in output]

    accepted = merge([
        (float(row["chunk_start"]), float(row["chunk_end"]))
        for row in chunks
        if row.get("classification_status") == "complete"
        and row.get("vocal_status") in {"instrumental", "vocal_chops"}
    ])
    rejected = merge([
        (float(row["chunk_start"]), float(row["chunk_end"]))
        for row in chunks
        if row.get("classification_status") != "complete"
        or row.get("vocal_status") not in {"instrumental", "vocal_chops"}
    ])
    safe = accepted
    for cut_start, cut_end in rejected:
        next_safe = []
        for start, end in safe:
            if cut_end <= start or cut_start >= end:
                next_safe.append((start, end))
                continue
            if start < cut_start:
                next_safe.append((start, cut_start))
            if cut_end < end:
                next_safe.append((cut_end, end))
        safe = next_safe
    return [(start, end) for start, end in safe if end - start >= minimum]


def align_to_beats(source: Path, start: float, end: float) -> tuple[float, float]:
    try:
        from essentia.standard import MonoLoader, RhythmExtractor2013

        audio = MonoLoader(filename=str(source), sampleRate=44100)()
        _, beats, _, _, _ = RhythmExtractor2013(method="multifeature")(audio)
        after = [float(value) for value in beats if start <= float(value) <= start + 3.0]
        before = [float(value) for value in beats if end - 3.0 <= float(value) <= end]
        aligned_start = min(after) if after else start
        aligned_end = max(before) if before else end
        if aligned_end - aligned_start >= 45.0:
            return round(aligned_start, 3), round(aligned_end, 3)
    except Exception:
        pass
    return round(start, 3), round(end, 3)


def render_flac(source: Path, target: Path, start: float, end: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.flac")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}",
            "-i", str(source), "-t", f"{end - start:.3f}", "-map", "0:a:0",
            "-ac", "2", "-ar", "48000", "-c:a", "flac", "-compression_level", "8", str(tmp),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("clean_section_render_failed:" + result.stderr[-400:].replace("\n", " "))
    os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pending", default="data/training_audio_pending.jsonl")
    parser.add_argument("--clean-check", default="data/separated_instrumental_clean_vocal_check.jsonl")
    parser.add_argument("--schema", default="configs/vocal_schema.json")
    parser.add_argument("--env-file", default="/workspace/.env")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    parser.add_argument("--stride", type=float, default=25.0)
    parser.add_argument("--minimum-clean-duration", type=float, default=50.0)
    parser.add_argument("--maximum-clean-duration", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    load_env_file(Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    schema = json.loads((root / args.schema).read_text(encoding="utf-8"))
    pending = read_jsonl(root / args.pending)
    clean_checks = {r["sample_id"]: r for r in read_jsonl(root / args.clean_check)}
    chunk_state_path = root / "data" / "clean_section_chunks.jsonl"
    search_state_path = root / "data" / "clean_section_search.jsonl"
    chunk_state = {r["chunk_id"]: r for r in read_jsonl(chunk_state_path)}
    search_state = {r["sample_id"]: r for r in read_jsonl(search_state_path)}

    for track_index, row in enumerate(pending, 1):
        sid = row["sample_id"]
        if search_state.get(sid, {}).get("search_status") == "accepted":
            continue
        clean = clean_checks.get(sid)
        if not clean or clean.get("vocal_status") != "lyrics":
            continue
        source = Path(clean["canonical_path"])
        duration = float(clean["duration"])
        starts = []
        value = 0.0
        while value < duration - 10.0:
            starts.append(value)
            value += args.stride
        track_chunks = []
        for chunk_index, start in enumerate(starts):
            chunk_id = f"{sid}__chunk{chunk_index:03d}"
            end = min(duration, start + args.chunk_duration)
            prior = chunk_state.get(chunk_id)
            if prior and prior.get("classification_status") == "complete":
                track_chunks.append(prior)
                continue
            preview = root / "data" / "clean_section_chunks" / sid / f"{chunk_index:03d}.mp3"
            render_preview(source, preview, start, end - start)
            classify_row = {
                **row,
                "sample_id": chunk_id,
                "preview_path": str(preview),
                "expected_title": f"{row.get('expected_title', '')} clean section candidate",
            }
            try:
                result, usage = call_classifier(
                    classify_row, api_key, schema, args.model, args.retries, args.timeout
                )
                record = {
                    "chunk_id": chunk_id,
                    "sample_id": sid,
                    "chunk_start": round(start, 3),
                    "chunk_end": round(end, 3),
                    "classification_status": "complete",
                    "classification_usage": usage,
                    **result,
                }
            except Exception as exc:
                record = {
                    "chunk_id": chunk_id,
                    "sample_id": sid,
                    "chunk_start": round(start, 3),
                    "chunk_end": round(end, 3),
                    "classification_status": "failed",
                    "classification_error": f"{type(exc).__name__}:{exc}",
                }
            chunk_state[chunk_id] = record
            track_chunks.append(record)
            atomic_jsonl(chunk_state_path, sorted(chunk_state.values(), key=lambda item: item["chunk_id"]))
            print(
                f"[{track_index}/{len(pending)}] {chunk_id} {record.get('vocal_status', 'FAILED')}",
                flush=True,
            )

        runs = clean_runs(track_chunks, args.stride, args.minimum_clean_duration)
        if not runs:
            search_state[sid] = {
                **row,
                "search_status": "rejected",
                "search_reason": "no_contiguous_checked_clean_section",
            }
            atomic_jsonl(search_state_path, sorted(search_state.values(), key=lambda item: item["sample_id"]))
            continue
        start, end = max(runs, key=lambda pair: (pair[1] - pair[0], -pair[0]))
        if end - start > args.maximum_clean_duration:
            end = start + args.maximum_clean_duration
        start, end = align_to_beats(source, start, end)
        section_dir = root / "data" / "separated" / sid / "clean_section"
        output = section_dir / "instrumental.flac"
        preview = section_dir / "instrumental.mp3"
        render_flac(source, output, start, end)
        preview.unlink(missing_ok=True)
        render_preview(output, preview, 0.0, end - start)
        candidate_row = {
            **row,
            "canonical_path": str(output),
            "preview_path": str(preview),
            "duration": round(end - start, 3),
            "sample_rate": 48000,
            "channels": 2,
            "audio_valid": True,
            "section_start": start,
            "section_end": end,
        }
        try:
            result, usage = call_classifier(
                candidate_row, api_key, schema, args.model, args.retries, args.timeout
            )
            accepted = result["vocal_status"] in {"instrumental", "vocal_chops"}
            search_state[sid] = {
                **candidate_row,
                **result,
                "classification_status": "complete",
                "classification_usage": usage,
                "search_status": "accepted" if accepted else "rejected",
                "search_reason": None if accepted else "combined_section_recheck_failed",
            }
        except Exception as exc:
            search_state[sid] = {
                **candidate_row,
                "classification_status": "failed",
                "search_status": "rejected",
                "search_reason": f"{type(exc).__name__}:{exc}",
            }
        atomic_jsonl(search_state_path, sorted(search_state.values(), key=lambda item: item["sample_id"]))
        print(f"[{track_index}/{len(pending)}] {sid} {search_state[sid]['search_status']}", flush=True)

    final = [search_state[r["sample_id"]] for r in pending if r["sample_id"] in search_state]
    accepted = [r for r in final if r.get("search_status") == "accepted"]
    atomic_jsonl(root / "data" / "clean_section_manifest.jsonl", accepted)
    print(json.dumps({"searched": len(final), "accepted": len(accepted), "rejected": len(final) - len(accepted)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

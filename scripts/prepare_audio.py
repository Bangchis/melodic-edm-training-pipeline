#!/usr/bin/env python3
"""Validate and canonicalize every downloaded catalog record without deduplication."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ACCEPTED_CHECKPOINT_STATUSES = {"COMPLETE", "REUSED_DUPLICATE"}
MIN_DURATION = 30.0
MAX_DURATION = 600.0
MAX_SILENCE_RATIO = 0.80


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id(record_key: str) -> str:
    value = record_key.replace(":", "__")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    if not value:
        raise ValueError(f"invalid record_key: {record_key!r}")
    return value


def run(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def ffprobe(path: Path) -> tuple[dict[str, Any] | None, str]:
    result = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ], timeout=120)
    if result.returncode:
        return None, (result.stderr or result.stdout)[-1000:].replace("\n", " ")
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"ffprobe_json_error:{exc}"


def audio_properties(probe: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    try:
        duration = float(probe.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
    return duration, stream


def silence_ratio(path: Path, duration: float) -> tuple[float | None, str]:
    result = run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", "silencedetect=noise=-50dB:d=2", "-f", "null", "-",
    ], timeout=900)
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    if result.returncode:
        return None, "full_decode_failed:" + output[-1000:].replace("\n", " ")
    total = sum(float(x) for x in re.findall(r"silence_duration:\s*([0-9.]+)", output))
    return (min(1.0, total / duration) if duration > 0 else None), ""


def canonicalize(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.stem + ".", suffix=".tmp.flac", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.unlink(missing_ok=True)
    result = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:a:0", "-ac", "2", "-ar", "48000", "-c:a", "flac",
        "-compression_level", "8", str(tmp),
    ])
    if result.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "canonicalize_failed:" + ((result.stderr or result.stdout)[-1000:].replace("\n", " "))
    os.replace(tmp, target)
    return ""


def make_preview(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp3")
    tmp.unlink(missing_ok=True)
    result = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:a:0", "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame",
        "-b:a", "128k", str(tmp),
    ])
    if result.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "preview_failed:" + ((result.stderr or result.stdout)[-1000:].replace("\n", " "))
    os.replace(tmp, target)
    return ""


def process_one(row: dict[str, str], selected: dict[str, str], project_root: Path) -> dict[str, Any]:
    sid = sample_id(row["record_key"])
    source = Path(row.get("audio_path", ""))
    canonical = project_root / "data" / "canonical" / f"{sid}.flac"
    preview = project_root / "data" / "annotation_preview" / f"{sid}.mp3"
    result: dict[str, Any] = {
        "sample_id": sid,
        "record_key": row["record_key"],
        "source_id": row.get("source_id", ""),
        "source_rank": int(row.get("source_rank") or 0),
        "video_id": row.get("video_id", ""),
        "expected_title": selected.get("expected_title", ""),
        "expected_artist": selected.get("expected_artist", ""),
        "expected_version": selected.get("expected_version", ""),
        "raw_path": str(source),
        "canonical_path": str(canonical),
        "preview_path": str(preview),
        "download_status": row.get("status", ""),
        "download_reason": row.get("reason", ""),
        "user_identity_override": "user_override_identity:" in row.get("reason", ""),
        "audio_valid": False,
        "reject_reason": None,
    }
    if row.get("status") not in ACCEPTED_CHECKPOINT_STATUSES:
        result["reject_reason"] = "not_downloaded:" + row.get("status", "")
        return result
    if not source.is_file():
        result["reject_reason"] = "source_missing"
        return result

    source_probe, error = ffprobe(source)
    if error or source_probe is None:
        result["reject_reason"] = "source_probe_failed:" + error
        return result
    source_duration, source_stream = audio_properties(source_probe)
    result["source_duration"] = round(source_duration, 3)
    result["source_codec"] = source_stream.get("codec_name") if source_stream else None
    result["source_sample_rate"] = int(source_stream.get("sample_rate") or 0) if source_stream else 0
    result["source_channels"] = int(source_stream.get("channels") or 0) if source_stream else 0
    if not source_stream:
        result["reject_reason"] = "no_audio_stream"
        return result
    if not MIN_DURATION <= source_duration <= MAX_DURATION:
        result["reject_reason"] = f"source_duration_out_of_range:{source_duration:.3f}"
        return result

    if not canonical.exists():
        error = canonicalize(source, canonical)
        if error:
            result["reject_reason"] = error
            return result
    canonical_probe, error = ffprobe(canonical)
    if error or canonical_probe is None:
        result["reject_reason"] = "canonical_probe_failed:" + error
        return result
    canonical_duration, canonical_stream = audio_properties(canonical_probe)
    result["duration"] = round(canonical_duration, 3)
    result["sample_rate"] = int(canonical_stream.get("sample_rate") or 0) if canonical_stream else 0
    result["channels"] = int(canonical_stream.get("channels") or 0) if canonical_stream else 0
    if not canonical_stream or result["sample_rate"] != 48000 or result["channels"] != 2:
        result["reject_reason"] = "canonical_format_mismatch"
        return result
    if abs(canonical_duration - source_duration) > 1.0:
        result["reject_reason"] = f"canonical_duration_mismatch:{source_duration:.3f}!={canonical_duration:.3f}"
        return result

    ratio, error = silence_ratio(canonical, canonical_duration)
    if error:
        result["reject_reason"] = error
        return result
    result["silence_ratio"] = round(ratio or 0.0, 6)
    if ratio is not None and ratio > MAX_SILENCE_RATIO:
        result["reject_reason"] = f"mostly_silence:{ratio:.4f}"
        return result

    if not preview.exists():
        error = make_preview(canonical, preview)
        if error:
            result["reject_reason"] = error
            return result
    preview_probe, error = ffprobe(preview)
    if error or preview_probe is None:
        result["reject_reason"] = "preview_probe_failed:" + error
        return result
    preview_duration, preview_stream = audio_properties(preview_probe)
    if not preview_stream or abs(preview_duration - canonical_duration) > 1.0:
        result["reject_reason"] = "preview_validation_failed"
        return result

    result["audio_valid"] = True
    return result


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    checkpoint = read_csv(Path(args.checkpoint))
    selection = {r["record_key"]: r for r in read_csv(Path(args.selection))}
    if len(checkpoint) != 240:
        raise SystemExit(f"expected 240 checkpoint rows, found {len(checkpoint)}")
    project_root = Path(args.project_root).resolve()
    for name in (
        "canonical", "annotation_preview", "separated", "training_audio", "mir",
        "annotations", "final_dataset", "tensors_part0", "tensors_part1", "tensors_all",
    ):
        (project_root / "data" / name).mkdir(parents=True, exist_ok=True)
    for name in ("smoke", "training", "inference", "release"):
        (project_root / "outputs" / name).mkdir(parents=True, exist_ok=True)
    for name in ("logs", "checkpoints"):
        (project_root / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(process_one, row, selection.get(row["record_key"], {}), project_root): row["record_key"]
            for row in checkpoint
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            record_key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                selected_row = selection.get(record_key, {})
                result = {
                    "sample_id": sample_id(record_key),
                    "record_key": record_key,
                    "source_id": selected_row.get("source_id", record_key.split(":", 1)[0]),
                    "source_rank": int(selected_row.get("source_rank") or 0),
                    "audio_valid": False,
                    "reject_reason": f"worker_exception:{type(exc).__name__}:{exc}",
                }
            rows.append(result)
            done += 1
            print(f"[{done}/240] {result['record_key']} {'PASS' if result['audio_valid'] else 'REJECT'}"
                  f" {result.get('reject_reason') or ''}", flush=True)
    rows.sort(key=lambda r: (r["source_id"], r["source_rank"]))
    accepted = [r for r in rows if r["audio_valid"]]
    rejected = [r for r in rows if not r["audio_valid"]]
    atomic_jsonl(project_root / "data" / "audio_manifest.jsonl", rows)
    atomic_jsonl(project_root / "data" / "rejected_downloads.jsonl", rejected)
    report = {
        "records": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "canonical_files": len(list((project_root / "data" / "canonical").glob("*.flac"))),
        "preview_files": len(list((project_root / "data" / "annotation_preview").glob("*.mp3"))),
        "deduplication_performed": False,
    }
    (project_root / "data" / "audio_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not [r for r in rejected if not r["reject_reason"].startswith("not_downloaded:")] else 1


if __name__ == "__main__":
    raise SystemExit(main())

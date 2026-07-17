#!/usr/bin/env python3
"""Build and validate a record-preserving ACE-Step train/validation dataset."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
        return None, (result.stderr or result.stdout)[-500:].replace("\n", " ")
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"ffprobe_json:{exc}"


def choose_window(duration: float, mir: dict[str, Any], max_duration: float) -> tuple[float, float]:
    if duration <= max_duration:
        return 0.0, duration
    downbeats = sorted({float(value) for value in mir.get("downbeats", []) if 0 <= float(value) <= duration})
    if not downbeats:
        raise ValueError("overlength_audio_has_no_downbeats")
    starts = [0.0] + downbeats
    ends = downbeats + [duration]
    weights = {
        "Final Drop": 5.0,
        "Drop": 4.0,
        "Build": 2.5,
        "Theme": 2.0,
        "Intro": 1.0,
        "Break": 1.0,
        "Outro": 1.0,
    }
    candidates = []
    for start in starts:
        possible = [end for end in ends if 60.0 <= end - start <= max_duration + 1e-6]
        if not possible:
            continue
        end = max(possible)
        score = 0.0
        for section in mir.get("sections", []):
            overlap = max(0.0, min(end, float(section["end"])) - max(start, float(section["start"])))
            score += overlap * weights.get(section.get("label", ""), 1.0)
        candidates.append((score, end - start, -start, start, end))
    if not candidates:
        raise ValueError("no_valid_downbeat_window")
    _, _, _, start, end = max(candidates)
    return round(start, 3), round(end, 3)


def render_audio(source: Path, target: Path, start: float, end: float, full_duration: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if start == 0.0 and abs(end - full_duration) <= 0.01:
        # Preserve the unique sample_id basename even if downstream libraries
        # resolve symlinks. A hard link consumes no additional audio blocks and
        # prevents separated files named ``instrumental.flac`` from collapsing.
        source = source.resolve(strict=True)
        if target.exists() and not target.is_symlink() and target.samefile(source):
            return
        target.unlink(missing_ok=True)
        os.link(source, target)
        return
    if target.is_symlink():
        target.unlink()
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
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    if result.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("trim_failed:" + result.stderr[-500:].replace("\n", " "))
    os.replace(tmp, target)


def section_labels(mir: dict[str, Any], start: float, end: float) -> list[str]:
    labels = []
    allowed = {"Intro", "Theme", "Build", "Drop", "Break", "Final Drop", "Outro"}
    for section in mir.get("sections", []):
        if min(end, float(section["end"])) - max(start, float(section["start"])) <= 1.0:
            continue
        label = section.get("label")
        if label in allowed and (not labels or labels[-1] != label):
            labels.append(label)
    return labels


def lyrics_text(labels: list[str]) -> str:
    if not labels:
        return "[Instrumental]\n"
    return "\n\n".join(f"[{label}]\n[Instrumental]" for label in labels) + "\n"


def choose_splits(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = row.get("raw_path") or row.get("video_id") or row["sample_id"]
        groups[str(group)].append(row["sample_id"])
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    target = round(len(rows) * val_ratio)
    validation: set[str] = set()
    count = 0
    for _, members in items:
        if count >= target:
            break
        validation.update(members)
        count += len(members)
    return {row["sample_id"]: ("validation" if row["sample_id"] in validation else "train") for row in rows}


def link_triplet(source_dir: Path, target_dir: Path, sid: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".flac", ".json", ".lyrics.txt"):
        source = source_dir / f"{sid}{suffix}"
        target = target_dir / source.name
        if target.is_symlink() and target.resolve() != source.resolve():
            target.unlink()
        if not target.exists():
            target.symlink_to(source)


def write_dataset_index(path: Path, rows: list[dict[str, Any]], audio_dir: Path) -> None:
    """Write ACE-Step's aggregate JSON while retaining sidecar triplets."""
    samples = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        sid = row["sample_id"]
        metadata = json.loads((Path(row["final_json_path"])).read_text(encoding="utf-8"))
        lyrics = Path(row["final_lyrics_path"]).read_text(encoding="utf-8")
        samples.append({
            "audio_path": str(audio_dir / f"{sid}.flac"),
            "filename": f"{sid}.flac",
            "caption": metadata["caption"],
            "caption_variants": metadata["caption_variants"],
            "lyrics": lyrics,
            "bpm": metadata.get("bpm"),
            "keyscale": metadata.get("keyscale", ""),
            "timesignature": metadata.get("timesignature", ""),
            "duration": row["final_duration"],
            "is_instrumental": True,
        })
    atomic_json(path, {
        "metadata": {"genre_ratio": 0, "tag_position": "prepend", "custom_tag": ""},
        "samples": samples,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    source = [r for r in read_jsonl(root / args.manifest) if r.get("quality_status") == "accepted"]
    annotations = {}
    mir_by_id = {}
    preflight_errors = []
    for row in source:
        sid = row["sample_id"]
        annotation_path = root / "data" / "annotations" / f"{sid}.json"
        mir_path = root / "data" / "mir" / f"{sid}.json"
        if not annotation_path.is_file():
            preflight_errors.append({"sample_id": sid, "reason": "annotation_missing"})
            continue
        if not mir_path.is_file():
            preflight_errors.append({"sample_id": sid, "reason": "mir_missing"})
            continue
        annotations[sid] = json.loads(annotation_path.read_text(encoding="utf-8"))
        mir_by_id[sid] = json.loads(mir_path.read_text(encoding="utf-8"))
    if preflight_errors:
        atomic_json(root / "data" / "final_validation_report.json", {
            "status": "preflight_failed", "errors": preflight_errors,
        })
        print(json.dumps({"preflight_errors": len(preflight_errors)}))
        return 1

    splits = choose_splits(source, args.validation_ratio, args.seed)
    final_root = root / "data" / "final_dataset"
    for split in ("train", "validation"):
        (final_root / split).mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    for index, row in enumerate(source, 1):
        sid = row["sample_id"]
        split = splits[sid]
        output_dir = final_root / split
        audio = output_dir / f"{sid}.flac"
        try:
            mir = mir_by_id[sid]
            annotation_record = annotations[sid]
            annotation = annotation_record["annotation"]
            full_duration = float(row["duration"])
            start, end = choose_window(full_duration, mir, args.max_duration)
            render_audio(Path(row["training_audio_path"]), audio, start, end, full_duration)
            variants = {item["type"]: item["text"] for item in annotation["caption_variants"]}
            captions = [variants[name] for name in ("full", "composition", "production", "tags")]
            metadata: dict[str, Any] = {
                "caption": annotation["canonical_caption"],
                "caption_variants": captions,
                "language": "instrumental",
                "master_annotation": annotation,
            }
            for field in ("bpm", "keyscale", "timesignature"):
                if mir.get(field) not in {None, ""}:
                    metadata[field] = mir[field]
            atomic_json(output_dir / f"{sid}.json", metadata)
            labels = section_labels(mir, start, end)
            lyrics = output_dir / f"{sid}.lyrics.txt"
            tmp_lyrics = lyrics.with_suffix(lyrics.suffix + ".tmp")
            tmp_lyrics.write_text(lyrics_text(labels), encoding="utf-8")
            os.replace(tmp_lyrics, lyrics)

            probe_data, error = probe(audio)
            if probe_data is None:
                raise RuntimeError("output_probe_failed:" + error)
            stream = next((x for x in probe_data.get("streams", []) if x.get("codec_type") == "audio"), None)
            duration = float(probe_data.get("format", {}).get("duration") or 0)
            if not stream or int(stream.get("sample_rate") or 0) != 48000 or int(stream.get("channels") or 0) != 2:
                raise RuntimeError("output_format_mismatch")
            if not 30 <= duration <= args.max_duration + 0.1:
                raise RuntimeError(f"output_duration_invalid:{duration}")
            records.append({
                **row,
                "split": split,
                "final_audio_path": str(audio),
                "final_json_path": str(output_dir / f"{sid}.json"),
                "final_lyrics_path": str(lyrics),
                "trim_start": start,
                "trim_end": end,
                "final_duration": round(duration, 3),
                "trimmed": start > 0 or end < full_duration - 0.01,
                "validation_status": "pass",
            })
            print(f"[{index}/{len(source)}] {sid} PASS {split} {duration:.1f}s", flush=True)
        except Exception as exc:
            errors.append({"sample_id": sid, "record_key": row["record_key"], "reason": f"{type(exc).__name__}:{exc}"})
            print(f"[{index}/{len(source)}] {sid} FAILED {errors[-1]['reason']}", flush=True)

    valid_ids = {row["sample_id"] for row in records}
    for split in ("train", "validation"):
        for path in (final_root / split).glob("*"):
            sid = path.name.split(".", 1)[0]
            if sid not in valid_ids and (path.is_file() or path.is_symlink()):
                path.unlink()

    train_rows = [row for row in records if row["split"] == "train"]
    part_rows = [[], []]
    part_duration = [0.0, 0.0]
    for row in sorted(train_rows, key=lambda item: float(item["final_duration"]), reverse=True):
        part = 0 if part_duration[0] <= part_duration[1] else 1
        part_rows[part].append(row)
        part_duration[part] += float(row["final_duration"])
    for part in (0, 1):
        target = root / "data" / f"final_dataset_part{part}"
        target.mkdir(parents=True, exist_ok=True)
        wanted = {row["sample_id"] for row in part_rows[part]}
        for row in part_rows[part]:
            link_triplet(final_root / "train", target, row["sample_id"])
        for path in target.glob("*"):
            sid = path.name.split(".", 1)[0]
            if sid not in wanted and (path.is_file() or path.is_symlink()):
                path.unlink()

    write_dataset_index(
        root / "data" / "dataset_train_part0.json",
        part_rows[0],
        root / "data" / "final_dataset_part0",
    )
    write_dataset_index(
        root / "data" / "dataset_train_part1.json",
        part_rows[1],
        root / "data" / "final_dataset_part1",
    )
    validation_rows = [row for row in records if row["split"] == "validation"]
    write_dataset_index(
        root / "data" / "dataset_validation.json",
        validation_rows,
        final_root / "validation",
    )

    atomic_jsonl(root / "data" / "final_manifest.jsonl", records)
    raw_counts = Counter(str(row.get("raw_path") or row.get("video_id")) for row in source)
    report = {
        "status": "pass" if not errors and len(records) == len(source) else "failed",
        "input_records": len(source),
        "validated_records": len(records),
        "errors": errors,
        "deduplication_performed": False,
        "shared_audio_records_preserved": sum(count - 1 for count in raw_counts.values()),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "test_records": 0,
        "trimmed_records": sum(bool(row["trimmed"]) for row in records),
        "part0_records": len(part_rows[0]),
        "part1_records": len(part_rows[1]),
        "part0_hours": round(part_duration[0] / 3600, 3),
        "part1_hours": round(part_duration[1] / 3600, 3),
    }
    atomic_json(root / "data" / "final_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

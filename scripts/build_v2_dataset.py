#!/usr/bin/env python3
"""Build grouped v2 ACE-Step datasets without duplicating audio bytes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from v2_common import CAPTION_TYPES, atomic_json, atomic_jsonl, read_jsonl


def safe_symlink(source: Path, target: Path) -> None:
    """Create or refresh a relative symlink while refusing real-file overwrite."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"refusing to overwrite non-symlink {target}")
    target.symlink_to(os.path.relpath(source, target.parent))


def metadata_for(root: Path, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return ACE-Step metadata and lyrics for one merged annotation."""
    annotation = json.loads(Path(row["v2_annotation_path"]).read_text(encoding="utf-8"))
    variants = annotation["caption_variants"]
    if [item["type"] for item in variants] != list(CAPTION_TYPES):
        raise ValueError(f"{row['sample_id']}: caption order is not canonical/composition/production")
    mir = json.loads((root / "data" / "mir" / f"{row['sample_id']}.json").read_text(encoding="utf-8"))
    lyrics = Path(row["final_lyrics_path"]).read_text(encoding="utf-8")
    metadata: dict[str, Any] = {
        "caption": variants[0]["text"],
        "caption_variants": [item["text"] for item in variants],
        "caption_variant_types": list(CAPTION_TYPES),
        "language": "instrumental",
        "parent_song_id": row["parent_song_id"],
        "split": row["split"],
        "master_annotation": annotation["master_annotation"],
    }
    for field in ("bpm", "keyscale", "timesignature"):
        if mir.get(field) not in (None, ""):
            metadata[field] = mir[field]
    return metadata, lyrics


def write_sidecars(
    target_dir: Path,
    sample_id: str,
    audio_source: Path,
    metadata: dict[str, Any],
    lyrics: str,
) -> None:
    """Link audio and atomically write v2 metadata/lyrics sidecars."""
    safe_symlink(audio_source, target_dir / f"{sample_id}.flac")
    atomic_json(target_dir / f"{sample_id}.json", metadata)
    lyrics_path = target_dir / f"{sample_id}.lyrics.txt"
    temporary = lyrics_path.with_suffix(lyrics_path.suffix + ".tmp")
    temporary.write_text(lyrics, encoding="utf-8")
    os.replace(temporary, lyrics_path)


def dataset_sample(row: dict[str, Any], target_dir: Path, metadata: dict[str, Any], lyrics: str) -> dict[str, Any]:
    """Create one aggregate ACE-Step dataset record."""
    sample_id = str(row["sample_id"])
    return {
        "audio_path": str(target_dir / f"{sample_id}.flac"),
        "filename": f"{sample_id}.flac",
        "caption": metadata["caption"],
        "caption_variants": metadata["caption_variants"],
        "caption_variant_types": metadata["caption_variant_types"],
        "lyrics": lyrics,
        "bpm": metadata.get("bpm"),
        "keyscale": metadata.get("keyscale", ""),
        "timesignature": metadata.get("timesignature", ""),
        "duration": row["final_duration"],
        "is_instrumental": True,
        "parent_song_id": row["parent_song_id"],
        "split": row["split"],
    }


def write_index(path: Path, samples: list[dict[str, Any]], split: str) -> None:
    """Write one aggregate dataset index with immutable prompt semantics."""
    atomic_json(path, {
        "metadata": {
            "schema_version": "2.0",
            "split": split,
            "genre_ratio": 0,
            "tag_position": "prepend",
            "custom_tag": "",
            "caption_variant_types": list(CAPTION_TYPES),
            "training_prompt_selection": "uniform_random",
            "validation_prompt_selection": "canonical_index_0",
        },
        "samples": samples,
    })


def balanced_parts(rows: list[dict[str, Any]], parts: int = 2) -> list[list[dict[str, Any]]]:
    """Balance rows by duration for parallel preprocessing."""
    output: list[list[dict[str, Any]]] = [[] for _ in range(parts)]
    durations = [0.0] * parts
    for row in sorted(rows, key=lambda item: float(item["final_duration"]), reverse=True):
        index = min(range(parts), key=lambda part: (durations[part], len(output[part]), part))
        output[index].append(row)
        durations[index] += float(row["final_duration"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = sorted(read_jsonl(root / "data_v2" / "manifest.jsonl"), key=lambda row: row["sample_id"])
    if len(rows) != 231:
        raise ValueError(f"expected 231 merged rows, found {len(rows)}")

    dataset_root = root / "data_v2" / "final_dataset"
    samples_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "all": []}
    metadata_cache: dict[str, tuple[dict[str, Any], str]] = {}
    for index, row in enumerate(rows, 1):
        sample_id = str(row["sample_id"])
        metadata, lyrics = metadata_for(root, row)
        metadata_cache[sample_id] = (metadata, lyrics)
        audio_source = Path(row["final_audio_path"])
        split_dir = dataset_root / row["split"]
        all_dir = dataset_root / "all"
        write_sidecars(split_dir, sample_id, audio_source, metadata, lyrics)
        write_sidecars(all_dir, sample_id, audio_source, metadata, lyrics)
        samples_by_split[row["split"]].append(dataset_sample(row, split_dir, metadata, lyrics))
        samples_by_split["all"].append(dataset_sample(row, all_dir, metadata, lyrics))
        print(f"[{index}/{len(rows)}] {sample_id} LINKED {row['split']}", flush=True)

    write_index(root / "data_v2" / "dataset_train.json", samples_by_split["train"], "train")
    write_index(
        root / "data_v2" / "dataset_validation.json",
        samples_by_split["validation"],
        "validation",
    )
    write_index(root / "data_v2" / "dataset_all.json", samples_by_split["all"], "all")

    for split_name, selected_rows in (
        ("train", [row for row in rows if row["split"] == "train"]),
        ("all", rows),
    ):
        for part_index, part_rows in enumerate(balanced_parts(selected_rows)):
            part_dir = root / "data_v2" / f"final_dataset_{split_name}_part{part_index}"
            part_samples = []
            for row in sorted(part_rows, key=lambda item: item["sample_id"]):
                sample_id = str(row["sample_id"])
                metadata, lyrics = metadata_cache[sample_id]
                write_sidecars(part_dir, sample_id, Path(row["final_audio_path"]), metadata, lyrics)
                part_samples.append(dataset_sample(row, part_dir, metadata, lyrics))
            write_index(
                root / "data_v2" / f"dataset_{split_name}_part{part_index}.json",
                part_samples,
                f"{split_name}_part{part_index}",
            )

    report = {
        "status": "pass",
        "records": len(rows),
        "train_records": len(samples_by_split["train"]),
        "validation_records": len(samples_by_split["validation"]),
        "all_records": len(samples_by_split["all"]),
        "test_records": 0,
        "caption_variants_per_record": 3,
        "training_prompt_types": list(CAPTION_TYPES),
        "audio_storage": "relative_symlinks_to_validated_v1_audio",
        "deduplication_performed": False,
    }
    atomic_json(root / "data_v2" / "dataset_build_report.json", report)
    atomic_jsonl(root / "data_v2" / "dataset_manifest.jsonl", rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

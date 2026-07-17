#!/usr/bin/env python3
"""Choose original/full/clean audio after vocal rechecks and build symlink set."""
from __future__ import annotations

import argparse
import json
import os
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


def acceptable(check: dict[str, Any] | None) -> bool:
    return bool(
        check
        and check.get("classification_status") == "complete"
        and check.get("vocal_status") in {"instrumental", "vocal_chops"}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--vocal-manifest", default="data/vocal_manifest.jsonl")
    parser.add_argument("--full-check", default="data/separated_instrumental_full_vocal_check.jsonl")
    parser.add_argument("--clean-check", default="data/separated_instrumental_clean_vocal_check.jsonl")
    parser.add_argument("--clean-sections", default="data/clean_section_manifest.jsonl")
    parser.add_argument("--section-search", default="data/clean_section_search.jsonl")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    vocals = read_jsonl(root / args.vocal_manifest)
    full = {r["sample_id"]: r for r in read_jsonl(root / args.full_check)}
    clean = {r["sample_id"]: r for r in read_jsonl(root / args.clean_check)}
    sections = {r["sample_id"]: r for r in read_jsonl(root / args.clean_sections)}
    section_search = {r["sample_id"]: r for r in read_jsonl(root / args.section_search)}
    training_dir = root / "data" / "training_audio"
    training_dir.mkdir(parents=True, exist_ok=True)

    accepted = []
    pending = []
    rejected = []
    clean_targets = []
    selected_ids = set()
    for row in sorted(vocals, key=lambda r: (r.get("source_id", ""), int(r.get("source_rank") or 0))):
        sid = row["sample_id"]
        original_status = row.get("vocal_status")
        source: Path | None = None
        selected_duration = float(row.get("duration") or 0)
        audio_source = ""
        final_status = original_status
        preset = None
        if original_status in {"instrumental", "vocal_chops"}:
            source = Path(row["canonical_path"])
            audio_source = "original"
        elif original_status == "lyrics" and acceptable(full.get(sid)):
            source = Path(full[sid]["canonical_path"])
            audio_source = "separated_full"
            final_status = full[sid]["vocal_status"]
            preset = "instrumental_full"
        elif original_status == "lyrics" and acceptable(clean.get(sid)):
            source = Path(clean[sid]["canonical_path"])
            audio_source = "separated_clean"
            final_status = clean[sid]["vocal_status"]
            preset = "instrumental_clean"
        elif original_status == "lyrics" and sections.get(sid, {}).get("search_status") == "accepted":
            source = Path(sections[sid]["canonical_path"])
            audio_source = "clean_section"
            final_status = sections[sid]["vocal_status"]
            preset = "instrumental_clean+checked_section"
            selected_duration = float(sections[sid]["duration"])
        else:
            section_result = section_search.get(sid, {})
            if (
                original_status == "lyrics"
                and clean.get(sid, {}).get("vocal_status") == "lyrics"
                and section_result.get("search_status") == "rejected"
            ):
                rejected.append({
                    **row,
                    "quality_status": "rejected",
                    "reject_reason": "lyrics_remain_and_no_checked_clean_section",
                    "full_recheck_status": full.get(sid, {}).get("vocal_status"),
                    "clean_recheck_status": clean.get(sid, {}).get("vocal_status"),
                    "section_search_reason": section_result.get("search_reason"),
                })
                continue
            reason = "needs_clean_separation" if original_status == "lyrics" and sid in full else "vocal_check_incomplete"
            pending.append({
                **row,
                "quality_status": "pending",
                "pending_reason": reason,
                "full_recheck_status": full.get(sid, {}).get("vocal_status"),
                "clean_recheck_status": clean.get(sid, {}).get("vocal_status"),
            })
            if original_status == "lyrics":
                clean_targets.append(row)
            continue

        if not source or not source.is_file():
            pending.append({**row, "quality_status": "pending", "pending_reason": "selected_source_missing"})
            continue
        target = training_dir / f"{sid}.flac"
        if target.is_symlink() and target.resolve() != source.resolve():
            target.unlink()
        if not target.exists():
            target.symlink_to(source)
        selected_ids.add(sid)
        accepted.append({
            **row,
            "duration": selected_duration,
            "training_audio_path": str(target),
            "training_audio_source_path": str(source),
            "audio_source": audio_source,
            "vocal_status_after_processing": final_status,
            "separator_preset": preset,
            "quality_status": "accepted",
        })

    for path in training_dir.glob("*.flac"):
        if path.stem not in selected_ids and path.is_symlink():
            path.unlink()
    atomic_jsonl(root / "data" / "training_audio_manifest.jsonl", accepted)
    atomic_jsonl(root / "data" / "training_audio_pending.jsonl", pending)
    atomic_jsonl(root / "data" / "training_audio_rejected.jsonl", rejected)
    atomic_jsonl(root / "data" / "vocal_clean_targets.jsonl", clean_targets)
    report = {
        "vocal_records": len(vocals),
        "accepted": len(accepted),
        "pending": len(pending),
        "rejected": len(rejected),
        "clean_targets": len(clean_targets),
        "training_audio_files": len(list(training_dir.glob("*.flac"))),
    }
    (root / "data" / "training_audio_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())

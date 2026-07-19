#!/usr/bin/env python3
"""Analyze BPM, beats/downbeats, sections and key with resumable GPU parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def map_sections(segments: list[Any]) -> list[dict[str, Any]]:
    chorus_indices = [i for i, segment in enumerate(segments) if segment.label == "chorus"]
    last_chorus = chorus_indices[-1] if len(chorus_indices) > 1 else -1
    labels = {
        "intro": "Intro",
        "outro": "Outro",
        "break": "Break",
        "bridge": "Break",
        "inst": "Theme",
        "solo": "Theme",
        "verse": "Theme",
        "pre-chorus": "Build",
        "prechorus": "Build",
        "pre_chorus": "Build",
        "chorus": "Drop",
        "start": "Intro",
        "end": "Outro",
    }
    output = []
    for index, segment in enumerate(segments):
        label = labels.get(segment.label, "Theme")
        if index == last_chorus:
            label = "Final Drop"
        output.append({
            "label": label,
            "raw_label": segment.label,
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
        })
    return output


def infer_timesignature(beat_positions: list[int], downbeats: list[float]) -> str | None:
    positions = [int(value) for value in beat_positions]
    if len(positions) < 16 or len(downbeats) < 4:
        return None
    valid = sum(value in {1, 2, 3, 4} for value in positions) / len(positions)
    return "4" if valid >= 0.90 and max(positions) == 4 else None


def normalize_edm_bpm(raw_bpm: int) -> tuple[int | None, str | None]:
    """Normalize an unambiguous half-time estimate for this EDM-only corpus."""
    if not 40 <= raw_bpm <= 250:
        return None, None
    if raw_bpm < 80 and raw_bpm * 2 <= 250:
        return raw_bpm * 2, "double_half_time_below_80"
    return raw_bpm, None


def estimate_key(audio_path: Path, threshold: float) -> tuple[str | None, float]:
    from essentia.standard import KeyExtractor, MonoLoader

    audio = MonoLoader(filename=str(audio_path), sampleRate=44100)()
    key, scale, strength = KeyExtractor(sampleRate=44100, profileType="edma")(audio)
    confidence = float(strength)
    keyscale = f"{key} {scale}" if key and scale and confidence >= threshold else None
    return keyscale, confidence


def prepare_unique_inputs(rows: list[dict[str, Any]], input_dir: Path) -> list[str]:
    """Hard-link audio to stable sample_id basenames for collision-free MIR."""
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        source = Path(row["training_audio_path"]).resolve(strict=True)
        link = input_dir / f"{row['sample_id']}.flac"
        if not (link.exists() and link.samefile(source)):
            link.unlink(missing_ok=True)
            os.link(source, link)
        paths.append(str(link))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/training_audio_manifest.jsonl")
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--key-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.part_index < args.num_parts:
        raise SystemExit("part-index must be in [0, num-parts)")

    root = Path(args.project_root).resolve()
    manifest = [r for r in read_jsonl(root / args.manifest) if r.get("quality_status") == "accepted"]
    targets = [
        row for row in sorted(manifest, key=lambda item: item["sample_id"])
        if int(hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(), 16) % args.num_parts
        == args.part_index
    ]
    mir_dir = root / "data" / "mir"
    allin1_dir = mir_dir / "allin1"
    work_dir = mir_dir / f"work_part{args.part_index}"
    input_dir = work_dir / "inputs"
    demix_dir = work_dir / "demix"
    spec_dir = work_dir / "spec"
    state_path = mir_dir / f"mir_part{args.part_index}.jsonl"
    allin1_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for row in targets:
        output = mir_dir / f"{row['sample_id']}.json"
        if output.is_file() and not args.overwrite:
            try:
                if json.loads(output.read_text(encoding="utf-8")).get("analysis_status") == "complete":
                    continue
            except Exception:
                pass
        pending.append(row)

    print(json.dumps({"part": args.part_index, "targets": len(targets), "pending": len(pending)}), flush=True)
    if pending:
        import allin1

        # Separated tracks often share the basename ``instrumental.flac``. allin1
        # keys its intermediate/output files by basename, so passing those paths
        # directly would overwrite results across unrelated catalog records. Give
        # every analysis input the stable, unique sample_id while retaining the
        # original audio bytes through a hard link. A symlink is not sufficient:
        # allin1 resolves symlinks before deriving the basename.
        paths = prepare_unique_inputs(pending, input_dir)
        all_cached = all((allin1_dir / f"{row['sample_id']}.json").is_file() for row in pending)
        results = allin1.analyze(
            paths,
            out_dir=allin1_dir,
            model="harmonix-all",
            device=args.device,
            demix_dir=demix_dir,
            spec_dir=spec_dir,
            # allin1 1.1.0 references an uninitialized demix_paths variable when
            # every result is cached and cleanup is requested. Retain byproducts
            # for this fast cache-only pass; the runbook removes work dirs after
            # final MIR validation.
            keep_byproducts=all_cached,
            overwrite=args.overwrite,
            multiprocess=False,
        )
        if not isinstance(results, list):
            results = [results]
        by_stem = {result.path.stem: result for result in results}
        failures = 0
        state = {r["sample_id"]: r for r in read_jsonl(state_path)}
        for index, row in enumerate(pending, 1):
            sid = row["sample_id"]
            try:
                result = by_stem[sid]
                keyscale, key_confidence = estimate_key(
                    Path(row["training_audio_path"]), args.key_confidence_threshold
                )
                raw_bpm = int(result.bpm)
                bpm, bpm_normalization = normalize_edm_bpm(raw_bpm)
                record = {
                    "sample_id": sid,
                    "record_key": row["record_key"],
                    "training_audio_path": row["training_audio_path"],
                    "analysis_status": "complete",
                    "bpm": bpm,
                    "bpm_raw": raw_bpm,
                    "bpm_normalization": bpm_normalization,
                    "keyscale": keyscale,
                    "key_confidence": round(key_confidence, 6),
                    "timesignature": infer_timesignature(result.beat_positions, result.downbeats),
                    "beats": [round(float(value), 3) for value in result.beats],
                    "downbeats": [round(float(value), 3) for value in result.downbeats],
                    "beat_positions": [int(value) for value in result.beat_positions],
                    "sections": map_sections(result.segments),
                    "allin1_model": "harmonix-all",
                    "key_estimator": "essentia.KeyExtractor(edma)",
                }
                atomic_json(mir_dir / f"{sid}.json", record)
                state[sid] = {"sample_id": sid, "analysis_status": "complete"}
                print(f"[{index}/{len(pending)}] {sid} PASS bpm={bpm} key={keyscale or 'unknown'}", flush=True)
            except Exception as exc:
                failures += 1
                state[sid] = {
                    "sample_id": sid,
                    "analysis_status": "failed",
                    "analysis_error": f"{type(exc).__name__}:{exc}",
                }
                print(f"[{index}/{len(pending)}] {sid} FAILED {state[sid]['analysis_error']}", flush=True)
            atomic_jsonl(state_path, sorted(state.values(), key=lambda item: item["sample_id"]))
        if failures:
            return 1

    completed = sum(
        (mir_dir / f"{row['sample_id']}.json").is_file()
        for row in targets
    )
    print(json.dumps({"part": args.part_index, "complete_files": completed, "targets": len(targets)}))
    return 0 if completed == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())

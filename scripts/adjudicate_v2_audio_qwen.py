#!/usr/bin/env python3
"""Run an identity-blind Qwen2.5-Omni adjudication for disputed V2 audio."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_qwen_local import build_prompt, generate_annotation, load_model
from v2_common import atomic_json, file_sha256, read_jsonl
from verify_v2_audio_claims_moss import build_montage


ADJUDICATION_REVISION = "qwen-identity-blind-full-and-montage-v1"


def blind_prompt(taxonomy: dict[str, Any], schema: dict[str, Any], view: str) -> str:
    """Return the ordinary master-annotation request with all identity and MIR removed."""
    prompt = build_prompt(
        {
            "expected_title": "",
            "expected_artist": "",
            "expected_version": "",
            "audio_source": "",
            "vocal_status_after_processing": "instrumental",
        },
        {},
        taxonomy,
        schema,
    )
    context = (
        "This is the complete track."
        if view == "full"
        else "This is a chronological montage of early, middle and late excerpts."
    )
    return (
        context
        + " Artist, title, catalog family, prior captions and prior instrument claims are withheld. "
        "Determine from the waveform whether the dominant production is electronic, orchestral, or "
        "a hybrid, and name exact sources only when their timbre is clear. "
        + prompt
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3600)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "data_v2" / "manifest.jsonl")
    }
    missing = sorted(set(args.sample_id) - set(rows))
    if missing:
        raise ValueError(f"unknown sample ids: {missing}")
    taxonomy = json.loads((root / "configs" / "taxonomy.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "configs" / "annotation_schema.json").read_text(encoding="utf-8"))
    model, processor = load_model(Path(args.model_path).resolve())
    output_dir = root / "data_v2" / "qwen_audio_adjudication"
    results: list[dict[str, Any]] = []
    for sample_id in sorted(set(args.sample_id)):
        audio = Path(rows[sample_id]["final_audio_path"])
        montage = output_dir / "audio" / f"{sample_id}.overview.wav"
        build_montage(audio, montage)
        views: dict[str, Any] = {}
        for view, source in (("full", audio), ("overview_montage", montage)):
            error = ""
            for attempt in range(1, max(1, args.attempts) + 1):
                try:
                    annotation = generate_annotation(
                        model,
                        processor,
                        source,
                        blind_prompt(taxonomy, schema, "full" if view == "full" else "montage"),
                        args.max_new_tokens,
                    )
                    views[view] = {"attempt": attempt, "annotation": annotation}
                    atomic_json(output_dir / f"{sample_id}.partial.json", {
                        "status": "in_progress",
                        "revision": ADJUDICATION_REVISION,
                        "sample_id": sample_id,
                        "identity_blind": True,
                        "views": views,
                    })
                    print(f"{sample_id} {view} PASS", flush=True)
                    break
                except ValueError as exc:
                    error = f"{type(exc).__name__}:{exc}"
            else:
                views[view] = {"status": "failed", "error": error}
                print(f"{sample_id} {view} FAILED {error}", flush=True)
        record = {
            "status": "pass" if "annotation" in views.get("full", {}) else "failed",
            "revision": ADJUDICATION_REVISION,
            "sample_id": sample_id,
            "audio_sha256": file_sha256(audio),
            "model_id": "Qwen/Qwen2.5-Omni-7B",
            "identity_blind": True,
            "views": views,
            "adjudicated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output_dir / f"{sample_id}.json", record)
        results.append(record)
    report = {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "failed",
        "revision": ADJUDICATION_REVISION,
        "records": len(results),
        "sample_ids": [item["sample_id"] for item in results],
        "results": results,
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

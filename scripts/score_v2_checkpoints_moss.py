#!/usr/bin/env python3
"""Use MOSS-Music as an audio listener for fixed checkpoint samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotate_moss_music import generate, load_runtime
from v2_common import atomic_json, extract_json_object


SCORE_FIELDS = ("prompt_alignment", "melody", "structure", "audio_quality")


def prompt_for(record: dict[str, Any]) -> str:
    """Build a strict audio-judging request grounded in one fixed prompt."""
    return (
        "Listen to the generated instrumental audio and judge it against the requested prompt. "
        "Return only JSON with integer scores from 1 to 5 for prompt_alignment, melody, "
        "structure and audio_quality, plus a short evidence object using those same keys. "
        "Prompt alignment means audible compliance, melody means coherence and memorability, "
        "structure means clear development and section contrast, and audio quality means absence "
        "of clipping, collapse, harsh artifacts or obvious generation failures. Do not reward "
        "artist similarity and do not infer hidden metadata. Return a single JSON object without "
        "Markdown or commentary. Required shape: "
        '{"prompt_alignment": 1, "melody": 1, "structure": 1, "audio_quality": 1, '
        '"evidence": {"prompt_alignment": "...", "melody": "...", "structure": "...", '
        '"audio_quality": "..."}}.\nRequested prompt: ' + record["prompt"]
    )


def parse_score(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate one MOSS listening score object."""
    errors = []
    scores = {}
    for field in SCORE_FIELDS:
        try:
            score = int(value.get(field))
        except (TypeError, ValueError):
            score = 0
        if not 1 <= score <= 5:
            errors.append(f"{field}_outside_1_5")
        scores[field] = score
    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    for field in SCORE_FIELDS:
        detail = str(evidence.get(field) or "").strip()
        if not detail:
            errors.append(f"evidence_{field}_missing")
        evidence[field] = detail
    return {"scores": scores, "evidence": evidence}, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"
    generation = json.loads((evaluation / "generation_report.json").read_text(encoding="utf-8"))
    if generation.get("status") != "pass":
        raise RuntimeError("checkpoint generation gate has not passed")
    model_path = root / "checkpoints" / "MOSS-Music-8B-Thinking"
    model, processor = load_runtime(model_path)
    report_path = evaluation / "listening_scores.json"
    previous = {}
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    cached = {
        (row.get("checkpoint"), row.get("prompt_id")): row
        for row in previous.get("results", [])
        if isinstance(row, dict) and row.get("checkpoint") and row.get("prompt_id")
    }
    results = []
    errors = []
    for index, record in enumerate(generation["results"], 1):
        cache_key = (record["checkpoint"], record["prompt_id"])
        if cache_key in cached:
            results.append(cached[cache_key])
            print(f"[{index}/{len(generation['results'])}] {record['checkpoint']} {record['prompt_id']} CACHED", flush=True)
            continue
        request = prompt_for(record)
        last_error = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            response = generate(model, processor, Path(record["audio_path"]), request, 700)
            try:
                score, validation_errors = parse_score(extract_json_object(response))
                if validation_errors:
                    raise ValueError(",".join(validation_errors))
                result = {
                    "checkpoint": record["checkpoint"],
                    "epoch": record["epoch"],
                    "optimizer_step": record["optimizer_step"],
                    "prompt_id": record["prompt_id"],
                    "audio_path": record["audio_path"],
                    "judge_model": "OpenMOSS-Team/MOSS-Music-8B-Thinking",
                    "judge_model_revision": "2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3",
                    **score,
                }
                results.append(result)
                atomic_json(report_path, {
                    "status": "in_progress",
                    "judge": "MOSS-Music-8B-Thinking audio-grounded fixed-prompt scoring",
                    "score_scale": [1, 5],
                    "dimensions": list(SCORE_FIELDS),
                    "results": results,
                    "errors": [],
                })
                print(f"[{index}/{len(generation['results'])}] {record['checkpoint']} {record['prompt_id']} PASS", flush=True)
                break
            except (ValueError, KeyError, TypeError) as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                request += "\nPrevious response failed validation: " + last_error + ". Return corrected JSON only."
        else:
            errors.append({
                "checkpoint": record["checkpoint"],
                "prompt_id": record["prompt_id"],
                "reason": last_error,
            })
    report = {
        "status": "pass" if not errors and len(results) == len(generation["results"]) else "failed",
        "judge": "MOSS-Music-8B-Thinking audio-grounded fixed-prompt scoring",
        "score_scale": [1, 5],
        "dimensions": list(SCORE_FIELDS),
        "results": results,
        "errors": errors,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Use MOSS-Music as an audio listener for fixed checkpoint samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotate_moss_music import generate, load_runtime
from v2_common import atomic_json, extract_json_object, file_sha256, object_sha256


SCORE_FIELDS = ("prompt_alignment", "melody", "structure", "audio_quality")
SCORER_REVISION = "fixed-prompt-audio-judge-v2.3"


def prompt_for(record: dict[str, Any]) -> str:
    """Build a strict audio-judging request grounded in one fixed prompt."""
    return (
        "Listen to the generated instrumental audio and judge it against the requested prompt. "
        "Return only JSON with integer scores from 1 to 5 for prompt_alignment, melody, "
        "structure and audio_quality, plus a short evidence object using those same keys. "
        "Score the four dimensions independently. Prompt alignment alone measures compliance "
        "with the requested genre and audible details. Melody measures internal coherence and "
        "memorability even if the melody is a different style than requested. Structure measures "
        "internal development and section contrast even if the exact requested section labels are "
        "absent. Audio quality measures only technical cleanliness: clipping, collapse, harsh "
        "artifacts, noise, muddiness and obvious generation failure. A clean professional mix must "
        "not receive a low audio_quality score merely because the style or instruments mismatch. "
        "Likewise, a coherent melody must not receive a low melody score merely for style mismatch. "
        "Also return explicit boolean failure_modes for distorted, collapsed, static_loop and "
        "intelligible_vocals; mark a flag true whenever that failure is audible. Intelligible "
        "singing, rap or spoken words in any language count as intelligible_vocals. Non-lexical "
        "vocal chops used only as an instrument do not count. Do not reward "
        "artist similarity and do not infer hidden metadata. Return a single JSON object without "
        "Markdown or commentary. Required shape: "
        '{"prompt_alignment": 1, "melody": 1, "structure": 1, "audio_quality": 1, '
        '"evidence": {"prompt_alignment": "...", "melody": "...", "structure": "...", '
        '"audio_quality": "..."}, "failure_modes": {"distorted": false, '
        '"collapsed": false, "static_loop": false, "intelligible_vocals": false}}.\nRequested prompt: ' + record["prompt"]
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
    quality_evidence = evidence.get("audio_quality", "").lower()
    positive_quality = (
        "clean", "professional", "well-mixed", "well mixed", "no clipping", "no artifacts",
    )
    if scores["audio_quality"] <= 2 and any(term in quality_evidence for term in positive_quality):
        errors.append("audio_quality_score_contradicts_positive_evidence")
    melody_evidence = evidence.get("melody", "").lower()
    if (
        scores["melody"] <= 2
        and "coherent" in melody_evidence
        and "incoherent" not in melody_evidence
        and "not coherent" not in melody_evidence
    ):
        errors.append("melody_score_contradicts_coherent_evidence")
    raw_failures = value.get("failure_modes")
    if not isinstance(raw_failures, dict):
        raw_failures = {}
        errors.append("failure_modes_missing")
    failure_modes: dict[str, bool] = {}
    for name in ("distorted", "collapsed", "static_loop", "intelligible_vocals"):
        observed = raw_failures.get(name)
        if not isinstance(observed, bool):
            errors.append(f"failure_mode_not_boolean:{name}")
            observed = False
        failure_modes[name] = observed
    structure_evidence = evidence.get("structure", "").lower()
    explicit_loop_phrases = (
        "single looped section",
        "static, unchanging loop",
        "continuous, unchanging loop",
        "unchanging loop",
        "static loop",
    )
    if any(phrase in structure_evidence for phrase in explicit_loop_phrases):
        failure_modes["static_loop"] = True
    return {"scores": scores, "evidence": evidence, "failure_modes": failure_modes}, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--evaluation-dir",
        default="outputs/v2/checkpoint-evaluation",
        help="Directory containing generation_report.json; relative to project root by default.",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    evaluation = Path(args.evaluation_dir)
    if not evaluation.is_absolute():
        evaluation = root / evaluation
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
        (row.get("checkpoint"), row.get("prompt_id"), row.get("lora_scale")): row
        for row in previous.get("results", [])
        if isinstance(row, dict) and row.get("checkpoint") and row.get("prompt_id")
    }
    results = []
    errors = []
    for index, record in enumerate(generation["results"], 1):
        cache_key = (record["checkpoint"], record["prompt_id"], record.get("lora_scale"))
        audio_sha256 = file_sha256(Path(record["audio_path"]))
        prompt_sha256 = object_sha256({
            "prompt": record["prompt"],
            "scorer_revision": SCORER_REVISION,
        })
        cached_row = cached.get(cache_key)
        if (
            cached_row
            and cached_row.get("scorer_revision") == SCORER_REVISION
            and cached_row.get("audio_sha256") == audio_sha256
            and cached_row.get("prompt_sha256") == prompt_sha256
        ):
            results.append(cached_row)
            print(f"[{index}/{len(generation['results'])}] {record['checkpoint']} {record['prompt_id']} CACHED", flush=True)
            continue
        request = prompt_for(record)
        last_error = ""
        last_response = ""
        for attempt in range(1, max(1, args.attempts) + 1):
            response = generate(model, processor, Path(record["audio_path"]), request, 1200)
            last_response = response
            try:
                score, validation_errors = parse_score(extract_json_object(response))
                if validation_errors:
                    raise ValueError(",".join(validation_errors))
                result = {
                    "checkpoint": record["checkpoint"],
                    "epoch": record.get("epoch"),
                    "optimizer_step": record.get("optimizer_step"),
                    "prompt_id": record["prompt_id"],
                    "source_prompt_id": record.get("source_prompt_id", record["prompt_id"]),
                    "seed": record.get("seed"),
                    "duration": record.get("duration"),
                    "lora_scale": record.get("lora_scale"),
                    "audio_path": record["audio_path"],
                    "audio_sha256": audio_sha256,
                    "prompt_sha256": prompt_sha256,
                    "scorer_revision": SCORER_REVISION,
                    "judge_model": "OpenMOSS-Team/MOSS-Music-8B-Thinking",
                    "judge_model_revision": "2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3",
                    **score,
                }
                results.append(result)
                atomic_json(report_path, {
                    "status": "in_progress",
                    "judge": "MOSS-Music-8B-Thinking audio-grounded fixed-prompt scoring",
                    "scorer_revision": SCORER_REVISION,
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
                "raw_response_tail": last_response[-2000:],
            })
    report = {
        "status": "pass" if not errors and len(results) == len(generation["results"]) else "failed",
        "judge": "MOSS-Music-8B-Thinking audio-grounded fixed-prompt scoring",
        "scorer_revision": SCORER_REVISION,
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

#!/usr/bin/env python3
"""Select v2 best step from validation, MOSS listening and audio features."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from v2_common import atomic_json, read_jsonl
from v2_listening_quality import (
    MINIMUM_DIMENSION_MEAN,
    MINIMUM_INDIVIDUAL_BY_DIMENSION,
    MINIMUM_INDIVIDUAL_SCORE,
    summarize_quality,
)


def feature(path: Path) -> np.ndarray:
    """Extract a compact timbre/rhythm/harmony vector for diversity checks."""
    audio, sample_rate = librosa.load(path, sr=22050, mono=True, duration=60)
    if audio.size == 0:
        raise ValueError(f"empty audio: {path}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    spectral = np.vstack([
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
        librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
        librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
        librosa.feature.rms(y=audio),
        librosa.feature.zero_crossing_rate(audio),
    ])
    vector = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        spectral.mean(axis=1), spectral.std(axis=1),
    ]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity for normalized vectors."""
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def scale(
    values: dict[str, float],
    higher_is_better: bool,
    *,
    minimum_range: float = 1e-12,
) -> dict[str, float]:
    """Min-max scale candidate values to 0..1."""
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum - minimum < minimum_range:
        return {key: 1.0 for key in values}
    output = {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}
    return output if higher_is_better else {key: 1.0 - value for key, value in output.items()}


def copy_best(source: Path, destination: Path) -> None:
    """Copy the selected flat adapter into the public release layout."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing selection {destination}")
    shutil.copytree(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    evaluation = root / "outputs" / "v2" / "checkpoint-evaluation"
    generation = json.loads((evaluation / "generation_report.json").read_text(encoding="utf-8"))
    listening = json.loads((evaluation / "listening_scores.json").read_text(encoding="utf-8"))
    if generation.get("status") != "pass" or listening.get("status") != "pass":
        raise RuntimeError("generation and listening score gates must pass")

    training_rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    training_features = np.stack([feature(Path(row["final_audio_path"])) for row in training_rows])
    generated_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in generation["results"]:
        generated_by_checkpoint[record["checkpoint"]].append(record)
    listening_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in listening["results"]:
        listening_by_checkpoint[record["checkpoint"]].append(record)

    metrics_path = root / "outputs" / "v2" / "train-validation" / "metrics_history.jsonl"
    validation_by_epoch = {
        int(record["epoch"]): float(record["loss"])
        for record in read_jsonl(metrics_path)
        if record.get("event") == "validation"
    }
    raw: dict[str, dict[str, Any]] = {}
    for label, records in generated_by_checkpoint.items():
        vectors = [feature(Path(record["audio_path"])) for record in records]
        diversity = np.mean([1.0 - cosine(vectors[a], vectors[b]) for a, b in combinations(range(len(vectors)), 2)])
        nearest = max(float(np.max(training_features @ vector)) for vector in vectors)
        judge_records = listening_by_checkpoint[label]
        quality = summarize_quality(judge_records)
        judge_score = float(np.mean([
            np.mean(list(record["scores"].values())) / 5.0
            for record in judge_records
        ]))
        epoch = int(records[0]["epoch"])
        if epoch not in validation_by_epoch:
            nearest_epoch = min(validation_by_epoch, key=lambda value: abs(value - epoch))
            val_loss = validation_by_epoch[nearest_epoch]
        else:
            val_loss = validation_by_epoch[epoch]
        raw[label] = {
            "epoch": epoch,
            "optimizer_step": int(records[0]["optimizer_step"]),
            "adapter_path": records[0]["adapter_path"],
            "checkpoint_base": records[0].get("checkpoint_base", label),
            "lora_scale": float(records[0].get("lora_scale", 1.0)),
            "validation_loss": val_loss,
            "moss_listening_score": judge_score,
            "absolute_listening_quality": quality,
            "prompt_output_diversity": float(diversity),
            "max_training_feature_similarity": nearest,
        }
    eligible = {
        key: value for key, value in raw.items()
        if value["absolute_listening_quality"]["quality_accepted"]
    }
    if not eligible:
        report = {
            "status": "failed",
            "quality_accepted": False,
            "selected_checkpoint": None,
            "best_optimizer_step": None,
            "selected_epoch": None,
            "selection_method": {
                "absolute_listening_gate_required": True,
                "minimum_dimension_mean": MINIMUM_DIMENSION_MEAN,
                "minimum_individual_score": MINIMUM_INDIVIDUAL_SCORE,
                "minimum_individual_by_dimension": MINIMUM_INDIVIDUAL_BY_DIMENSION,
                "human_listening_completed": False,
                "listening_proxy": "OpenMOSS-Team/MOSS-Music-8B-Thinking",
            },
            "candidates": raw,
            "best_val_output": None,
            "errors": ["no_checkpoint_passed_absolute_listening_quality"],
        }
        atomic_json(evaluation / "selection.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    val_scaled = scale({key: value["validation_loss"] for key, value in eligible.items()}, False)
    diversity_scaled = scale(
        {key: value["prompt_output_diversity"] for key, value in eligible.items()},
        True,
        minimum_range=0.01,
    )
    for label, value in eligible.items():
        no_memorization = max(0.0, 1.0 - value["max_training_feature_similarity"])
        value["composite_score"] = (
            0.40 * val_scaled[label]
            + 0.45 * value["moss_listening_score"]
            + 0.10 * diversity_scaled[label]
            + 0.05 * no_memorization
        )
    selected_label = max(eligible, key=lambda label: eligible[label]["composite_score"])
    selected = eligible[selected_label]
    copy_best(Path(selected["adapter_path"]), root / "outputs" / "v2" / "best-val")
    report = {
        "status": "pass",
        "quality_accepted": True,
        "selected_checkpoint": selected_label,
        "best_optimizer_step": selected["optimizer_step"],
        "selected_epoch": selected["epoch"],
        "selected_lora_scale": selected["lora_scale"],
        "selection_method": {
            "validation_loss_weight": 0.40,
            "moss_audio_listening_weight": 0.45,
            "fixed_prompt_diversity_weight": 0.10,
            "diversity_minimum_meaningful_range": 0.01,
            "training_similarity_penalty_weight": 0.05,
            "minimum_dimension_mean": MINIMUM_DIMENSION_MEAN,
            "minimum_individual_by_dimension": MINIMUM_INDIVIDUAL_BY_DIMENSION,
            "human_listening_completed": False,
            "listening_proxy": "OpenMOSS-Team/MOSS-Music-8B-Thinking"
        },
        "candidates": raw,
        "best_val_output": str(root / "outputs" / "v2" / "best-val"),
    }
    atomic_json(evaluation / "selection.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

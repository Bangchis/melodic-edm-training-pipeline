#!/usr/bin/env python3
"""Validate grouped coverage and exactly three cached prompt embeddings."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from v2_common import CAPTION_TYPES, atomic_json, read_jsonl


def tensor_errors(path: Path) -> list[str]:
    """Return structural errors for one preprocessed tensor record."""
    errors: list[str] = []
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return [f"load_failed:{type(exc).__name__}:{exc}"]
    for key in ("target_latents", "attention_mask", "context_latents"):
        value = data.get(key)
        if not torch.is_tensor(value) or value.numel() == 0:
            errors.append(f"{key}_missing_or_empty")
        elif not bool(torch.isfinite(value).all()):
            errors.append(f"{key}_nonfinite")
    states = data.get("encoder_hidden_states_variants")
    masks = data.get("encoder_attention_mask_variants")
    if not isinstance(states, list) or len(states) != 3:
        errors.append(f"prompt_state_count:{len(states) if isinstance(states, list) else 'invalid'}")
        states = []
    if not isinstance(masks, list) or len(masks) != 3:
        errors.append(f"prompt_mask_count:{len(masks) if isinstance(masks, list) else 'invalid'}")
        masks = []
    for index, value in enumerate(states):
        if not torch.is_tensor(value) or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            errors.append(f"prompt_state_{index}_invalid")
    for index, value in enumerate(masks):
        if not torch.is_tensor(value) or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            errors.append(f"prompt_mask_{index}_invalid")
    if states and not torch.equal(data.get("encoder_hidden_states"), states[0]):
        errors.append("canonical_state_is_not_legacy_index_0")
    if masks and not torch.equal(data.get("encoder_attention_mask"), masks[0]):
        errors.append("canonical_mask_is_not_legacy_index_0")
    metadata = data.get("metadata", {})
    if metadata.get("caption_variants") is None or len(metadata.get("caption_variants", [])) != 3:
        errors.append("metadata_caption_variant_count")
    return errors


def names(path: Path) -> set[str]:
    """Return sample IDs declared by a tensor directory manifest."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return {Path(value).stem for value in manifest.get("samples", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    rows = read_jsonl(root / "data_v2" / "manifest.jsonl")
    expected_train = {row["sample_id"] for row in rows if row["split"] == "train"}
    expected_validation = {row["sample_id"] for row in rows if row["split"] == "validation"}
    expected_all = expected_train | expected_validation
    train_dir = root / "data_v2" / "tensors_train"
    validation_dir = root / "data_v2" / "tensors_validation"
    all_dir = root / "data_v2" / "tensors_all"
    errors: list[dict[str, Any]] = []
    for label, directory, expected in (
        ("train", train_dir, expected_train),
        ("validation", validation_dir, expected_validation),
        ("all", all_dir, expected_all),
    ):
        observed = names(directory)
        if observed != expected:
            errors.append({
                "sample_id": "*",
                "reason": f"{label}_coverage_mismatch",
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            })
    parent_splits: dict[str, set[str]] = {}
    for row in rows:
        parent_splits.setdefault(row["parent_song_id"], set()).add(row["split"])
    for parent, splits in parent_splits.items():
        if len(splits) != 1:
            errors.append({"sample_id": "*", "reason": f"parent_split_crossing:{parent}"})

    for index, sample_id in enumerate(sorted(expected_all), 1):
        path = all_dir / f"{sample_id}.pt"
        for reason in tensor_errors(path):
            errors.append({"sample_id": sample_id, "reason": reason})
        if index % 20 == 0 or index == len(expected_all):
            print(f"validated {index}/{len(expected_all)}", flush=True)
    report = {
        "status": "pass" if not errors else "failed",
        "records": len(rows),
        "train_tensors": len(expected_train),
        "validation_tensors": len(expected_validation),
        "all_tensors": len(expected_all),
        "parent_groups": len(parent_splits),
        "parent_split_crossings": sum(len(value) > 1 for value in parent_splits.values()),
        "audio_latents_per_record": 1,
        "prompt_embeddings_per_record": 3,
        "caption_variant_types": list(CAPTION_TYPES),
        "validation_caption_index": 0,
        "validation_cfg_dropout": 0.0,
        "deduplication_performed": False,
        "errors": errors,
    }
    atomic_json(root / "data_v2" / "tensor_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

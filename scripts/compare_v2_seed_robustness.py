#!/usr/bin/env python3
"""Pair XL-Base and LoRA scores by prompt/seed to localize regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_v2_seed_robustness import SCORE_FIELDS, seed_passes
from v2_common import atomic_json


def key(record: dict[str, Any]) -> tuple[str, int]:
    return (
        str(record.get("source_prompt_id") or record.get("prompt_id") or ""),
        int(record.get("seed", -1)),
    )


def compare(
    lora_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    *,
    minimum_score: int = 3,
) -> dict[str, Any]:
    lora = {key(record): record for record in lora_records}
    base = {key(record): record for record in base_records}
    errors: list[str] = []
    if set(lora) != set(base):
        errors.append("paired_prompt_seed_sets_differ")
    pairs = []
    buckets = {"both_pass": 0, "lora_only_pass": 0, "base_only_pass": 0, "both_fail": 0}
    deltas = {field: [] for field in SCORE_FIELDS}
    for pair_key in sorted(set(lora) & set(base)):
        lora_row = lora[pair_key]
        base_row = base[pair_key]
        lora_pass = seed_passes(lora_row, minimum_score)
        base_pass = seed_passes(base_row, minimum_score)
        if lora_pass and base_pass:
            bucket = "both_pass"
        elif lora_pass:
            bucket = "lora_only_pass"
        elif base_pass:
            bucket = "base_only_pass"
        else:
            bucket = "both_fail"
        buckets[bucket] += 1
        score_delta = {}
        for field in SCORE_FIELDS:
            value = int(lora_row["scores"][field]) - int(base_row["scores"][field])
            deltas[field].append(value)
            score_delta[field] = value
        pairs.append({
            "source_prompt_id": pair_key[0],
            "seed": pair_key[1],
            "bucket": bucket,
            "lora_pass": lora_pass,
            "base_pass": base_pass,
            "score_delta_lora_minus_base": score_delta,
        })
    mean_deltas = {
        field: sum(values) / len(values) if values else 0.0 for field, values in deltas.items()
    }
    return {
        "status": "pass" if not errors and pairs else "failed",
        "minimum_score_per_dimension": minimum_score,
        "paired_records": len(pairs),
        "outcomes": buckets,
        "mean_score_delta_lora_minus_base": mean_deltas,
        "pairs": pairs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--lora-report", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-score", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    lora_path = resolve(args.lora_report)
    base_path = resolve(args.base_report)
    output = resolve(args.output)
    lora = json.loads(lora_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if lora.get("status") != "pass" or base.get("status") != "pass":
        raise RuntimeError("both listening score reports must pass technical validation")
    result = compare(
        lora.get("results", []),
        base.get("results", []),
        minimum_score=args.minimum_score,
    )
    result["lora_report"] = str(lora_path.relative_to(root))
    result["base_report"] = str(base_path.relative_to(root))
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

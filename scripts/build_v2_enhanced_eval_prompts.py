#!/usr/bin/env python3
"""Enhance fixed evaluation prompts without weakening their required conditions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from enhance_prompt_openrouter import enhance_prompt
from v2_common import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="outputs/v2/enhanced-prompts/enhanced_eval_prompts.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    fixed = json.loads(
        (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (root / "configs" / "v2" / "eval_prompt_enhancement.json").read_text(encoding="utf-8")
    )
    token = os.environ.get("OPENROUTER_API_KEY", "")
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    enhanced_prompts = []
    lineage = []
    for prompt in fixed["prompts"]:
        prompt_policy = policy["prompts"][prompt["id"]]
        result = enhance_prompt(
            prompt["caption"],
            token,
            explicit_conditions={
                "bpm": prompt["bpm"],
                "keyscale": prompt["keyscale"],
                "timesignature": prompt["timesignature"],
                "required_terms": prompt_policy["required_terms"],
            },
        )
        conditions = result["conditions"]
        enhanced_prompts.append({
            **prompt,
            "caption": conditions["caption"],
            "duration": float(policy["duration"]),
            "seed": int(prompt["seed"]) + int(prompt_policy["seed_offset"]),
        })
        lineage.append({
            "id": prompt["id"],
            "source_caption": prompt["caption"],
            "required_terms": conditions["required_terms"],
            "word_count": conditions["word_count"],
            "requested_model": result["requested_model"],
            "resolved_model": result["resolved_model"],
            "usage": result["usage"],
        })
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_json(output, {
        "status": "pass",
        "schema_version": "2.0",
        "prompt_count": len(enhanced_prompts),
        "prompts": enhanced_prompts,
        "lineage": lineage,
    })
    print(json.dumps({
        "status": "pass",
        "output": str(output),
        "prompt_count": len(enhanced_prompts),
        "word_counts": {row["id"]: row["word_count"] for row in lineage},
        "resolved_models": sorted({row["resolved_model"] for row in lineage}),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

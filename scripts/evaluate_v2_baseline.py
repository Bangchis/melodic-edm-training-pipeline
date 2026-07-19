#!/usr/bin/env python3
"""Generate fixed prompts with pristine XL-Base and no LoRA as a hard baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_v2_checkpoints import DEFAULT_LYRICS, audio_path, probe
from v2_common import atomic_json, object_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    prompt_document = json.loads(
        (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
    )
    prompts = prompt_document["prompts"]

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    message, loaded = handler.initialize_service(
        project_root=str(root),
        config_path="acestep-v15-xl-base",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not loaded:
        raise RuntimeError(f"XL-Base initialization failed: {message}")
    output = root / "outputs" / "v2" / "baseline-xl-base"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for prompt in prompts:
        target = output / "audio" / prompt["id"]
        params = GenerationParams(
            caption=prompt["caption"],
            lyrics=prompt.get("lyrics", DEFAULT_LYRICS),
            instrumental=True,
            bpm=int(prompt["bpm"]),
            keyscale=prompt["keyscale"],
            timesignature=str(prompt["timesignature"]),
            duration=float(prompt["duration"]),
            inference_steps=64,
            guidance_scale=8.0,
            shift=1.0,
            use_adg=True,
            dcw_enabled=False,
            seed=int(prompt["seed"]),
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_cot_lyrics=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[int(prompt["seed"])],
            audio_format="wav",
        )
        generated = generate_music(handler, None, params, config, save_dir=str(target))
        if not generated.success or len(generated.audios) != 1:
            errors.append({
                "prompt_id": prompt["id"],
                "reason": generated.error or generated.status_message,
            })
            continue
        path = audio_path(generated.audios[0])
        if path is None or not path.is_file():
            errors.append({"prompt_id": prompt["id"], "reason": "generated_audio_path_missing"})
            continue
        valid, audio_probe = probe(path)
        results.append({
            "checkpoint": "base_xl_no_lora",
            "epoch": 0,
            "optimizer_step": 0,
            "prompt_id": prompt["id"],
            "source_prompt_id": prompt["id"],
            "prompt": prompt["caption"],
            "lyrics": prompt.get("lyrics", DEFAULT_LYRICS),
            "duration": float(prompt["duration"]),
            "seed": prompt["seed"],
            "audio_path": str(path),
            "probe": audio_probe,
            "status": "pass" if valid else "failed",
        })
        if not valid:
            errors.append({"prompt_id": prompt["id"], "reason": "audio_validation_failed"})
        print(f"[base-xl] {prompt['id']} {'PASS' if valid else 'FAIL'}", flush=True)
    report = {
        "status": "pass" if not errors and len(results) == len(prompts) else "failed",
        "adapter": "none",
        "base_model": "ACE-Step 1.5 XL-Base",
        "prompts_file": "configs/v2/fixed_eval_prompts.json",
        "prompt_document_sha256": object_sha256(prompt_document),
        "fixed_prompt_count": len(prompts),
        "generated_outputs": len(results),
        "sampling": {
            "inference_steps": 64,
            "guidance_scale": 8.0,
            "shift": 1.0,
            "use_adg": True,
            "dcw_enabled": False,
        },
        "results": results,
        "errors": errors,
    }
    atomic_json(output / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

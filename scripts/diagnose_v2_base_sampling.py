#!/usr/bin/env python3
"""A/B pristine XL-Base sampling profiles before any new LoRA training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_v2_checkpoints import LYRICS, audio_path, probe
from v2_common import atomic_json


PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "current_shift1_dcw",
        "inference_steps": 50,
        "guidance_scale": 7.0,
        "shift": 1.0,
        "use_adg": False,
        "dcw_enabled": True,
    },
    {
        "name": "current_shift1_no_dcw",
        "inference_steps": 50,
        "guidance_scale": 7.0,
        "shift": 1.0,
        "use_adg": False,
        "dcw_enabled": False,
    },
    {
        "name": "hq_shift3_adg_dcw",
        "inference_steps": 64,
        "guidance_scale": 8.0,
        "shift": 3.0,
        "use_adg": True,
        "dcw_enabled": True,
    },
    {
        "name": "hq_shift3_adg_no_dcw",
        "inference_steps": 64,
        "guidance_scale": 8.0,
        "shift": 3.0,
        "use_adg": True,
        "dcw_enabled": False,
    },
    {
        "name": "hq_shift1_adg_no_dcw",
        "inference_steps": 64,
        "guidance_scale": 8.0,
        "shift": 1.0,
        "use_adg": True,
        "dcw_enabled": False,
    },
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    prompts = json.loads(
        (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
    )["prompts"]
    prompt = next(item for item in prompts if item["id"] == "gaming_progressive")

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
    output = root / "outputs" / "v2" / "base-sampling-diagnostic"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for profile in PROFILES:
        params = GenerationParams(
            caption=prompt["caption"],
            lyrics=LYRICS,
            instrumental=True,
            bpm=int(prompt["bpm"]),
            keyscale=prompt["keyscale"],
            timesignature=str(prompt["timesignature"]),
            duration=float(prompt["duration"]),
            seed=int(prompt["seed"]),
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_cot_lyrics=False,
            **{key: value for key, value in profile.items() if key != "name"},
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[int(prompt["seed"])],
            audio_format="wav",
        )
        target = output / "audio" / profile["name"]
        generated = generate_music(handler, None, params, config, save_dir=str(target))
        if not generated.success or len(generated.audios) != 1:
            errors.append({"profile": profile["name"], "reason": generated.error or generated.status_message})
            continue
        path = audio_path(generated.audios[0])
        if path is None or not path.is_file():
            errors.append({"profile": profile["name"], "reason": "generated_audio_path_missing"})
            continue
        valid, audio_probe = probe(path)
        results.append({
            "checkpoint": profile["name"],
            "epoch": 0,
            "optimizer_step": 0,
            "prompt_id": prompt["id"],
            "prompt": prompt["caption"],
            "seed": prompt["seed"],
            "sampling": profile,
            "audio_path": str(path),
            "probe": audio_probe,
            "status": "pass" if valid else "failed",
        })
        if not valid:
            errors.append({"profile": profile["name"], "reason": "audio_validation_failed"})
        print(f"[{profile['name']}] {'PASS' if valid else 'FAIL'}", flush=True)
    report = {
        "status": "pass" if not errors and len(results) == len(PROFILES) else "failed",
        "base_model": "ACE-Step 1.5 XL-Base",
        "adapter": "none",
        "prompt_id": prompt["id"],
        "profiles": len(PROFILES),
        "results": results,
        "errors": errors,
    }
    atomic_json(output / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

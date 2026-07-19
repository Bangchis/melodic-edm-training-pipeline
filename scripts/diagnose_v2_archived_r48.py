#!/usr/bin/env python3
"""Re-evaluate the archived rank-48 adapter with the corrected XL-Base sampler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_v2_checkpoints import LYRICS, audio_path, probe, resolve_adapter
from v2_common import atomic_json


SCALES = (0.25, 0.5, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--adapter",
        default="outputs/v2-r48-failed-20260718T0841Z/train-validation/checkpoints/best_val",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    adapter_source = Path(args.adapter)
    if not adapter_source.is_absolute():
        adapter_source = root / adapter_source
    adapter = resolve_adapter(adapter_source)
    prompt = next(
        item for item in json.loads(
            (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
        )["prompts"]
        if item["id"] == "gaming_progressive"
    )

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
    adapter_name = "archived_r48_best_val_epoch45"
    load_message = handler.add_lora(str(adapter), adapter_name=adapter_name)
    if not load_message.startswith("✅"):
        raise RuntimeError(load_message)
    active_message = handler.set_active_lora_adapter(adapter_name)
    if not active_message.startswith("✅"):
        raise RuntimeError(active_message)

    output = root / "outputs" / "v2" / "diagnostic-r48-corrected-sampler"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scale in SCALES:
        scale_message = handler.set_lora_scale(adapter_name, scale)
        if not scale_message.startswith("✅"):
            raise RuntimeError(scale_message)
        label = f"r48_epoch45_scale_{scale:.2f}"
        params = GenerationParams(
            caption=prompt["caption"],
            lyrics=LYRICS,
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
        generated = generate_music(
            handler, None, params, config, save_dir=str(output / "audio" / label)
        )
        if not generated.success or len(generated.audios) != 1:
            errors.append({"checkpoint": label, "reason": generated.error or generated.status_message})
            continue
        path = audio_path(generated.audios[0])
        if path is None or not path.is_file():
            errors.append({"checkpoint": label, "reason": "generated_audio_path_missing"})
            continue
        valid, audio_probe = probe(path)
        results.append({
            "checkpoint": label,
            "epoch": 45,
            "optimizer_step": 585,
            "prompt_id": prompt["id"],
            "prompt": prompt["caption"],
            "seed": prompt["seed"],
            "lora_rank": 48,
            "lora_alpha": 96,
            "lora_scale": scale,
            "audio_path": str(path),
            "probe": audio_probe,
            "status": "pass" if valid else "failed",
        })
        if not valid:
            errors.append({"checkpoint": label, "reason": "audio_validation_failed"})
        print(f"[{label}] {'PASS' if valid else 'FAIL'}", flush=True)
    report = {
        "status": "pass" if not errors and len(results) == len(SCALES) else "failed",
        "purpose": "separate rank-48 training effect from the previously invalid sampler",
        "base_model": "ACE-Step 1.5 XL-Base",
        "adapter": str(adapter),
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

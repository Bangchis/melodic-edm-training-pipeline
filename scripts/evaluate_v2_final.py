#!/usr/bin/env python3
"""Generate the three fixed prompts with the fresh all-data V2 adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_v2_checkpoints import LYRICS, audio_path, probe, resolve_adapter
from v2_common import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    final_root = root / "outputs" / "v2" / "final-all-data"
    gate = json.loads((final_root / "final_validation_report.json").read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("final all-data validation gate has not passed")
    prompts = json.loads(
        (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
    )["prompts"]

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
    adapter = resolve_adapter(final_root / "final")
    load_message = handler.add_lora(str(adapter), adapter_name="final_all_data")
    if not load_message.startswith("✅"):
        raise RuntimeError(load_message)
    active_message = handler.set_active_lora_adapter("final_all_data")
    if not active_message.startswith("✅"):
        raise RuntimeError(active_message)

    output = final_root / "evaluation"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for prompt in prompts:
        target = output / "audio" / prompt["id"]
        params = GenerationParams(
            caption=prompt["caption"],
            lyrics=LYRICS,
            instrumental=True,
            bpm=int(prompt["bpm"]),
            keyscale=prompt["keyscale"],
            timesignature=str(prompt["timesignature"]),
            duration=float(prompt["duration"]),
            inference_steps=50,
            guidance_scale=7.0,
            shift=1.0,
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
            "prompt_id": prompt["id"],
            "seed": prompt["seed"],
            "audio_path": str(path),
            "probe": audio_probe,
            "status": "pass" if valid else "failed",
        })
        if not valid:
            errors.append({"prompt_id": prompt["id"], "reason": "audio_validation_failed"})
        print(f"[final-all-data] {prompt['id']} {'PASS' if valid else 'FAIL'}", flush=True)
    report = {
        "status": "pass" if not errors and len(results) == len(prompts) else "failed",
        "adapter": "final-all-data",
        "fixed_prompt_count": len(prompts),
        "generated_outputs": len(results),
        "results": results,
        "errors": errors,
    }
    atomic_json(output / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

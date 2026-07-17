#!/usr/bin/env python3
"""Generate identical fixed prompts from middle, best-val, and final adapters."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


LYRICS = """[Intro]
[Instrumental]

[Theme]
[Instrumental]

[Build]
[Instrumental]

[Drop]
[Instrumental]

[Break]
[Instrumental]

[Final Drop]
[Instrumental]

[Outro]
[Instrumental]
"""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def probe(path: Path) -> tuple[bool, dict[str, Any]]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=120,
    )
    if result.returncode:
        return False, {"error": (result.stderr or result.stdout)[-500:]}
    data = json.loads(result.stdout)
    stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration") or 0)
    valid = int(stream.get("sample_rate") or 0) == 48000 and int(stream.get("channels") or 0) == 2 and duration >= 10
    return valid, {"duration": duration, "sample_rate": stream.get("sample_rate"), "channels": stream.get("channels")}


def output_audio_path(audio: dict[str, Any]) -> Path | None:
    for key in ("path", "file", "audio_path"):
        value = audio.get(key)
        if value and not str(value).startswith("/v1/audio?"):
            return Path(str(value))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--prompts", default="configs/inference_prompts.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    train_output = root / "outputs" / "training" / "melodic-edm-core-v1"
    gate_path = train_output / "training_validation_report.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    if gate.get("status") != "pass":
        raise SystemExit("main training validation gate has not passed")
    adapters = gate["selected_checkpoints"]
    prompts = json.loads((root / args.prompts).read_text(encoding="utf-8"))

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    message, loaded = handler.initialize_service(
        project_root=str(root), config_path="acestep-v15-xl-base", device="cuda",
        use_flash_attention=False, compile_model=False, offload_to_cpu=False,
    )
    if not loaded:
        raise RuntimeError(f"XL-Base initialization failed: {message}")

    output_root = root / "outputs" / "inference" / "checkpoint_comparison"
    results = []
    errors = []
    for label in ("middle", "best_val", "last"):
        load_message = handler.add_lora(adapters[label], adapter_name=label)
        if not load_message.startswith("✅"):
            raise RuntimeError(load_message)
        active_message = handler.set_active_lora_adapter(label)
        if not active_message.startswith("✅"):
            raise RuntimeError(active_message)
        for prompt in prompts:
            target_dir = output_root / label / prompt["id"]
            params = GenerationParams(
                caption=prompt["caption"], lyrics=LYRICS, instrumental=True,
                bpm=int(prompt["bpm"]), keyscale=prompt["keyscale"],
                timesignature=str(prompt["timesignature"]), duration=float(prompt["duration"]),
                inference_steps=50, guidance_scale=7.0, shift=1.0, seed=int(prompt["seed"]),
                thinking=False, use_cot_metas=False, use_cot_caption=False,
                use_cot_language=False, use_cot_lyrics=False,
            )
            config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(prompt["seed"])], audio_format="wav")
            generated = generate_music(handler, None, params, config, save_dir=str(target_dir))
            if not generated.success or len(generated.audios) != 1:
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": generated.error or generated.status_message})
                continue
            audio_path = output_audio_path(generated.audios[0])
            if audio_path is None or not audio_path.is_file():
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": "generated_audio_path_missing"})
                continue
            valid, audio_probe = probe(audio_path)
            record = {
                "checkpoint": label, "adapter_path": adapters[label], "prompt_id": prompt["id"],
                "seed": prompt["seed"], "audio_path": str(audio_path), "probe": audio_probe,
                "status": "pass" if valid else "failed",
            }
            results.append(record)
            if not valid:
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": "audio_validation_failed", "probe": audio_probe})
            print(f"[{label}] {prompt['id']} {'PASS' if valid else 'FAIL'} {audio_path}", flush=True)

    report = {
        "status": "pass" if not errors and len(results) == 9 else "failed",
        "fixed_prompt_count": len(prompts), "checkpoint_count": 3,
        "generated_outputs": len(results), "results": results, "errors": errors,
    }
    atomic_json(output_root / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

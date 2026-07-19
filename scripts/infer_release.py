#!/usr/bin/env python3
"""Run deterministic inference from the packaged Melodic EDM LoRA release."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def audio_path(audio: dict[str, Any]) -> Path | None:
    for key in ("path", "file", "audio_path"):
        value = audio.get(key)
        if value and not str(value).startswith("/v1/audio?"):
            return Path(str(value))
    return None


def validate_audio(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=120,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-500:])
    data = json.loads(result.stdout)
    stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration") or 0)
    if int(stream.get("sample_rate") or 0) != 48000 or int(stream.get("channels") or 0) != 2 or duration < 10:
        raise RuntimeError(f"invalid generated audio: duration={duration}, stream={stream}")
    return {"duration": duration, "sample_rate": 48000, "channels": 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--adapter-dir", default=".")
    parser.add_argument("--prompts", default="inference_prompts.json")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-dir", default="generated")
    args = parser.parse_args()

    ace_root = Path(args.ace_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = adapter_dir / prompts_path
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompt = prompts[args.prompt_index]
    sys.path.insert(0, str(ace_root))

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    message, loaded = handler.initialize_service(
        project_root=str(checkpoint_root.parent), config_path="acestep-v15-xl-base",
        device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False,
    )
    if not loaded:
        raise RuntimeError(f"XL-Base initialization failed: {message}")
    lora_message = handler.add_lora(str(adapter_dir), adapter_name="melodic_edm_core_v1")
    if not lora_message.startswith("✅"):
        raise RuntimeError(lora_message)

    params = GenerationParams(
        caption=prompt["caption"], lyrics=LYRICS, instrumental=True,
        bpm=int(prompt["bpm"]), keyscale=prompt["keyscale"],
        timesignature=str(prompt["timesignature"]), duration=float(prompt["duration"]),
        inference_steps=50, guidance_scale=7.0, shift=1.0, seed=int(prompt["seed"]),
        thinking=False, use_cot_metas=False, use_cot_caption=False,
        use_cot_language=False, use_cot_lyrics=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(prompt["seed"])], audio_format="wav")
    output_dir = Path(args.output_dir).resolve()
    generated = generate_music(handler, None, params, config, save_dir=str(output_dir))
    if not generated.success or len(generated.audios) != 1:
        raise RuntimeError(generated.error or generated.status_message)
    path = audio_path(generated.audios[0])
    if path is None or not path.is_file():
        raise RuntimeError("generated audio path missing")
    probe = validate_audio(path)
    report = {"status": "pass", "prompt_id": prompt["id"], "seed": prompt["seed"], "audio_path": str(path), "probe": probe}
    atomic_json(output_dir / "inference_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

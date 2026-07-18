#!/usr/bin/env python3
"""Run deterministic ACE-Step XL-Base inference with a packaged v2 adapter."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LYRICS = """[Intro]
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
    """Write one JSON report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_adapter(path: Path) -> Path:
    """Resolve an adapter folder in either flat or named-PEFT layout."""
    nested = path / "adapter"
    return nested if nested.is_dir() else path


def audio_path(audio: dict[str, Any]) -> Path | None:
    """Extract a real local output path from an ACE-Step result."""
    for key in ("path", "file", "audio_path"):
        value = audio.get(key)
        if value and not str(value).startswith("/v1/audio?"):
            return Path(str(value))
    return None


def validate_audio(path: Path) -> dict[str, Any]:
    """Require a decodable 48 kHz stereo WAV of plausible duration."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-500:])
    data = json.loads(result.stdout)
    stream = next(
        (item for item in data.get("streams", []) if item.get("codec_type") == "audio"),
        {},
    )
    duration = float(data.get("format", {}).get("duration") or 0)
    sample_rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    if sample_rate != 48000 or channels != 2 or duration < 10:
        raise RuntimeError(
            f"invalid generated audio: duration={duration}, sample_rate={sample_rate}, channels={channels}"
        )
    return {"duration": duration, "sample_rate": sample_rate, "channels": channels}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--release-dir", default=".")
    parser.add_argument(
        "--adapter-subdirectory",
        choices=("best-val", "final-all-data"),
        default="final-all-data",
    )
    parser.add_argument("--prompts", default="fixed_eval_prompts.json")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-dir", default="generated-v2")
    parser.add_argument("--offload-to-cpu", action="store_true")
    args = parser.parse_args()

    # Colab exports its notebook-only matplotlib_inline backend to subprocesses.
    # The isolated ACE-Step venv does not include that backend, and Lightning's
    # torchmetrics import touches matplotlib even though inference never plots.
    os.environ["MPLBACKEND"] = "Agg"

    ace_root = Path(args.ace_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    release = Path(args.release_dir).resolve()
    adapter = resolve_adapter(release / args.adapter_subdirectory)
    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = release / prompts_path
    prompt_document = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = prompt_document.get("prompts", prompt_document)
    prompt = prompts[args.prompt_index]
    sys.path.insert(0, str(ace_root))

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    handler = AceStepHandler()
    message, loaded = handler.initialize_service(
        project_root=str(checkpoint_root.parent),
        config_path="acestep-v15-xl-base",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=args.offload_to_cpu,
    )
    if not loaded:
        raise RuntimeError(f"XL-Base initialization failed: {message}")
    adapter_name = f"melodic_edm_core_v2_{args.adapter_subdirectory.replace('-', '_')}"
    load_message = handler.add_lora(str(adapter), adapter_name=adapter_name)
    if not load_message.startswith("✅"):
        raise RuntimeError(load_message)
    active_message = handler.set_active_lora_adapter(adapter_name)
    if not active_message.startswith("✅"):
        raise RuntimeError(active_message)

    params = GenerationParams(
        caption=prompt["caption"],
        lyrics=prompt.get("lyrics", DEFAULT_LYRICS),
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
    output = Path(args.output_dir).resolve()
    generated = generate_music(handler, None, params, config, save_dir=str(output))
    if not generated.success or len(generated.audios) != 1:
        raise RuntimeError(generated.error or generated.status_message)
    path = audio_path(generated.audios[0])
    if path is None or not path.is_file():
        raise RuntimeError("generated audio path missing")
    report = {
        "status": "pass",
        "adapter": args.adapter_subdirectory,
        "prompt_id": prompt["id"],
        "seed": prompt["seed"],
        "audio_path": str(path),
        "probe": validate_audio(path),
    }
    atomic_json(output / "inference_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

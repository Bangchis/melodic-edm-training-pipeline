#!/usr/bin/env python3
"""Run configurable ACE-Step XL-Base inference with a packaged v2 adapter."""
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


# These are backward-compatible defaults for older prompt files. The Colab
# notebook writes every value explicitly, so sampling quality is controlled by
# the user-facing configuration cell instead of being locked in this script.
SAMPLING_DEFAULTS: dict[str, Any] = {
    "inference_steps": 64,
    "guidance_scale": 8.0,
    "shift": 1.0,
    "use_adg": True,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "infer_method": "ode",
    "sampler_mode": "euler",
    "velocity_norm_threshold": 0.0,
    "velocity_ema_factor": 0.0,
    "dcw_enabled": False,
    "dcw_mode": "double",
    "dcw_scaler": 0.05,
    "dcw_high_scaler": 0.02,
    "dcw_wavelet": "haar",
    "timesteps": None,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "fade_in_duration": 0.0,
    "fade_out_duration": 0.0,
    "latent_shift": 0.0,
    "latent_rescale": 1.0,
    "thinking": False,
    "ace_lm_model": "acestep-5Hz-lm-1.7B",
    "lm_temperature": 0.8,
    "lm_cfg_scale": 2.0,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "use_cot_metas": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "use_cot_lyrics": False,
}

OUTPUT_DEFAULTS: dict[str, Any] = {
    "batch_size": 1,
    "use_random_seed": False,
    "seeds": None,
    "audio_format": "wav",
    "mp3_bitrate": "320k",
    "mp3_sample_rate": 48000,
}

SAMPLING_KEYS = frozenset(SAMPLING_DEFAULTS)
OUTPUT_KEYS = frozenset(OUTPUT_DEFAULTS)
OUTPUT_FORMATS = frozenset(("mp3", "wav", "flac", "wav32", "opus", "aac"))


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


def merged_generation_settings(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and merge user-controlled sampling and output settings."""
    sampling_input = document.get("sampling", {})
    output_input = document.get("output", {})
    if not isinstance(sampling_input, dict) or not isinstance(output_input, dict):
        raise ValueError("sampling and output settings must be JSON objects")
    unknown_sampling = sorted(set(sampling_input) - SAMPLING_KEYS)
    unknown_output = sorted(set(output_input) - OUTPUT_KEYS)
    if unknown_sampling:
        raise ValueError(f"unsupported sampling settings: {unknown_sampling}")
    if unknown_output:
        raise ValueError(f"unsupported output settings: {unknown_output}")

    sampling = {**SAMPLING_DEFAULTS, **sampling_input}
    output = {**OUTPUT_DEFAULTS, **output_input}
    if int(sampling["inference_steps"]) < 1:
        raise ValueError("inference_steps must be at least 1")
    if float(sampling["guidance_scale"]) < 0:
        raise ValueError("guidance_scale must be non-negative")
    if float(sampling["shift"]) <= 0:
        raise ValueError("shift must be greater than zero")
    start = float(sampling["cfg_interval_start"])
    end = float(sampling["cfg_interval_end"])
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("CFG interval must satisfy 0 <= start <= end <= 1")
    if sampling["infer_method"] not in ("ode", "sde"):
        raise ValueError("infer_method must be 'ode' or 'sde'")
    if sampling["sampler_mode"] not in ("euler", "heun"):
        raise ValueError("sampler_mode must be 'euler' or 'heun'")
    if sampling["dcw_mode"] not in ("low", "high", "double", "pix"):
        raise ValueError("dcw_mode must be low, high, double or pix")
    timesteps = sampling["timesteps"]
    if timesteps is not None:
        if not isinstance(timesteps, list) or len(timesteps) < 2:
            raise ValueError("timesteps must be null or a list with at least two values")
        sampling["timesteps"] = [float(value) for value in timesteps]

    batch_size = int(output["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output["batch_size"] = batch_size
    if output["audio_format"] not in OUTPUT_FORMATS:
        raise ValueError(f"audio_format must be one of {sorted(OUTPUT_FORMATS)}")
    seeds = output["seeds"]
    if seeds is not None:
        if not isinstance(seeds, list):
            raise ValueError("seeds must be null or a list of integers")
        output["seeds"] = [int(seed) for seed in seeds]
    if not bool(output["use_random_seed"]):
        if output["seeds"] is not None and len(output["seeds"]) != batch_size:
            raise ValueError("deterministic seeds must contain exactly batch_size values")
    return sampling, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--release-dir", default=".")
    parser.add_argument(
        "--adapter-subdirectory",
        choices=("experimental-r32", "best-val", "final-all-data"),
        default="final-all-data",
    )
    parser.add_argument("--prompts", default="fixed_eval_prompts.json")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-dir", default="generated-v2")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-scale", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.lora_scale <= 1.0:
        parser.error("--lora-scale must be between 0 and 1")

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
    sampling, output_settings = merged_generation_settings(prompt_document)
    ace_lm_model = str(sampling.pop("ace_lm_model"))
    if output_settings["seeds"] is None and not output_settings["use_random_seed"]:
        output_settings["seeds"] = [int(prompt["seed"])] * output_settings["batch_size"]
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
    llm_handler = None
    if bool(sampling["thinking"]):
        from acestep.llm_inference import LLMHandler

        llm_handler = LLMHandler()
        lm_message, lm_loaded = llm_handler.initialize(
            checkpoint_dir=str(checkpoint_root),
            lm_model_path=ace_lm_model,
            backend="pt",
            device="cuda",
            offload_to_cpu=args.offload_to_cpu,
            dtype=None,
        )
        if not lm_loaded:
            raise RuntimeError(f"ACE 5Hz LM initialization failed: {lm_message}")
    adapter_name = f"melodic_edm_core_v2_{args.adapter_subdirectory.replace('-', '_')}"
    if not args.disable_lora:
        load_message = handler.add_lora(str(adapter), adapter_name=adapter_name)
        if not load_message.startswith("✅"):
            raise RuntimeError(load_message)
        active_message = handler.set_active_lora_adapter(adapter_name)
        if not active_message.startswith("✅"):
            raise RuntimeError(active_message)
        scale_message = handler.set_lora_scale(adapter_name, args.lora_scale)
        if not scale_message.startswith("✅"):
            raise RuntimeError(scale_message)

    params = GenerationParams(
        caption=prompt["caption"],
        lyrics=prompt.get("lyrics", DEFAULT_LYRICS),
        instrumental=True,
        bpm=int(prompt["bpm"]),
        keyscale=prompt["keyscale"],
        timesignature=str(prompt["timesignature"]),
        duration=float(prompt["duration"]),
        **sampling,
        seed=int(prompt["seed"]),
    )
    config = GenerationConfig(
        **output_settings,
    )
    output = Path(args.output_dir).resolve()
    generated = generate_music(handler, llm_handler, params, config, save_dir=str(output))
    if not generated.success or len(generated.audios) != output_settings["batch_size"]:
        raise RuntimeError(generated.error or generated.status_message)
    verified_audio = []
    for index, audio in enumerate(generated.audios):
        path = audio_path(audio)
        if path is None or not path.is_file():
            raise RuntimeError(f"generated audio path missing for output {index}")
        verified_audio.append({"audio_path": str(path), "probe": validate_audio(path)})
    report = {
        "status": "pass",
        "adapter": args.adapter_subdirectory if not args.disable_lora else "base-xl-no-lora",
        "lora_enabled": not args.disable_lora,
        "lora_scale": args.lora_scale if not args.disable_lora else 0.0,
        "prompt_id": prompt["id"],
        "seed": prompt["seed"],
        "seeds": output_settings["seeds"],
        "sampling": sampling,
        "ace_lm_model": ace_lm_model if sampling["thinking"] else None,
        "output": output_settings,
        "audio_path": verified_audio[0]["audio_path"],
        "probe": verified_audio[0]["probe"],
        "audios": verified_audio,
    }
    atomic_json(output / "inference_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

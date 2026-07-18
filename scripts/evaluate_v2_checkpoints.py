#!/usr/bin/env python3
"""Generate identical fixed prompts from every tenth v2 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from v2_common import atomic_json


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


def resolve_adapter(path: Path) -> Path:
    """Resolve PEFT's optional named adapter subdirectory."""
    nested = path / "adapter"
    return nested if nested.is_dir() else path


def checkpoint_state(path: Path) -> dict[str, int]:
    """Read epoch/global-step metadata when present."""
    state_path = path / "training_state.pt"
    if not state_path.is_file():
        return {}
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    return {"epoch": int(state.get("epoch", 0)), "optimizer_step": int(state.get("global_step", 0))}


def candidates(output: Path) -> list[dict[str, Any]]:
    """Discover every tenth checkpoint plus best-val and last adapters."""
    cutoff_path = output / "training_stop_override.json"
    cutoff = None
    if cutoff_path.is_file():
        override = json.loads(cutoff_path.read_text(encoding="utf-8"))
        if override.get("status") != "pass" or override.get("termination") != "user_requested_cutoff":
            raise RuntimeError("invalid training stop override")
        cutoff = int(override["requested_cutoff_epoch"])
    selected: list[dict[str, Any]] = []
    eligible_states: list[dict[str, int]] = []
    for path in sorted((output / "checkpoints").glob("epoch_*_loss_*")):
        match = re.match(r"epoch_(\d+)_loss_", path.name)
        if not match:
            continue
        state = checkpoint_state(path)
        if cutoff is not None and int(match.group(1)) > cutoff:
            continue
        eligible_states.append(state)
        if int(match.group(1)) % 10:
            continue
        selected.append({"label": f"epoch_{int(match.group(1)):03d}", "path": path, **state})
    validation = json.loads((output / "validation_state.json").read_text(encoding="utf-8"))
    selected.append({
        "label": "best_val",
        "path": output / "checkpoints" / "best_val",
        "epoch": int(validation["best_epoch"]),
        "optimizer_step": int(validation["best_optimizer_step"]),
    })
    if not eligible_states:
        raise RuntimeError("no eligible checkpoint remains after applying cutoff")
    last_state = max(eligible_states, key=lambda value: value.get("optimizer_step", 0))
    selected.append({"label": "last", "path": output / "final", **last_state})
    return selected


def probe(path: Path) -> tuple[bool, dict[str, Any]]:
    """Validate generated WAV format and duration."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        return False, {"error": (result.stderr or result.stdout)[-500:]}
    data = json.loads(result.stdout)
    stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration") or 0)
    valid = int(stream.get("sample_rate") or 0) == 48000 and int(stream.get("channels") or 0) == 2 and duration >= 30
    return valid, {
        "duration": duration,
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
    }


def audio_path(value: dict[str, Any]) -> Path | None:
    """Extract a real local path from one ACE-Step audio response."""
    for key in ("path", "file", "audio_path"):
        candidate = value.get(key)
        if candidate and not str(candidate).startswith("/v1/audio?"):
            return Path(str(candidate))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "train-validation"
    gate = json.loads((output / "training_validation_report.json").read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("v2 train-validation gate has not passed")
    prompts = json.loads(
        (root / "configs" / "v2" / "fixed_eval_prompts.json").read_text(encoding="utf-8")
    )["prompts"]
    checkpoints = candidates(output)

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
    evaluation_root = root / "outputs" / "v2" / "checkpoint-evaluation"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        label = checkpoint["label"]
        adapter = resolve_adapter(Path(checkpoint["path"]))
        load_message = handler.add_lora(str(adapter), adapter_name=label)
        if not load_message.startswith("✅"):
            raise RuntimeError(load_message)
        active_message = handler.set_active_lora_adapter(label)
        if not active_message.startswith("✅"):
            raise RuntimeError(active_message)
        for prompt in prompts:
            target_dir = evaluation_root / "audio" / label / prompt["id"]
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
            generated = generate_music(handler, None, params, config, save_dir=str(target_dir))
            if not generated.success or len(generated.audios) != 1:
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": generated.error or generated.status_message})
                continue
            path = audio_path(generated.audios[0])
            if path is None or not path.is_file():
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": "generated_audio_path_missing"})
                continue
            valid, audio_probe = probe(path)
            record = {
                "checkpoint": label,
                "adapter_path": str(adapter),
                "epoch": checkpoint.get("epoch"),
                "optimizer_step": checkpoint.get("optimizer_step"),
                "prompt_id": prompt["id"],
                "prompt": prompt["caption"],
                "seed": prompt["seed"],
                "audio_path": str(path),
                "probe": audio_probe,
                "status": "pass" if valid else "failed",
            }
            results.append(record)
            if not valid:
                errors.append({"checkpoint": label, "prompt": prompt["id"], "reason": "audio_validation_failed"})
            print(f"[{label}] {prompt['id']} {'PASS' if valid else 'FAIL'}", flush=True)
        handler.remove_lora(label)
    expected = len(checkpoints) * len(prompts)
    report = {
        "status": "pass" if not errors and len(results) == expected else "failed",
        "fixed_prompt_count": len(prompts),
        "checkpoint_count": len(checkpoints),
        "expected_outputs": expected,
        "generated_outputs": len(results),
        "results": results,
        "errors": errors,
    }
    atomic_json(evaluation_root / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

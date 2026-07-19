#!/usr/bin/env python3
"""Generate fixed instrumental prompts from v2 training adapters."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from v2_common import atomic_json, object_sha256


LORA_SCALES = (0.5,)


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


def candidates(
    output: Path,
    *,
    final_only: bool = False,
    best_only: bool = False,
) -> list[dict[str, Any]]:
    """Discover every fifth checkpoint plus best-val and last adapters."""
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
        if int(match.group(1)) % 5:
            continue
        selected.append({"label": f"epoch_{int(match.group(1)):03d}", "path": path, **state})
    validation = json.loads((output / "validation_state.json").read_text(encoding="utf-8"))
    best_candidate = {
        "label": "best_val",
        "path": output / "checkpoints" / "best_val",
        "epoch": int(validation["best_epoch"]),
        "optimizer_step": int(validation["best_optimizer_step"]),
    }
    selected_identities = {
        (int(candidate.get("epoch", 0)), int(candidate.get("optimizer_step", 0)))
        for candidate in selected
    }
    best_identity = (best_candidate["epoch"], best_candidate["optimizer_step"])
    if best_identity not in selected_identities:
        selected.append(best_candidate)
        selected_identities.add(best_identity)
    if best_only:
        return [best_candidate]
    if not eligible_states:
        raise RuntimeError("no eligible checkpoint remains after applying cutoff")
    last_state = max(eligible_states, key=lambda value: value.get("optimizer_step", 0))
    final_candidate = {"label": "last", "path": output / "final", **last_state}
    if final_only:
        return [final_candidate]
    final_identity = (
        int(final_candidate.get("epoch", 0)),
        int(final_candidate.get("optimizer_step", 0)),
    )
    if final_identity not in selected_identities:
        selected.append(final_candidate)
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
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Generate only from the final epoch adapter after training completes.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Generate only from the lowest-validation-loss adapter.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Generate only from the checkpoint selected by the full quality/alignment gate.",
    )
    parser.add_argument(
        "--evaluation-dir",
        default="outputs/v2/checkpoint-evaluation",
        help="Output directory, relative to the project root by default.",
    )
    parser.add_argument(
        "--ace-lm-thinking",
        action="store_true",
        help="Use the local ACE 5Hz LM to plan semantic audio codes before diffusion.",
    )
    parser.add_argument("--ace-lm-model", default="acestep-5Hz-lm-1.7B")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Generate with XL-Base only, without loading the trained LoRA.",
    )
    parser.add_argument("--prompt-id", help="Generate only one fixed prompt id.")
    parser.add_argument(
        "--seed-offsets",
        default="0",
        help="Comma-separated deterministic offsets added to each fixed prompt seed.",
    )
    parser.add_argument("--duration-override", type=float)
    parser.add_argument(
        "--prompts-file",
        default="configs/v2/fixed_eval_prompts.json",
        help="Prompt JSON file, relative to the project root by default.",
    )
    args = parser.parse_args()
    if sum((args.final_only, args.best_only, args.selected_only)) > 1:
        parser.error("--final-only, --best-only and --selected-only are mutually exclusive")
    if args.base_only and any((args.final_only, args.best_only, args.selected_only)):
        parser.error("--base-only cannot be combined with a LoRA checkpoint selector")
    root = Path(args.project_root).resolve()
    output = root / "outputs" / "v2" / "train-validation"
    gate = json.loads((output / "training_validation_report.json").read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("v2 train-validation gate has not passed")
    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = root / prompts_path
    prompt_document = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = prompt_document["prompts"]
    if args.prompt_id:
        prompts = [prompt for prompt in prompts if prompt["id"] == args.prompt_id]
        if len(prompts) != 1:
            raise RuntimeError(f"fixed prompt id not found: {args.prompt_id}")
    try:
        seed_offsets = [int(value.strip()) for value in args.seed_offsets.split(",") if value.strip()]
    except ValueError as exc:
        raise RuntimeError("--seed-offsets must contain integers") from exc
    if not seed_offsets:
        raise RuntimeError("--seed-offsets must not be empty")
    expanded_prompts = []
    for prompt in prompts:
        for offset in seed_offsets:
            expanded = dict(prompt)
            expanded["source_prompt_id"] = prompt["id"]
            expanded["seed"] = int(prompt["seed"]) + offset
            if len(seed_offsets) > 1 or offset:
                expanded["id"] = f"{prompt['id']}_seed_{expanded['seed']}"
            if args.duration_override is not None:
                expanded["duration"] = float(args.duration_override)
            expanded_prompts.append(expanded)
    prompts = expanded_prompts
    selection_sha256 = None
    if args.base_only:
        checkpoints = [{"label": "base_xl", "path": None, "epoch": 0, "optimizer_step": 0}]
    elif args.selected_only:
        selection_path = root / "outputs" / "v2" / "checkpoint-evaluation" / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "pass" or selection.get("quality_accepted") is not True:
            raise RuntimeError("quality-gated checkpoint selection has not passed")
        adapter_path = Path(str(selection.get("best_val_output") or ""))
        if not adapter_path.is_absolute():
            adapter_path = root / adapter_path
        if not adapter_path.is_dir():
            raise RuntimeError(f"selected best-val adapter is missing: {adapter_path}")
        selection_sha256 = object_sha256(selection)
        checkpoints = [{
            "label": "selected_best",
            "path": adapter_path,
            "epoch": int(selection["selected_epoch"]),
            "optimizer_step": int(selection["best_optimizer_step"]),
        }]
    else:
        checkpoints = candidates(output, final_only=args.final_only, best_only=args.best_only)

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
    llm_handler = None
    if args.ace_lm_thinking:
        from acestep.llm_inference import LLMHandler

        llm_handler = LLMHandler()
        lm_message, lm_loaded = llm_handler.initialize(
            checkpoint_dir=str(root / "checkpoints"),
            lm_model_path=args.ace_lm_model,
            backend="pt",
            device="cuda",
            offload_to_cpu=False,
            dtype=None,
        )
        if not lm_loaded:
            raise RuntimeError(f"ACE 5Hz LM initialization failed: {lm_message}")
    evaluation_root = Path(args.evaluation_dir)
    if not evaluation_root.is_absolute():
        evaluation_root = root / evaluation_root
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        label = checkpoint["label"]
        adapter = None
        if checkpoint["path"] is not None:
            adapter = resolve_adapter(Path(checkpoint["path"]))
            load_message = handler.add_lora(str(adapter), adapter_name=label)
            if not load_message.startswith("✅"):
                raise RuntimeError(load_message)
            active_message = handler.set_active_lora_adapter(label)
            if not active_message.startswith("✅"):
                raise RuntimeError(active_message)
        active_scales = (0.0,) if args.base_only else LORA_SCALES
        for lora_scale in active_scales:
            scaled_label = f"{label}_scale_{lora_scale:.2f}"
            if adapter is not None:
                scale_message = handler.set_lora_scale(label, lora_scale)
                if not scale_message.startswith("✅"):
                    raise RuntimeError(scale_message)
            for prompt in prompts:
                target_dir = evaluation_root / "audio" / scaled_label / prompt["id"]
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
                    thinking=args.ace_lm_thinking,
                    lm_temperature=0.8,
                    lm_cfg_scale=2.0,
                    lm_top_k=0,
                    lm_top_p=0.9,
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
                    handler,
                    llm_handler,
                    params,
                    config,
                    save_dir=str(target_dir),
                )
                if not generated.success or len(generated.audios) != 1:
                    errors.append({"checkpoint": scaled_label, "prompt": prompt["id"], "reason": generated.error or generated.status_message})
                    continue
                path = audio_path(generated.audios[0])
                if path is None or not path.is_file():
                    errors.append({"checkpoint": scaled_label, "prompt": prompt["id"], "reason": "generated_audio_path_missing"})
                    continue
                valid, audio_probe = probe(path)
                record = {
                    "checkpoint": scaled_label,
                    "checkpoint_base": label,
                    "lora_scale": lora_scale,
                    "adapter_path": str(adapter) if adapter is not None else None,
                    "epoch": checkpoint.get("epoch"),
                    "optimizer_step": checkpoint.get("optimizer_step"),
                    "prompt_id": prompt["id"],
                    "source_prompt_id": prompt["source_prompt_id"],
                    "prompt": prompt["caption"],
                    "lyrics": prompt.get("lyrics", DEFAULT_LYRICS),
                    "duration": float(prompt["duration"]),
                    "seed": prompt["seed"],
                    "audio_path": str(path),
                    "probe": audio_probe,
                    "status": "pass" if valid else "failed",
                }
                results.append(record)
                if not valid:
                    errors.append({"checkpoint": scaled_label, "prompt": prompt["id"], "reason": "audio_validation_failed"})
                print(f"[{scaled_label}] {prompt['id']} {'PASS' if valid else 'FAIL'}", flush=True)
        if adapter is not None:
            handler.remove_lora(label)
    expected = len(checkpoints) * len(LORA_SCALES) * len(prompts)
    report = {
        "status": "pass" if not errors and len(results) == expected else "failed",
        "fixed_prompt_count": len(prompts),
        "checkpoint_count": len(checkpoints),
        "checkpoint_mode": (
            "base_only"
            if args.base_only
            else "selected_only"
            if args.selected_only
            else "final_only"
            if args.final_only
            else "best_only"
            if args.best_only
            else "all_candidates"
        ),
        "ace_lm_thinking": args.ace_lm_thinking,
        "ace_lm_model": args.ace_lm_model if args.ace_lm_thinking else None,
        "structure_conditioning": "per_prompt_training_vocabulary_sections_v3",
        "prompt_filter": args.prompt_id,
        "seed_offsets": seed_offsets,
        "duration_override": args.duration_override,
        "prompts_file": str(prompts_path),
        "prompt_document_sha256": object_sha256(prompt_document),
        "checkpoint_selection_sha256": selection_sha256,
        "lora_scales": [0.0] if args.base_only else list(LORA_SCALES),
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

# Verified Vast environment

All large downloads and installations were performed on the Vast instance, not on
the local workstation.

## Host

- 2× NVIDIA GeForce RTX 4090 (24 GB each)
- NVIDIA driver 550.78
- FFmpeg 4.4.2
- `uv` 0.11.19

Do not replace the NVIDIA driver inside a rented instance.

## MIR environment

Python 3.11 at `.venvs/mir`:

- PyTorch 2.0.0+cu118
- NATTEN 0.14.6+torch200cu118
- All-In-One 1.1.0
- Demucs 4.1.0
- Essentia 2.1b6.dev1389
- madmom 0.17.dev0 at commit
  `27f032e8947204902c675e5e341a3faf5dc86dae`
- NumPy 1.26.4
- librosa 0.11.0

The verified NATTEN wheel came from the official release:

```text
https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch200cu118-cp311-cp311-linux_x86_64.whl
```

All-In-One 1.1.0 has a cache-only cleanup bug when every requested raw JSON already
exists. `scripts/analyze_mir.py` avoids that branch and removes its work directories
only after the final MIR gate passes.

## Separator environment

Python environment at `.venvs/separator`:

- audio-separator 0.44.3
- PyTorch 2.10.0+cu128
- ONNX Runtime GPU 1.23.2

## ACE-Step environment

Pinned repository commit:

```text
6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
```

Its `.venv` includes:

- PyTorch / torchaudio 2.10.0+cu128
- Transformers 4.57.6
- Accelerate 1.12.0
- PEFT 0.18.1
- Lightning 2.6.1
- huggingface_hub 0.36.0
- qwen-omni-utils 0.0.9

The local audio annotator is `Qwen/Qwen2.5-Omni-7B`, pinned at immutable revision:

```text
ae9e1690543ffd5c0221dc27f79834d0294cba00
```

It is stored under `checkpoints/Qwen2.5-Omni-7B` on Vast with a
`PINNED_REVISION` completion marker. The talker is disabled and the model is loaded
with an automatic two-GPU device map using SDPA. No local workstation model download
is required, and local annotation audio is not uploaded to another provider.

Apply `patches/acestep-xl-validation-caption-variants.patch` to the pinned checkout.
The authoritative patch checksum and apply commands are in `RUNBOOK.md`.

`/workspace` is ordinary container storage on this instance, not a mounted persistent
volume. Stop/start preserves it, but recycle/destroy does not; use the documented
GitHub and private Hugging Face backups before recycling or destroying the instance.

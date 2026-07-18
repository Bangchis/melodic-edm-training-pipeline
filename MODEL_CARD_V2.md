---
license: other
library_name: peft
base_model: ACE-Step/acestep-v15-xl-base
tags:
  - music-generation
  - ace-step
  - lora
  - instrumental
  - edm
---

# Melodic EDM Core V2

Private ACE-Step 1.5 XL-Base LoRA adapters for instrumental melodic EDM generation. The release contains two adapters:

- `final-all-data`: recommended deployment adapter, freshly trained on all 231 validated records.
- `best-val`: checkpoint selected from a grouped 196-train/35-validation run.

## Training facts

- LoRA rank 48, alpha 96, dropout 0.1.
- Exact targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- BF16 AdamW, learning rate `7.5e-5`, cosine schedule and 75-step warmup.
- DDP on 2 × RTX 4090, batch 1/GPU, accumulation 8, effective batch 16.
- Three caption embeddings per record: canonical, composition and production.
- Uniform random caption selection during training; canonical-only validation.
- CFG dropout 0.15 for training and 0.0 for validation.
- Grouped split by `parent_song_id`; no parent crosses train/validation.
- No test split in the first research run.

MOSS-Music-8B-Thinking was used only to listen to source audio for annotation supplements and to score fixed checkpoint examples. It was not trained, fine-tuned, connected to the ACE-Step gradient graph or included in these adapters.

## Files

```text
best-val/
final-all-data/
configs/
docs/
examples/best-val/
examples/final-all-data/
metrics/
notebooks/melodic_edm_core_v2_colab.ipynb
reports/
SHA256SUMS
```

The model release excludes source audio, stems, cached latents, optimizer states, service credentials, browser cookies and model reasoning traces.

The exact 231 FLAC training records are backed up separately in the private dataset
`Bangchis/melodic-edm-audio-v2`, preserving 196 train and 35 validation records with
one file per catalog record, a manifest and SHA-256 checksums. Audio is intentionally
not duplicated inside this model repository.

## Inference

Use `final-all-data` with the exact ACE-Step source and XL-Base revisions recorded in `release_manifest.json`. The included Colab notebook sends a free-form idea through an OpenRouter LLM, requires exactly five JSON music-description fields, then applies a local validator/compiler with a hard 300-word inference limit. BPM, key, time signature and instrumental sections remain separate fixed conditions. The notebook performs immutable download, checksum verification, generation and 48 kHz stereo validation. Training captions remain 40–80 words.

See `docs/COLAB_INFERENCE_V2.md` and `notebooks/melodic_edm_core_v2_colab.ipynb`.

## Evaluation and limitations

The selected adapter combines grouped validation loss, fixed-seed audio samples, MOSS audio-listening scores, prompt-output diversity and a conservative similarity check against training audio features. The included selection report states whether human listening was completed; automated MOSS scoring is not represented as a human preference test.

This is a small, style-focused research dataset. The adapter can over-specialize, produce inconsistent long-form structure, or reflect bias in the source catalog and machine annotations. Feature similarity is a screening heuristic, not proof against memorization. Users should review generated audio and rights obligations before publication or commercial use.

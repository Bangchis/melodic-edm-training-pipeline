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

- LoRA rank 32, alpha 32, dropout 0.1.
- Exact targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- BF16 AdamW, learning rate `5e-5`, cosine schedule and 25-step warmup.
- DDP on 2 × RTX 4090, batch 1/GPU, accumulation 8, effective batch 16.
- Three fused prompt embeddings per record (canonical, composition and production); each independently combines that song's old prompt properties with new MOSS audio evidence.
- Uniform random selection among all three prompt embeddings during training; canonical index 0 only during validation.
- CFG dropout 0.15 for training and 0.0 for validation.
- Grouped split by `parent_song_id`; no parent crosses train/validation.
- No test split in the first research run.

MOSS-Music-8B-Thinking was used only to listen to source audio for annotation supplements and to score fixed checkpoint examples. It was not trained, fine-tuned, connected to the ACE-Step gradient graph or included in these adapters.

The annotation listener runs under prompt revision `audio-blind-v2.2`, which withholds identity, MIR and prior claims for an independent waveform reading. Named sound sources from both the old per-song annotation and this new reading then pass two independent full-track checks plus an intro/middle/late montage. Compiler revision `openrouter-per-track-prior-audio-fusion-v2.8` uses `google/gemini-3.1-flash-lite` through OpenRouter as a text-only editor to fuse the old prompt properties of that exact song with the independently heard facts under separately hashed lineage. Exact instrument names are preserved when supported by consensus; absent claims are removed and unresolved timbres are explicitly qualified rather than silently asserted. The resulting canonical caption is the one training prompt; composition and production remain audit-only views. A final MOSS listening audit checks caption fidelity against audio and is included in the completion evidence.

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

Use `final-all-data` with the exact ACE-Step source and XL-Base revisions recorded in `release_manifest.json`. The included Colab notebook sends a free-form idea through an OpenRouter LLM, requires exactly five JSON music-description fields, then applies a local validator/compiler with a hard 300-word inference limit. BPM, key, time signature and instrumental sections remain separate fixed conditions. The notebook performs immutable download, checksum verification, generation and 48 kHz stereo validation. Training captions remain 40–80 words. The packaged default uses 64 steps, guidance 8, XL-Base shift 1, ADG enabled and DCW disabled; every control remains editable.

See `docs/COLAB_INFERENCE_V2.md` and `notebooks/melodic_edm_core_v2_colab.ipynb`.

## Evaluation and limitations

The selected adapter combines grouped validation loss, fixed-seed audio samples, MOSS audio-listening scores, prompt-output diversity and a conservative similarity check against training audio features. The included selection report states whether human listening was completed; automated MOSS scoring is not represented as a human preference test.

This is a small, style-focused research dataset. The adapter can over-specialize, produce inconsistent long-form structure, or reflect bias in the source catalog and machine annotations. Feature similarity is a screening heuristic, not proof against memorization. Users should review generated audio and rights obligations before publication or commercial use.

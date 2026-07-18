---
license: other
library_name: peft
base_model: ACE-Step/acestep-v15-xl-base
tags:
  - music-generation
  - ace-step
  - lora
  - experimental
---

# Melodic EDM Core V2 R32 — experimental preview

This is a temporary, private inference preview of the rank-32/alpha-32 epoch-30 LoRA. It is published so the owner can test prompts and sampling in Colab while a corrected-data retraining run is prepared. It is **not the final V2 model** and must not be described as generally better than XL-Base.

## Measured result

The fixed comparison used three held-out styles, five deterministic seeds per style, 180-second instrumental structure conditioning, LoRA scale `0.5`, and the same prompt/seed pairs for LoRA and pristine XL-Base.

| Style | LoRA pass | Base pass |
|---|---:|---:|
| Chinese melodic EDM | 4/5 | 3/5 |
| Gaming progressive house | 1/5 | 1/5 |
| Cinematic glitch-hop | 1/5 | 2/5 |
| **Overall** | **6/15 (40%)** | **6/15 (40%)** |

Mean LoRA-minus-Base score deltas were approximately `alignment -0.13`, `melody -0.13`, `structure -0.07`, and `audio quality +0.20`. The adapter improved Chinese melody/structure in this test, did not fix the weak Gaming prompt, and regressed Cinematic slightly. The new training pipeline therefore remains in progress.

## Colab use

Open the reviewed notebook:

<https://colab.research.google.com/github/Bangchis/melodic-edm-training-pipeline/blob/agent/training-v2-r32/notebooks/melodic_edm_core_v2_colab.ipynb>

Colab Secrets:

- `HF_TOKEN`: required read token for this private model repository.
- `OPENROUTER_API_KEY`: optional; required only when the enhancer switch is on.

The notebook exposes prompt enhancement, direct caption mode, optional paired `REFERENCE_ARTIST`/`REFERENCE_TRACK_TITLE` style conditioning, LoRA enable/scale, ACE Thinking, duration and sections, seed/batch, diffusion steps, guidance, shift, ADG, CFG interval, ODE/SDE, Euler/Heun, DCW, normalization, fades, latent controls and output format in one cell. Set both reference fields to identify an artist and a specific reference track, or leave both empty. The notebook prepends the same deterministic style sentence used by training in both enhancer and direct-caption modes; OpenRouter may describe compatible audible traits but cannot invent a substitute identity.

Start with the preset, then edit duration and sections independently in the same control cell. Seven sections at 180 seconds are only the evaluated starter values, not a hard constraint; descriptive labels such as `Atmospheric Intro` and `First Melodic Drop` are accepted. Only empty, duplicate, bracketed or multiline labels are rejected because they break ACE structure syntax. The notebook warns about unusually dense section plans but never rewrites or blocks them. `CUSTOM_LYRICS` replaces the generated structure text before section validation. Begin with a concrete 40–80 word caption and only a few truly mandatory terms. Long captions and large required-keyword lists can conflict and reduce coherence. Compare Base and LoRA with the same prompt and seed. Post-processing cannot repair a collapsed melody or static loop.

`USE_ACE_LM_THINKING=False` matches the published 6/15 evaluation. Thinking is an optional inference planner and should be tested as a separate A/B variable; it was not trained together with this LoRA.

## Reproducibility

- Adapter: LoRA rank 32, alpha 32, dropout 0.1.
- Recommended initial LoRA scale: `0.5`.
- Base: `ACE-Step/acestep-v15-xl-base`.
- Default diagnostic sampler: 64 steps, guidance 8, shift 1, ADG on, DCW off.
- Output validation: decodable 48 kHz stereo audio.

The package includes sanitized training and multi-seed evaluation reports, prompt files, inference scripts, the Colab notebook and `SHA256SUMS`. Tokens, source audio, tensors, optimizer state and MOSS hidden reasoning are excluded.

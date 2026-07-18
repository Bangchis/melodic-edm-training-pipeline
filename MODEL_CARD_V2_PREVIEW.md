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

# Melodic EDM Core V2 — best-val preview

This private preview contains the validation-selected `best-val` adapter from the
grouped 196-train/35-validation run. It is published before the fresh all-231
retraining run so inference can be tested without waiting for `final-all-data`.

This is not the final deployment adapter. The repository will receive
`final-all-data` only after a new rank-48 LoRA has been trained from pristine
ACE-Step 1.5 XL-Base weights on all 231 records.

## Preview files

```text
best-val/
configs/
docs/
examples/best-val/
notebooks/melodic_edm_core_v2_colab.ipynb
reports/
scripts/
SHA256SUMS
```

The preview excludes source audio, stems, cached tensors, optimizer states,
credentials, browser cookies and MOSS reasoning. Use the exact immutable Hugging
Face revision printed by the Colab notebook and select `best-val` when prompted.

The included notebook enhances a free-form idea through OpenRouter strict JSON
output, then validates and compiles genre, mood, melody, arrangement and
production through `scripts/prompt_enhancer.py` before inference. The OpenRouter
key stays in Colab Secrets and is never packaged with this release.

See `docs/COLAB_INFERENCE_V2.md` for the pinned ACE-Step/XL-Base setup and the
48 kHz stereo inference checks.

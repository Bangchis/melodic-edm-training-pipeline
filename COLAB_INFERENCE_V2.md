# Colab Pro inference for Melodic EDM Core V2

This path is inference-only. It does not install MOSS, load the training tensors or continue LoRA training. At the owner's request, the notebook now defaults to the packaged epoch-30 rank-32 adapter (`USE_LORA = True`, `LORA_SCALE = 0.5`) with a text-only Xomu — Lanterns reconstruction preset. It also enables the official ACE 5 Hz 4B planner with low-temperature semantic planning. The adapter remains experimental because it did not pass the general release quality gate, and a title-conditioned text prompt cannot guarantee an exact note-for-note reconstruction.

## How Colab Pro is authenticated

Consumer Colab Pro runs from the Colab website in a browser. It does not provide a supported CLI that a Vast server can use to submit a job to a hosted Pro runtime. Therefore:

- do not install a Google/Colab CLI on Vast for this workflow;
- do not copy a Google password, browser cookie or Google OAuth refresh token to Vast;
- sign in to the Google account that owns Colab Pro in the browser, open the notebook, select an NVIDIA GPU and run it there;
- put a Hugging Face **read** token in Colab Secrets as `HF_TOKEN` so the notebook can download the private release;
- optionally put an OpenRouter key in Colab Secrets as `OPENROUTER_API_KEY` when the prompt enhancer is enabled.

The notebook and inference script are published by GitHub/Hugging Face. The Colab VM pulls the immutable release directly from Hugging Face; Vast does not push a process into Colab.

Open the current reviewed V2 branch from:

```text
https://colab.research.google.com/github/Bangchis/melodic-edm-training-pipeline/blob/agent/training-v2-r32/notebooks/melodic_edm_core_v2_colab.ipynb
```

While the GitHub repository is private, first authorize GitHub access from Colab's **File → Open notebook → GitHub** tab. Once the repository is public, the direct URL works without GitHub authorization.

If unattended server-side submission is required, that is a separate Google Cloud Colab Enterprise workflow. It requires a Google Cloud project, billing, IAM permissions, a runtime template and its own quota; a consumer Colab Pro subscription is not that service.

## Requirements

- A browser signed into the Google account that owns the active Colab Pro subscription.
- A Colab Pro runtime with an NVIDIA GPU. GPU type and availability are assigned dynamically by Colab and are not guaranteed.
- At least 20 GB GPU memory is preferred for XL-Base. A 12–16 GB GPU may work with CPU offload and will be slower.
- Roughly 50–65 GB free disk for the ACE-Step environment, XL-Base, the 4B LM planner and the private adapter release.
- A Hugging Face read token stored in Colab Secrets as `HF_TOKEN`.
- An OpenRouter API key stored in Colab Secrets as `OPENROUTER_API_KEY` only when `USE_OPENROUTER_ENHANCER = True`.

Never paste either token into a notebook cell or commit it to GitHub. In Colab, open the key icon, create `HF_TOKEN` and, if needed, `OPENROUTER_API_KEY`, then enable notebook access. Only the text idea and explicit music conditions are sent to OpenRouter; no audio, adapter or Hugging Face token is sent there. With the enhancer disabled, no OpenRouter request is made.

No Google authentication is required on the local machine beyond the browser session, and no Google authentication is required on Vast.

## Reproducible versions

```text
ACE-Step source: 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
XL-Base model: ACE-Step/acestep-v15-xl-base
XL-Base revision: 220c1166efbdd9583eafcb12eb160594bbfcb241
V2 adapter repo: Bangchis/melodic-edm-core-v2
```

At the beginning of one Colab run, the notebook resolves the model repository's current head to its immutable Hugging Face commit SHA, prints that SHA and uses it for the entire download. It never passes the moving `main` name to `snapshot_download`. Record the printed SHA with any generated example that must be reproduced later.

## Notebook flow

Use `notebooks/melodic_edm_core_v2_colab.ipynb`. It performs these gates in order:

1. Confirm NVIDIA GPU, VRAM and disk space.
2. Read required `HF_TOKEN` and optional `OPENROUTER_API_KEY` from Colab Secrets without displaying them.
3. Clone ACE-Step at the pinned source revision.
4. Install the official environment with `uv sync`.
5. Download the core ACE-Step checkpoints and pinned XL-Base weights.
6. Resolve the private V2 release to one immutable commit and download exactly that revision.
7. Verify every release file with `SHA256SUMS`.
8. Download and verify the epoch-30 experimental adapter package, then enable it at scale `0.5` for the requested reference-oriented preset.
9. Either enhance the free-form idea through OpenRouter or use the direct caption unchanged, according to one switch.
10. Pass every user-selected sampling/output setting to ACE-Step, then validate each generated audio file.
11. Inspect and play the result inside Colab.

When enabled, the optional enhancer passes the free-form idea through `scripts/enhance_prompt_openrouter.py`. The available route is `~google/gemini-flash-latest`, configurable with an exact OpenRouter model slug. The notebook defaults to direct-caption mode, so no OpenRouter request is made. Separately, ACE-Step runs with `thinking=True` and `acestep-5Hz-lm-4B` to create semantic music codes from the exact artist/title-conditioned caption. Caption rewriting and metadata guessing remain disabled, so the explicit 128 BPM, A minor, 4/4 and 232-second controls are preserved. This planner may improve reconstruction from model knowledge, but source-audio Cover mode would still be required to guarantee structural melody control.

```text
free-form idea + explicit BPM/key/time/sections
→ OpenRouter LLM with strict JSON Schema
→ genre + mood + melody + arrangement + production
→ deterministic validator/compiler
→ 40–300 word inference caption
+ BPM/key/time signature/sections
→ ACE-Step XL-Base + selected LoRA
```

The LLM is the optional enhancer; the deterministic stage is only a safety/format gate. The inference idea is authoritative and is never passed through the conservative audio-annotation claim policy. Exact instruments or other phrases listed in `REQUIRED_PROMPT_TERMS` must survive in the compiled caption or the enhancer retries/fails before ACE-Step runs. For example, requiring `pipa` and `dizi` prevents the LLM from weakening them to generic `plucked-string-like` and `flute-like` terms. This list is user-controlled and may be empty. `REFERENCE_ARTIST` and `REFERENCE_TRACK_TITLE` are an optional pair; when both are set, the notebook prepends the same deterministic artist/track style-reference sentence used during training. These are real style-conditioning words sent to ACE-Step, not metadata stored beside the prompt. The LLM may translate that explicit reference into compatible audible traits but cannot invent or substitute another identity. Leave both fields empty to use a purely descriptive prompt. Explicit BPM, key, time signature and sections always overwrite any LLM guess. Set `USE_OPENROUTER_ENHANCER = False` to send the style-prefixed `DIRECT_CAPTION` straight to ACE-Step without needing an OpenRouter secret. The notebook records the complete secret-free conditioning payload in `/content/v2_prompt_enhancement.json` for reproducibility.

## One generation-control cell

Edit only the notebook cell titled **All generation controls**. It contains the adapter choice, LoRA enable/scale, enhancer switch, musical conditions, duration, seed(s), diffusion steps, guidance, shift, ADG/CFG interval, ODE/SDE method, Euler/Heun sampler, velocity controls, custom timesteps, DCW controls, normalization, fades, latent post-processing, batch size and output encoding. The inference script validates and uses those values; it does not replace them with hidden quality settings.

The requested starter state is:

```python
ADAPTER_CHOICE = "experimental-r32"  # epoch 30 / step 360
USE_LORA = True
LORA_SCALE = 0.5
USE_ACE_LM_THINKING = True
ACE_LM_MODEL = "acestep-5Hz-lm-4B"
LM_TEMPERATURE = 0.3
GUIDANCE_SCALE = 9.0
```

For a controlled diagnosis, keep the prompt and seed unchanged and compare it with:

```python
USE_LORA = False   # pristine XL-Base comparison

USE_LORA = True
LORA_SCALE = 0.25

USE_LORA = True
LORA_SCALE = 0.5

USE_LORA = True
LORA_SCALE = 1.0  # optional stress test only; the adapter failed the release gate
```

If the base output is coherent while higher LoRA scales become noisy, the adapter is the cause; mastering or normalization will not repair it. If the base is also broken, investigate the pinned base/checkpoint/sampling path first.

## Prompt format

The enhancer accepts up to 300 words including the optional deterministic artist/track prefix, but ACE-Step documents its main caption as a short input, so begin with roughly 40–80 descriptive words for diagnosis. Increase detail only after a short prompt generates coherently. Describe melody, composition and production; keep BPM/key/time signature in their fields. Use artist/title names only through the explicit paired fields instead of burying or repeating them in free prose. Avoid vague quality claims. Training caption bodies remain 40–80 words before their style-reference prefix.

The notebook accepts a free-form idea plus optional fixed conditions:

```python
USER_IDEA = "Melodic EDM không lời, không khí đèn lồng đêm hoài niệm nhưng tươi sáng"
EXPLICIT_CONDITIONS = {
    "reference_artist": "Xomu",
    "reference_track": "Lanterns",
    "bpm": 128,
    "keyscale": "A minor",
    "timesignature": "4",
    "sections": ["Filtered Intro", "Theme Development", "First Build", "First Progressive-House Drop", "Atmospheric Breakdown", "Second Build", "Final Euphoric Drop", "Filtered Outro"],
    "required_terms": [],
}
enhancement = enhance_prompt(
    USER_IDEA,
    OPENROUTER_API_KEY,
    model="~google/gemini-flash-latest",
    explicit_conditions=EXPLICIT_CONDITIONS,
)
```

```python
caption = (
    "Match the recognizable central melody, note contour, rhythmic phrasing, chord progression, "
    "section order and energy arc as closely as possible. Oriental progressive house with a glassy "
    "pentatonic pluck, soft piano, sparkling arpeggios, rolling bass, side-chained synth chords, "
    "clean four-on-the-floor drums, an atmospheric breakdown and euphoric final drop. Instrumental only."
)
bpm = 128
keyscale = "A minor"
timesignature = "4"
```

Use instrumental structure text:

```text
[Intro]
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
```

## Adapter choice

- `USE_LORA = True`, `LORA_SCALE = 0.5`: requested notebook default using epoch 30.
- `USE_LORA = False`: pristine XL-Base comparison mode.
- `experimental-r32`: the packaged epoch-30 diagnostic adapter. It was fully evaluated but did not pass the release gate, so do not treat it as a final model.
- `best-val` and `final-all-data`: reserved names for a future run that actually passes selection; they are not claimed to exist for this failed experiment.

Use the same prompt and seed when comparing adapters. A different seed changes the composition and makes the comparison less meaningful.

## VRAM handling

For a GPU with at least 20 GB VRAM, begin with:

```python
offload_to_cpu = False
```

For a 12–16 GB GPU or after a CUDA OOM:

```python
offload_to_cpu = True
```

Also keep batch size at 1, disable compilation, close old model objects and restart the runtime after an OOM. Do not silently switch to a different ACE-Step base checkpoint; the LoRA was trained for XL-Base.

## Success criteria

The final inference cell must report:

- adapter loaded successfully;
- one output generated;
- file decodes;
- sample rate `48000`;
- channels `2`;
- duration at least 10 seconds;
- no checksum failure, OOM, NaN or missing-file error.

The notebook displays the generated WAV only after these checks pass.

## Common failures

`401/403 from Hugging Face`: verify that `HF_TOKEN` can read the private repo and that notebook access is enabled in Colab Secrets.

`401/402/429 from OpenRouter`: verify `OPENROUTER_API_KEY`, available credits and rate limits, or set `USE_OPENROUTER_ENHANCER = False`. Do not redownload the model.

`OpenRouter output failed the deterministic gate`: rerun the enhancer cell or make the idea more concrete. ACE-Step is not called when the caption or structured fields fail validation.

`matplotlib ... backend_inline is not a valid value`: use the latest notebook, which forces the headless `MPLBACKEND=Agg` for the ACE-Step subprocess. On an already-running older notebook, run `os.environ["MPLBACKEND"] = "Agg"` and rerun only the inference cell.

`CUDA out of memory`: restart the runtime, set `offload_to_cpu=True`, and rerun from model initialization with batch size 1.

`Audio is chaotic or distorted`: do not add mastering first. Keep prompt/seed/sampling fixed, disable LoRA for a base-model control, then compare LoRA scales 0.25, 0.5 and 1.0. Post-processing changes level/tone but cannot restore missing melody or structure.

`adapter_config.json missing`: confirm the chosen subdirectory is exactly `final-all-data` or `best-val` under the downloaded release.

`checksum mismatch`: delete that release download and fetch the same immutable commit again. Do not infer from a partially downloaded folder.

`ffprobe not found`: run the notebook system-dependency cell, which installs `ffmpeg`.

`XL-Base initialization failed`: confirm both `checkpoints/` from `ACE-Step/Ace-Step1.5` and `checkpoints/acestep-v15-xl-base/` exist, then verify the pinned revisions.

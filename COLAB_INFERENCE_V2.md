# Colab Pro inference for Melodic EDM Core V2

This path is inference-only. It does not install MOSS, load the training tensors or continue LoRA training. Before final retraining completes, the notebook automatically uses the published `best-val` preview. After `final-all-data` is present, it becomes the default while `best-val` remains available for comparison.

## How Colab Pro is authenticated

Consumer Colab Pro runs from the Colab website in a browser. It does not provide a supported CLI that a Vast server can use to submit a job to a hosted Pro runtime. Therefore:

- do not install a Google/Colab CLI on Vast for this workflow;
- do not copy a Google password, browser cookie or Google OAuth refresh token to Vast;
- sign in to the Google account that owns Colab Pro in the browser, open the notebook, select an NVIDIA GPU and run it there;
- put a Hugging Face **read** token in Colab Secrets as `HF_TOKEN` so the notebook can download the private release;
- put an OpenRouter key in Colab Secrets as `OPENROUTER_API_KEY` so the notebook can enhance a free-form idea before inference.

The notebook and inference script are published by GitHub/Hugging Face. The Colab VM pulls the immutable release directly from Hugging Face; Vast does not push a process into Colab.

After this branch is merged, open the notebook from:

```text
https://colab.research.google.com/github/Bangchis/melodic-edm-training-pipeline/blob/main/notebooks/melodic_edm_core_v2_colab.ipynb
```

While the GitHub repository is private, first authorize GitHub access from Colab's **File → Open notebook → GitHub** tab. Once the repository is public, the direct URL works without GitHub authorization.

If unattended server-side submission is required, that is a separate Google Cloud Colab Enterprise workflow. It requires a Google Cloud project, billing, IAM permissions, a runtime template and its own quota; a consumer Colab Pro subscription is not that service.

## Requirements

- A browser signed into the Google account that owns the active Colab Pro subscription.
- A Colab Pro runtime with an NVIDIA GPU. GPU type and availability are assigned dynamically by Colab and are not guaranteed.
- At least 20 GB GPU memory is preferred for XL-Base. A 12–16 GB GPU may work with CPU offload and will be slower.
- Roughly 35–45 GB free disk for the ACE-Step environment, XL-Base checkpoints and the private adapter release.
- A Hugging Face read token stored in Colab Secrets as `HF_TOKEN`.
- An OpenRouter API key stored in Colab Secrets as `OPENROUTER_API_KEY`.

Never paste either token into a notebook cell or commit it to GitHub. In Colab, open the key icon, create `HF_TOKEN` and `OPENROUTER_API_KEY`, and enable notebook access for both. Only the text idea and explicit music conditions are sent to OpenRouter; no audio, adapter or Hugging Face token is sent there.

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
2. Read `HF_TOKEN` and `OPENROUTER_API_KEY` from Colab Secrets into memory without displaying them.
3. Clone ACE-Step at the pinned source revision.
4. Install the official environment with `uv sync`.
5. Download the core ACE-Step checkpoints and pinned XL-Base weights.
6. Resolve the private V2 release to one immutable commit and download exactly that revision.
7. Verify every release file with `SHA256SUMS`.
8. Load `final-all-data` when present, otherwise fall back explicitly to the verified `best-val` preview.
9. Send the free-form text idea to an OpenRouter LLM using strict JSON Schema output.
10. Validate and compile the returned music fields locally, then generate a deterministic 48 kHz stereo WAV.
11. Inspect and play the result inside Colab.

Inference first passes the free-form idea through `scripts/enhance_prompt_openrouter.py`. The default route is `~google/gemini-flash-latest`, configurable with an exact OpenRouter model slug. OpenRouter returns exactly five strict JSON description fields; the included deterministic `prompt_enhancer.py` then enforces a 40–300 word caption. ACE-Step itself runs with `thinking=False`, so the ACE 5 Hz language model is not used for prompt planning. BPM, key, time signature and instrumental section markers remain explicit separate conditions and are never overwritten by the LLM.

```text
free-form idea + explicit BPM/key/time/sections
→ OpenRouter LLM with strict JSON Schema
→ genre + mood + melody + arrangement + production
→ deterministic validator/compiler
→ 40–300 word inference caption
+ BPM/key/time signature/sections
→ ACE-Step XL-Base + selected LoRA
```

The LLM is the enhancer; the deterministic stage is only a safety/format gate. Explicit BPM, key, time signature and sections always overwrite any LLM guess. The notebook records the requested model route, OpenRouter's resolved model name and the complete secret-free conditioning payload in `/content/v2_prompt_enhancement.json` for reproducibility.

## Prompt format

Use 40–300 audible words. A concise 60–150 word prompt is the practical default; 300 is a hard maximum for unusually detailed requests. Describe melody, composition and production; keep BPM/key/time signature in their fields. Do not use artist names or vague quality claims. Training captions remain 40–80 words; only this inference enhancer has the wider limit.

The notebook accepts a free-form idea plus optional fixed conditions:

```python
USER_IDEA = "EDM Trung Hoa không lời với hook pipa dễ nhớ, dizi đối đáp và drop mạnh"
EXPLICIT_CONDITIONS = {
    "bpm": 128,
    "keyscale": "F# minor",
    "timesignature": "4",
    "sections": ["Intro", "Theme", "Build", "Drop", "Break", "Final Drop", "Outro"],
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
    "Instrumental Chinese melodic gaming EDM with an uplifting and adventurous mood. "
    "A memorable two-bar minor-pentatonic pipa hook is answered by airy dizi phrases "
    "and doubled by a bright synth pluck. A short build opens into a four-on-the-floor "
    "drop with wide supersaw chords, clean sub bass, punchy drums and spacious fantasy reverb."
)
bpm = 128
keyscale = "F# minor"
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

- `final-all-data`: recommended default, trained fresh on all 231 records for the scaled number of optimizer steps.
- `best-val`: preserves the exact validation-selected checkpoint from the 196/35 run and is useful for A/B comparison.

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

`401/402/429 from OpenRouter`: verify `OPENROUTER_API_KEY`, available credits and rate limits. Rerun only the enhancer cell; do not redownload the model.

`OpenRouter output failed the deterministic gate`: rerun the enhancer cell or make the idea more concrete. ACE-Step is not called when the caption or structured fields fail validation.

`CUDA out of memory`: restart the runtime, set `offload_to_cpu=True`, and rerun from model initialization with batch size 1.

`adapter_config.json missing`: confirm the chosen subdirectory is exactly `final-all-data` or `best-val` under the downloaded release.

`checksum mismatch`: delete that release download and fetch the same immutable commit again. Do not infer from a partially downloaded folder.

`ffprobe not found`: run the notebook system-dependency cell, which installs `ffmpeg`.

`XL-Base initialization failed`: confirm both `checkpoints/` from `ACE-Step/Ace-Step1.5` and `checkpoints/acestep-v15-xl-base/` exist, then verify the pinned revisions.

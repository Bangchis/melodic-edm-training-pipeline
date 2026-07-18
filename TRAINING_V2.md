# Melodic EDM Core V2 training

V2 is a fresh LoRA training run for ACE-Step 1.5 XL-Base. It preserves the validated 231-record audio catalog and each song's existing prompt/annotation from V1, fuses those song-specific properties with an independent audio-grounded MOSS-Music analysis, and trains from three grounded prompt views per unique audio item.

## Immutable inputs

- Catalog records with validated audio: **231**.
- Catalog split: **196 train / 35 validation / 0 test**.
- Exact audio content: **217 unique / 14 redundant catalog records**.
- Training view: **184 unique train / 33 unique validation**; all 231 records remain available for audit and upload.
- Split unit: `parent_song_id`; records sharing the same source video never cross the boundary.
- Final run: all **217 unique audio contents**, initialized again from the clean base model.
- Base model: `ACE-Step/acestep-v15-xl-base`.
- Base model revision: `220c1166efbdd9583eafcb12eb160594bbfcb241`.
- ACE-Step source revision: `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.
- MOSS annotation model: `OpenMOSS-Team/MOSS-Music-8B-Thinking` revision `2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3`.

MOSS-Music is used only as an audio listener for annotation and checkpoint scoring. It is not part of the ACE-Step model, receives no gradients, and is not included in the training graph.

## Annotation contract

Every record retains one master annotation and three review views in this fixed order:

1. `canonical`: balanced audible summary, 40–80 words.
2. `composition`: melody, harmony, motif, rhythm and arrangement, 25–80 words.
3. `production`: instruments, synths, drums, bass, space and mix, 25–80 words.

Artist names and source titles are rejected from captions. BPM, key and time signature remain structured fields. Song form remains in the instrumental lyrics sidecar.

MOSS prompt revision `audio-blind-v2.2` first receives no title, artist, filename, MIR or prior annotation. This prevents plausible catalog context from anchoring the listener on instruments it has not actually heard. Before tensors are accepted, every named sound-source claim from either the old annotation or the independent analysis is checked by two differently worded full-track passes and an intro/middle/late montage. An exact name is retained when at least two views support it without a strong full-track contradiction; an absent claim is removed. Conflicting evidence keeps the vocabulary token with an explicit `-like` qualifier, such as `pipa-like plucked lead`, rather than either asserting a physical pipa as fact or collapsing the label to a generic `plucked-string-like` phrase.

Compiler revision `openrouter-per-track-salient-audio-fusion-v3.1` then sends two separately hashed text packets for the same `sample_id` to `google/gemini-3.1-flash-lite` through OpenRouter: the old per-track prompt/annotation and the independent waveform analysis. No audio bytes are sent in this step. It preserves distinctive old genre, mood, melody, arrangement and production properties when supported or not contradicted by the audio, while multi-view decisions constrain exact sound-source claims. These decisions are safety constraints, not a keyword checklist: a present claim may be omitted when it is not salient, broad electronic timbre words remain usable, and no global instrument vocabulary is forced across songs. Its deterministic gate rejects newly introduced unverified specific instrument names, embedded BPM, time signature, exact key, quality hype, static-loop cues and generic `standard/classic EDM structure` boilerplate before a caption can enter tensors. Uncertain instruments may be omitted; they are never forced into all captions. A final MOSS listening audit then checks the compiled caption against the audio.

Each record has three training conditions: canonical, composition and production. Every view independently combines useful properties from that song's old prompt with the independent new MOSS audio evidence; composition emphasizes melody and arrangement, while production emphasizes instrumentation and sound design. Preprocessing stores one audio latent and three prompt embeddings per record. Training chooses one of the three uniformly on every load with CFG dropout `0.15`; validation always uses canonical index 0 with CFG dropout `0.0`.

## Fixed LoRA configuration

| Setting | Value |
|---|---:|
| Adapter | LoRA |
| Rank | 32 |
| Alpha | 32 |
| Dropout | 0.1 |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | `5e-5` |
| Optimizer | AdamW |
| Scheduler | cosine |
| Warmup | 25 optimizer steps |
| Precision | BF16 |
| Gradient checkpointing | enabled |
| GPUs | DDP, 2 × RTX 4090 |
| Per-GPU batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 16 |

The strict LoRA gate requires the exact four projection names, verifies that the attention forward path accepts `encoder_hidden_states`, and refuses to start if any non-LoRA model parameter remains trainable. Base DiT weights, VAE, text encoder, ACE 5 Hz LM and MOSS remain frozen/outside training.

DDP accumulation is exact at epoch boundaries. With 184 unique training records, each rank receives 92 microbatches: eleven full groups of eight plus a four-microbatch tail. The final tail batch explicitly re-enables DDP synchronization, then rescales gradients by `8/4` before clipping and stepping so both ranks remain identical and the update magnitude matches a full accumulated batch. The fresh all-data run performs the same proof for 217 unique records: 109 microbatches per rank and a five-microbatch tail rescaled by `8/5`. Validation disables Fabric's distributed sampler so both ranks evaluate the same complete 33-record canonical-caption set without padding a duplicate record; its globally reduced mean is therefore exact.

This choice follows the upstream ACE-Step model guidance: XL-Base is the high-quality, high-diversity model intended for fine-tuning, while the language model is a planner used by the Thinking inference path rather than a component that must be enabled during LoRA training. The upstream LoRA tutorial also warns that automatically generated captions need manual review because inferred BPM/key and descriptive claims may be wrong. Sources: [ACE-Step 1.5 README](https://github.com/ace-step/ACE-Step-1.5), [LoRA training tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md), [prompt and Thinking tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md), and [inference settings](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md).

## Pipeline

All long server jobs are managed by Supervisor.

The resume-safe `edm-v2-orchestrator` waits for both MOSS shards and advances only when each JSON gate reports `pass`. It stops at the first failed gate. Individual commands below remain available for inspection or manual recovery.

```bash
supervisorctl start edm-v2-orchestrator
```

```text
identity- and prior-claim-blind MOSS annotation (2 shards)
→ merge annotations + grouped 196/35 split
→ build sidecars and dataset indexes
→ multi-view named-claim consensus
→ per-track old-prompt + independent-audio fusion compiler (2 shards) + fidelity gate
→ preprocess train shards + validation
→ merge and validate 196/35/231 tensors
→ exact-audio SHA-256 views: 184/33/217 unique tensors via hardlinks
→ 66-step smoke with checkpoint resume
→ one train/validation run, exactly 30 epochs
→ fixed-prompt checkpoint generation at epochs 5/10/15/20/25/30
→ evaluate every candidate at fixed LoRA scale 0.5
→ MOSS listening score + feature checks
→ 3 styles × 5 seeds for LoRA and pristine Base, paired by exact prompt/seed
→ select best optimizer step
→ package + upload best-val preview
→ clean immutable preview download + inference gate
→ fresh all-217-unique-audio run to scaled optimizer steps
→ private upload of all 196 train + 35 validation FLAC records
→ clean immutable download + SHA-256 verification of all 231 audio files
→ package, upload, clean redownload and inference verification
→ final objective audit
→ private completion-evidence backup
```

### Annotation and tensors

```bash
supervisorctl start edm-v2-build-annotations
supervisorctl start edm-v2-upload-metadata
supervisorctl start edm-v2-preprocess-train-0 edm-v2-preprocess-train-1
supervisorctl start edm-v2-preprocess-validation
supervisorctl start edm-v2-merge-tensors
```

The merge gate must report exactly 231 catalog records, 196 train, 35 validation, no parent crossing and three fused annotation views. It also creates hardlinked loader views containing exactly 184 unique train tensors, 33 unique validation tensors and 217 unique all-data tensors. The original 196/35/231 tensor sets remain intact for auditability. Every tensor has one latent plus three fused prompt embeddings.

### Smoke test

```bash
supervisorctl start edm-v2-train-smoke
```

The smoke run reaches optimizer step 65, resumes, then stops at exact step 66. It must demonstrate:

- both GPUs allocated and active;
- finite loss with at least one decrease;
- no OOM, NaN or Inf marker;
- all three prompt indices selected during training;
- canonical-only validation;
- checkpoint save, resume and clean adapter reload;
- rank 32 / alpha 32 / dropout 0.1 and exact q/k/v/o adapter coverage.

### Train/validation and selection

```bash
supervisorctl start edm-v2-sync-checkpoints
supervisorctl start edm-v2-train-main
supervisorctl start edm-v2-evaluate-checkpoints
supervisorctl start edm-v2-score-moss
supervisorctl start edm-v2-select-checkpoint
supervisorctl start edm-v2-upload-evaluation
supervisorctl start edm-v2-package-preview
supervisorctl start edm-v2-upload-preview
supervisorctl start edm-v2-verify-preview
```

Validation, logging and checkpointing occur every five epochs through the full 30-epoch run. After training, each epoch 5/10/15/20/25/30 adapter is evaluated at the single fixed LoRA scale `0.5` with the same prompts and seeds. Scale `0.5` is recorded and packaged as the recommended inference default; the Colab inference UI can still expose it for manual experimentation after release.

After checkpoint selection, the deployable `best-val` adapter, its three fixed audio examples, metrics, scripts and Colab notebook are packaged and uploaded to the private model repository. A clean immutable redownload must pass checksum verification and 48 kHz stereo inference before the fresh all-231 run is allowed to start. This provides an inference-ready preview while final retraining continues.

Selection is not “last checkpoint wins.” The pristine baseline first proves that the sampler can produce coherent melody, structure and clean audio; its specialized prompt-alignment score is recorded as a control rather than used to block the training intended to improve it. Candidate ranking then weights validation loss `30%`, prompt alignment `30%`, melody/structure/audio quality `25%`, output diversity `10%` and a conservative training-feature similarity penalty `5%`. Every trained sample must score at least `3/5` for prompt alignment, and a candidate may not underperform the baseline mean. A clean but off-prompt result is rejected even when its melody, structure and mix score highly. The machine report records that human listening was not completed, so the automated listener is never presented as a human judgment.

The generalization gate is separate from the single fixed checkpoint sample. It generates Chinese melodic EDM, gaming progressive house and cinematic glitch-hop at five deterministic seeds each, using long-form section conditioning. At least four of five outputs per style must score `>=3/5` in all four dimensions with no distortion, collapse, static loop or intelligible vocals. Pristine XL-Base receives the identical 15 prompt/seed pairs, and the report records paired Base-only, LoRA-only, both-pass and both-fail outcomes. MOSS scores are a repeatable triage signal, not proof that a person will like the music; the release remains subject to human listening.

Thinking on/off and Base `shift=1` versus `shift=3` are inference A/B controls after the adapter passes the non-Thinking robustness gate. They are not silently mixed into training and are never used to rescue a failed adapter by changing the judging condition.

### Fresh all-data run

The selected optimizer step is scaled by dataset exposure:

```text
final_steps = round(best_optimizer_step × 217 / 184)
```

```bash
supervisorctl start edm-v2-train-final
supervisorctl start edm-v2-prepare-audio-dataset
supervisorctl start edm-v2-upload-audio-dataset
supervisorctl start edm-v2-verify-audio-dataset
supervisorctl start edm-v2-evaluate-final
```

The final job refuses to resume or overwrite an existing final run. It reloads the pristine XL-Base model, creates a fresh rank-32/alpha-32 LoRA, trains on all 217 unique audio contents with no validation split, and stops at the exact scaled optimizer step. Final fixed-prompt audio must also pass the absolute listening-quality gate before packaging. Its mean prompt-alignment score may not fall more than `0.34` points below the selected best-val checkpoint on the same three prompts, allowing at most one aggregate score-point difference across the three MOSS judgments.

After final training passes, all 231 exact FLAC records are staged without copying or deduplicating them and uploaded to the private dataset `Bangchis/melodic-edm-audio-v2`. The dataset keeps the original grouped `196 train / 35 validation` split, one file per catalog record, a sanitized manifest and `SHA256SUMS`. A separate gate force-downloads the immutable dataset revision, verifies every byte size and SHA-256 digest, then removes the temporary clean copy.

## Outputs

- `outputs/v2/best-val/`: selected adapter from the grouped train/validation run.
- `outputs/v2/final-all-data/final/`: fresh adapter trained on all 217 unique audio contents.
- `outputs/v2/checkpoint-evaluation/selection.json`: selected epoch and optimizer step.
- `outputs/release/melodic-edm-core-v2-preview/`: checksum-verified, inference-ready `best-val` preview published before final retraining.
- `outputs/v2/checkpoint-evaluation/preview_clean_verification_report.json`: immutable preview redownload, adapter-hash match and 48 kHz stereo inference evidence.
- `outputs/v2/final_plan.json`: exact scaling formula and final step count.
- `outputs/release/melodic-edm-core-v2/`: checksum-verified release folder.
- `outputs/release/melodic-edm-core-v2/clean_verification_report.json`: immutable Hub redownload, adapter-hash match and 48 kHz stereo inference evidence.
- Private training checkpoints: `Bangchis/melodic-edm-core-v2-training`.
- Private annotations: `Bangchis/melodic-edm-training-metadata-v2`.
- Private final model: `Bangchis/melodic-edm-core-v2`.
- Private complete audio dataset: `Bangchis/melodic-edm-audio-v2`.

Source audio, stems, cached tensors, optimizer states, secrets and MOSS reasoning are excluded from the final model release.

## Monitoring and recovery

```bash
supervisorctl status | grep edm-v2
tail -f /workspace/melodic_edm_training_pipeline/logs/v2/<job>.log
nvidia-smi
df -h /workspace
```

Annotation and preprocessing jobs are resumable because they write one atomic record at a time. Main training checkpoints contain optimizer/scheduler/scaler/RNG state. The final all-data run is intentionally non-resumable to prove fresh initialization; if interrupted, remove only its incomplete output after inspection and restart it from the base model.

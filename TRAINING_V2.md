# Melodic EDM Core V2 training

V2 is a fresh LoRA training run for ACE-Step 1.5 XL-Base. It preserves the validated 231-song audio set from V1, adds audio-grounded MOSS-Music annotations, and trains with three prompt views per song.

## Immutable inputs

- Catalog records with validated audio: **231**.
- Split run: **196 train / 35 validation / 0 test**.
- Split unit: `parent_song_id`; records sharing the same source video never cross the boundary.
- Final run: all **231** records, initialized again from the clean base model.
- Base model: `ACE-Step/acestep-v15-xl-base`.
- Base model revision: `220c1166efbdd9583eafcb12eb160594bbfcb241`.
- ACE-Step source revision: `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.
- MOSS annotation model: `OpenMOSS-Team/MOSS-Music-8B-Thinking` revision `2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3`.

MOSS-Music is used only as an audio listener for annotation and checkpoint scoring. It is not part of the ACE-Step model, receives no gradients, and is not included in the training graph.

## Annotation contract

Every record has one master annotation and exactly three captions in this fixed order:

1. `canonical`: balanced audible summary, 40–80 words.
2. `composition`: melody, harmony, motif, rhythm and arrangement, 25–80 words.
3. `production`: instruments, synths, drums, bass, space and mix, 25–80 words.

Artist names and source titles are rejected from captions. BPM, key and time signature remain structured fields. Song form remains in the instrumental lyrics sidecar.

Preprocessing stores one audio latent and three prompt embeddings per record. During training, the dataset chooses caption index 0, 1 or 2 uniformly at each load. CFG dropout is `0.15`. Validation always uses canonical index 0 and CFG dropout `0.0`.

## Fixed LoRA configuration

| Setting | Value |
|---|---:|
| Adapter | LoRA |
| Rank | 48 |
| Alpha | 96 |
| Dropout | 0.1 |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | `7.5e-5` |
| Optimizer | AdamW |
| Scheduler | cosine |
| Warmup | 75 optimizer steps |
| Precision | BF16 |
| Gradient checkpointing | enabled |
| GPUs | DDP, 2 × RTX 4090 |
| Per-GPU batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 16 |

The strict LoRA gate requires the exact four projection names, verifies that the attention forward path accepts `encoder_hidden_states`, and refuses to start if any non-LoRA model parameter remains trainable. Base DiT weights, VAE, text encoder, ACE 5 Hz LM and MOSS remain frozen/outside training.

## Pipeline

All long server jobs are managed by Supervisor.

```text
MOSS annotation (2 shards)
→ merge annotations + grouped 196/35 split
→ build sidecars and dataset indexes
→ preprocess train shards + validation
→ merge and validate 196/35/231 tensors
→ 66-step smoke with checkpoint resume
→ one train/validation run
→ fixed-prompt checkpoint generation
→ MOSS listening score + feature checks
→ select best optimizer step
→ fresh all-231 run to scaled optimizer steps
→ package, upload, clean redownload and inference verification
```

### Annotation and tensors

```bash
supervisorctl start edm-v2-build-annotations
supervisorctl start edm-v2-upload-metadata
supervisorctl start edm-v2-preprocess-train-0 edm-v2-preprocess-train-1
supervisorctl start edm-v2-preprocess-validation
supervisorctl start edm-v2-merge-tensors
```

The merge gate must report exactly 231 records, 196 train, 35 validation, no parent crossing and three caption variants. The tensor gate must report exactly 196 train tensors, 35 validation tensors and 231 all-data tensors, each with one latent plus three prompt embeddings.

### Smoke test

```bash
supervisorctl start edm-v2-train-smoke
```

The smoke run reaches optimizer step 65, resumes, then stops at exact step 66. It must demonstrate:

- both GPUs allocated and active;
- finite loss with at least one decrease;
- no OOM, NaN or Inf marker;
- all three prompt indexes selected;
- canonical-only validation;
- checkpoint save, resume and clean adapter reload;
- rank 48 / alpha 96 / dropout 0.1 and exact q/k/v/o adapter coverage.

### Train/validation and selection

```bash
supervisorctl start edm-v2-sync-checkpoints
supervisorctl start edm-v2-train-main
supervisorctl start edm-v2-evaluate-checkpoints
supervisorctl start edm-v2-score-moss
supervisorctl start edm-v2-select-checkpoint
supervisorctl start edm-v2-upload-evaluation
```

Validation, logging and checkpointing occur every five epochs. Every tenth checkpoint is synchronized to the private training repository and later receives the same three fixed prompt/seed audio samples plus MOSS listening evidence.

Selection is not “last checkpoint wins.” Candidate ranking combines validation loss, MOSS audio-grounded listening scores, output diversity and a conservative training-feature similarity penalty. The machine report records that human listening was not completed, so the automated listener is never presented as a human judgment.

### Fresh all-data run

The selected optimizer step is scaled by dataset exposure:

```text
final_steps = round(best_optimizer_step × 231 / 196)
```

```bash
supervisorctl start edm-v2-train-final
supervisorctl start edm-v2-evaluate-final
```

The final job refuses to resume or overwrite an existing final run. It reloads the pristine XL-Base model, creates a fresh rank-48 LoRA, trains on all 231 records with no validation split, and stops at the exact scaled optimizer step.

## Outputs

- `outputs/v2/best-val/`: selected adapter from the grouped train/validation run.
- `outputs/v2/final-all-data/final/`: fresh adapter trained on all 231 records.
- `outputs/v2/checkpoint-evaluation/selection.json`: selected epoch and optimizer step.
- `outputs/v2/final_plan.json`: exact scaling formula and final step count.
- `outputs/release/melodic-edm-core-v2/`: checksum-verified release folder.
- Private training checkpoints: `Bangchis/melodic-edm-core-v2-training`.
- Private annotations: `Bangchis/melodic-edm-training-metadata-v2`.
- Private final model: `Bangchis/melodic-edm-core-v2`.

Source audio, stems, cached tensors, optimizer states, secrets and MOSS reasoning are excluded from the final model release.

## Monitoring and recovery

```bash
supervisorctl status | grep edm-v2
tail -f /workspace/melodic_edm_training_pipeline/logs/v2/<job>.log
nvidia-smi
df -h /workspace
```

Annotation and preprocessing jobs are resumable because they write one atomic record at a time. Main training checkpoints contain optimizer/scheduler/scaler/RNG state. The final all-data run is intentionally non-resumable to prove fresh initialization; if interrupted, remove only its incomplete output after inspection and restart it from the base model.

# V2 rank-32 retrain evaluation — 2026-07-19

## Outcome

The requested fresh retrain and checkpoint evaluation completed. The run fixed the earlier technical audio collapse, but it did **not** improve prompt-following or musical coherence over pristine XL-Base. No checkpoint passed the absolute quality gate, so the pipeline correctly refused to create a misleading `final-all-data` adapter.

Selection result:

```text
status: failed
quality_accepted: false
selected_checkpoint: null
error: no_checkpoint_passed_absolute_listening_quality
```

## Training evidence

- Base: ACE-Step 1.5 XL-Base.
- Fresh LoRA: rank 32, alpha 32, dropout 0.1, scale 0.5 for evaluation.
- Optimization: learning rate `5e-5`, CFG dropout `0.15`, BF16, AdamW cosine schedule, two-GPU DDP, effective batch 16.
- Dataset: 217 unique audio files representing 231 catalog rows; 184 unique train and 33 unique validation files.
- Conditioning: three per-track captions selected randomly in training; canonical caption in validation. Every caption contains the exact row-bound artist/title style-reference prefix.
- Completed: 30 epochs / 360 optimizer steps.
- Lowest validation loss: `0.7474020015` at epoch 20 / step 240.
- Latest validation loss: `0.7624228452` at epoch 30.
- All 512 LoRA tensors were non-zero; both RTX 4090 GPUs reached 100% utilization. Training, checkpoint save/load and resume gates passed.

## Fixed-prompt listening scores

Each checkpoint generated the same three held-out 180-second instrumental prompts at the same seed and LoRA scale 0.5. MOSS-Music scored prompt alignment, melody, structure and technical audio quality independently from 1 to 5.

| Model/checkpoint | Alignment | Melody | Structure | Quality | Audible failures |
|---|---:|---:|---:|---:|---|
| Pristine XL-Base | 3.33 | 3.67 | 3.67 | 4.33 | none |
| Epoch 5 | 2.00 | 2.67 | 2.67 | 4.67 | none |
| Epoch 10 | 1.33 | 1.67 | 2.00 | 4.33 | none |
| Epoch 15 | 2.67 | 2.67 | 3.00 | 4.67 | none |
| Epoch 20 | 2.00 | 2.33 | 2.67 | 4.67 | none |
| Epoch 25 | 2.33 | 2.33 | 2.33 | 4.33 | none |
| Epoch 30 | 1.67 | 2.00 | 2.33 | 5.00 | one static-loop output |

Required gate: every dimension mean at least 3, every individual score at least 2, and no distorted/collapsed/static-loop output. None of the six checkpoints met it. Epoch 15 was the best relative LoRA checkpoint, but alignment and melody both averaged only 2.67 and therefore it was not accepted.

## What the evidence says caused the failure

1. **The old inference-duration bug is no longer the cause.** All samples used 180 seconds for seven sections, and pristine XL-Base passed on the exact same prompt documents and sampler configuration.
2. **The old rank-48 technical collapse is largely fixed.** Rank 32/alpha 32 produced technically clean audio: mean quality was 4.33–5.00, with no distortion, collapse or intelligible vocals. Only epoch 30 produced one static loop.
3. **The remaining failure is conditioning/generalization, not post-processing.** Compared with pristine XL-Base, every LoRA checkpoint reduced alignment and melody. Several generated tracks were coherent and clean but shifted into acoustic/folk material despite explicit electronic-EDM descriptions. Mastering or normalization cannot repair that semantic mismatch.
4. **Validation loss is not a reliable music-quality selector here.** Epoch 20 had the lowest validation loss, but epoch 15 had better perceptual alignment/structure. Epoch 30 had the highest technical-quality mean while also producing the only static loop and very weak alignment.
5. **Training longer is not the whole explanation.** Quality worsened by epoch 30, so overtraining contributes, but epoch 5 was already below XL-Base. The problem begins with the small heterogeneous dataset/conditioning objective rather than only the number of epochs.
6. **Artist/title identity prefixes are a plausible source of conditioning noise, not proven ground truth.** The 184 unique train audios contain many sparse artist/title combinations. Repeating a unique title anchor beside audible descriptions can encourage the adapter to fit corpus identity associations that do not generalize. This is an inference from the controlled Base-versus-LoRA result; the current evidence cannot isolate it from dataset heterogeneity without another ablation run.

## Decision

Do not publish any epoch as the final model and do not retrain on all data from these checkpoints. The scientifically correct output of this run is the failed selection report plus the preserved checkpoints/audio. A future run should be treated as a new experiment and must change the conditioning/training design rather than merely extending epoch count.

Authoritative machine-readable evidence is stored beside this file:

- `training_validation_report.json`
- `baseline_listening_scores.json`
- `generation_report.json`
- `listening_scores.json`
- `selection.json`

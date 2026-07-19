# V2 emergency server handoff — 2026-07-19

## Data already safe off-server

- Public source and Colab documentation: https://github.com/Bangchis/melodic-edm-training-pipeline/tree/agent/training-v2-r32
- Draft PR: https://github.com/Bangchis/melodic-edm-training-pipeline/pull/3
- Private Hugging Face backup: https://huggingface.co/Bangchis/melodic-edm-core-v2-r32-training
- Hugging Face contains LoRA checkpoints for epochs 5, 10, 15, 20, 25 and 30, plus `checkpoints/best_val`.
- Hugging Face contains all 18 fixed evaluation WAV files under `emergency/checkpoint-evaluation/audio`: six checkpoints × three prompts at LoRA scale 0.5.
- Earlier interrupted evaluation audio is retained under `emergency/archived-evaluations`.
- Training metrics, validation state, prompt-selection counts, GPU evidence and V2 configuration are under `emergency/evidence` and `emergency/configs-v2`.

No token or secret is stored in GitHub, this document, or the uploaded evidence.

## Training state

- Base: ACE-Step 1.5 XL-Base.
- LoRA: rank 32, alpha 32, dropout 0.1; q/k/v/o projections in self- and cross-attention.
- Learning rate: `5e-5`; CFG dropout: `0.15`; effective batch: 16; BF16 DDP on two RTX 4090 GPUs.
- Train-validation run completed all 30 epochs and 360 optimizer steps.
- Validation was measured at epochs 5, 10, 15, 20, 25 and 30.
- Lowest validation loss: `0.7474020015` at epoch 20 / optimizer step 240.
- Latest validation loss: `0.7624228452` at epoch 30.
- Every one of the 231 catalog rows has three captions. All 693 captions begin with the exact artist/track style-conditioning sentence bound to that row.

The lowest validation loss does not by itself make epoch 20 the final selected checkpoint. The 18 generated files still need listening scores and the absolute quality/alignment gate.

## Final evaluation state

MOSS completed all 18 fixed checkpoint scores. The formal selector returned:

```text
status: failed
quality_accepted: false
selected_checkpoint: null
error: no_checkpoint_passed_absolute_listening_quality
```

Pristine XL-Base averaged 3.33 alignment, 3.67 melody, 3.67 structure and 4.33 technical quality on the same three prompts. The best relative LoRA checkpoint was epoch 15 at 2.67 / 2.67 / 3.00 / 4.67, which remained below the absolute gate. Epoch 30 produced the only static-loop failure.

The final all-data adapter does not exist. This is intentional: the pipeline refused to amplify a train-validation adapter that underperformed XL-Base. Full machine-readable evidence and the diagnosis are in `reports/v2-retrain-2026-07-19/` on GitHub. A future retrain must be a newly approved experiment with a changed conditioning/training design, not a continuation of these checkpoints.

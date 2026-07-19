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

## Work that was running when this handoff was written

`edm-v2-score-moss` was scoring the 18 fixed checkpoint outputs. If the old instance survives, `edm-v2-orchestrator` continues automatically. If it is destroyed, restore the repository and HF backup on a new GPU server, then resume from checkpoint listening rather than retraining the completed 30-epoch train-validation run.

Required continuation:

1. Run MOSS listening scoring for the 18 fixed outputs.
2. Select the quality-gated checkpoint across epochs 5/10/15/20/25/30.
3. Run paired five-seed LoRA-versus-XL-Base robustness evaluation across all three held-out prompt families.
4. If the robust quality gate passes, scale the selected optimizer step to all 217 unique audio records and retrain a fresh rank-32 LoRA from XL-Base.
5. Evaluate the final adapter, upload the final model/audio/metrics, and publish the final Colab inference package.

The final all-data adapter does not exist yet. Do not label the current train-validation checkpoint set as the final model.

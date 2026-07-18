# Training V2 status

Last reconciled: 2026-07-18 (Asia/Ho_Chi_Minh)

## Current state

- V1 is complete and remains unchanged.
- V2 reuses 231 validated audio records with a grouped 196-train/35-validation split. No audio is redownloaded or deduplicated.
- The previous rank-48/alpha-96 run was stopped and preserved under `outputs/v2-r48-failed-20260718T0841Z`; it is not treated as a releasable final model.
- Diagnosis proved that the chaotic Colab samples were not enough to declare training failure. The old inference profile (`50 steps`, ADG off, DCW on, LoRA scale 1) made pristine XL-Base fail too. The corrected XL-Base profile is `64 steps`, guidance `8`, shift `1`, ADG on and DCW off.
- With that corrected profile, the archived rank-48 epoch-45 adapter at LoRA scale `0.5` scored 5/5 for prompt alignment, melody, structure and audio quality on the fixed gaming prompt. Per the updated training decision, V2 uses `0.5` as its only checkpoint-evaluation scale.
- The annotation-to-audio/tensor linkage is exact for 231/231 records, but the first stratified listening audit found unsupported audible claims, including pipa/guzheng on `myomouse__009` and strings/brass on `diversity__001`.
- Rank-32 training has **not started**. The first conservative caption-repair pass was stopped before applying anything because a single MOSS audit contradicted MOSS's earlier instrument identification on the same audio. No existing annotation or tensor was replaced.
- Caption checking is being changed to claim-level multi-view consensus: two differently worded full-track checks plus an intro/middle/late montage. Exact names such as pipa, dizi or guzheng are preserved when at least two views support them without a strong full-track contradiction; only absent or unresolved claims are removed or softened.
- Root cause in the previous annotation prompt was confirmation bias: MOSS received title/artist context and the prior instrument list before listening, so it could echo `pipa/guzheng/dizi` instead of independently identifying them. Prompt revision `audio-blind-v2.2` now withholds identity, MIR and prior claims; all 231 MOSS supplements must be regenerated under that revision.
- The five-record consensus pilot validated the policy. `diversity__001` retained three verified electronic claims and rejected only strings/brass; `xu_mengyuan__001` retained pipa as present; conflicted rows remained uncertain rather than being automatically erased.
- The Vast workspace currently has about 182 GiB free.

## Locked replacement configuration

- ACE-Step 1.5 XL-Base, fixed LoRA training.
- Rank `32`, alpha `32`, dropout `0.1`.
- `q_proj`, `k_proj`, `v_proj`, `o_proj` across self- and cross-attention.
- Learning rate `5e-5`, AdamW, cosine, 25 optimizer-step warmup.
- BF16, gradient checkpointing, DDP on 2 × RTX 4090.
- Batch 1/GPU, accumulation 8, effective batch 16.
- CFG dropout `0.15`; random canonical/composition/production embedding during training.
- Maximum 20 epochs; checkpoint/validation/evaluation at epochs 5, 10, 15 and 20.
- Absolute MOSS gate: every dimension mean at least 3/5 and every individual score at least 2/5.
- Every candidate is evaluated only at LoRA scale `0.5` using identical prompts/seeds.

## Active work

1. Regenerate all 231 MOSS supplements with the identity- and prior-claim-blind prompt.
2. Finish multi-view verification of exact audible instrument claims.
3. Compile prompt-useful captions from those decisions, preserving verified specific names, and apply only a fully validated 231-record repair set while keeping a full backup.
4. Rebuild the single fused canonical prompt embedding for every record and rerun exact annotation–tensor alignment checks.
5. Rerun the stratified audio-grounded caption fidelity gate.
6. Run a fresh 66-step rank-32 DDP smoke with save/resume/reload checks.
7. Train once through epoch 20 and evaluate epochs 5/10/15/20 at all three LoRA scales.
8. Select only a checkpoint/scale that passes the absolute quality gate, publish the private `best-val` preview, then fresh-retrain all 231 records to the scaled optimizer step.
9. Score final audio, upload model plus complete 196/35 audio dataset to private Hugging Face repositories, verify immutable redownloads, then update GitHub and Colab documentation.

## Safety

`/workspace` is not a persistent volume. Do not destroy the Vast instance before repaired annotations, rebuilt tensors, checkpoints, adapters, metrics and fixed audio examples have been backed up. Secrets, optimizer state, MOSS reasoning and browser credentials remain excluded from GitHub/model packages.

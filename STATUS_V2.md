# Training V2 status

Last reconciled: 2026-07-19 (Asia/Ho_Chi_Minh)

## Current state

- V1 is complete and remains unchanged.
- V2 preserves 231 validated catalog records with a grouped 196-train/35-validation split. Exact SHA-256 audit found 217 unique audio contents; training uses deduplicated 184-train/33-validation hardlink views, while all 231 records remain available for audit and Hugging Face upload.
- The previous rank-48/alpha-96 run was stopped and preserved under `outputs/v2-r48-failed-20260718T0841Z`; it is not treated as a releasable final model.
- Diagnosis proved that the first chaotic Colab samples mixed an inference/conditioning failure with model quality. Correcting the inference path removed collapse, but a later 15-sample, multi-seed comparison still produced only 6/15 passes for LoRA and 6/15 for pristine Base. Therefore the previous adapter is not accepted as a quality success.
- With that corrected profile, the archived rank-48 epoch-45 adapter at LoRA scale `0.5` scored 5/5 for prompt alignment, melody, structure and audio quality on the fixed gaming prompt. Per the updated training decision, V2 uses `0.5` as its only checkpoint-evaluation scale.
- The annotation-to-audio/tensor linkage is exact for 231/231 records, but the first stratified listening audit found unsupported audible claims, including pipa/guzheng on `myomouse__009` and strings/brass on `diversity__001`.
- Rank-32 retraining has **not started**. The previous rank-32 tensors did not contain the intended old-prompt + MOSS fusion: they were mostly generic MOSS-only descriptions. Static audit found 89/231 captions with repetitive/cyclical wording, 134/231 with only generic synth/bass/drum descriptions, 18 obvious section-order label errors, and 14 redundant catalog records. This is the primary retraining reason; simply enabling Thinking or adding epochs would not repair it.
- Caption checking is being changed to claim-level multi-view consensus: two differently worded full-track checks plus an intro/middle/late montage. Exact names such as pipa, dizi or guzheng are preserved when at least two views support them without a strong full-track contradiction; only absent or unresolved claims are removed or softened.
- Root cause in the previous annotation prompt was confirmation bias: MOSS received title/artist context and the prior instrument list before listening, so it could echo `pipa/guzheng/dizi` instead of independently identifying them. Prompt revision `audio-blind-v2.2` withholds identity, MIR and prior claims; its 231 supplements are now the independent evidence input to the stricter claim verifier.
- The five-record consensus pilot validated the policy. `diversity__001` retained three verified electronic claims and rejected only strings/brass; `xu_mengyuan__001` retained pipa as present; conflicted rows remained uncertain rather than being automatically erased.
- Claim verifier revision `multi-view-audio-claims-v2.6` additionally rejects contradictory responses such as verdict `present` with evidence saying the instrument is absent, and only migrates old cache entries that pass the stricter parser.
- Official ACE-Step guidance treats XL-Base as the fine-tuning model and describes Thinking as an inference planner, not a required training switch. The fixed trainer's timestep sampling does not use inference `shift`; `shift 1` versus `shift 3` and Thinking on/off will be controlled A/B tests after retraining, using identical prompts and seeds.
- The Vast workspace currently has about 182 GiB free.

## Locked replacement configuration

- ACE-Step 1.5 XL-Base, fixed LoRA training.
- Rank `32`, alpha `32`, dropout `0.1`.
- `q_proj`, `k_proj`, `v_proj`, `o_proj` across self- and cross-attention.
- Learning rate `5e-5`, AdamW, cosine, 25 optimizer-step warmup.
- BF16, gradient checkpointing, DDP on 2 × RTX 4090.
- Batch 1/GPU, accumulation 8, effective batch 16.
- CFG dropout `0.15`; random canonical/composition/production embedding during training.
- Train exactly 30 epochs; checkpoint and validation at epochs 5, 10, 15, 20, 25 and 30. Generate listening samples only after training completes.
- Absolute MOSS gate: every dimension mean at least 3/5 and every individual score at least 2/5.
- Every candidate is evaluated only at LoRA scale `0.5` using identical prompts/seeds.

## Active work

1. Finish the active two-GPU multi-view verification of exact audible instrument claims for all 231 records.
2. Pilot, then compile three conflict-free captions from each song's old prompt plus independent MOSS evidence; apply only a fully validated 231-record repair set while keeping a full backup.
3. Build the three fused prompt embeddings for every record and rerun exact annotation–tensor alignment checks.
4. Rerun the stratified audio-grounded caption fidelity gate.
5. Run a fresh 66-step rank-32 DDP smoke with save/resume/reload checks.
6. Train once through epoch 30 and evaluate epochs 5/10/15/20/25/30 only at the fixed LoRA scale `0.5`.
7. Select only a checkpoint/scale that passes the absolute quality gate, publish the private `best-val` preview, then fresh-retrain all 217 unique audio contents to the scaled optimizer step.
8. Score final audio, upload model plus complete 196/35 audio dataset to private Hugging Face repositories, verify immutable redownloads, then update GitHub and Colab documentation.

## Safety

`/workspace` is not a persistent volume. Do not destroy the Vast instance before repaired annotations, rebuilt tensors, checkpoints, adapters, metrics and fixed audio examples have been backed up. Secrets, optimizer state, MOSS reasoning and browser credentials remain excluded from GitHub/model packages.

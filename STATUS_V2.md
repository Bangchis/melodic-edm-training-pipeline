# Training V2 status

Last reconciled: 2026-07-18 (Asia/Ho_Chi_Minh)

## Current state

- V1 is complete and remains unchanged.
- V2 runs from branch `agent/training-v2-r48` and a separate ACE-Step worktree.
- The 231 validated V1 audio records are reused; no audio is redownloaded and no record is deduplicated.
- MOSS annotation, three-caption merge, grouped 196/35 split and preprocessing of all 231 records have passed their strict gates.
- MOSS is an annotation/listening model only. It is not loaded by the ACE-Step trainer and is not updated.
- The 66-step DDP smoke, checkpoint save/resume and clean adapter reload have passed.
- The single train/validation run is active. At the latest reconciliation it completed epoch 35; the best validation state remained epoch 20, optimizer step 260, loss `0.7067630870`, with three stale checks.
- Epoch 10, 20 and 30 resumable checkpoints plus metrics are confirmed in the private Hugging Face training repository.
- Both RTX 4090 GPUs remain active without OOM/NaN. The Vast workspace has about 186 GiB free.

## Completed V2 implementation

- Pinned MOSS source/model installation in a separate virtual environment.
- Real MOSS audio smoke inference.
- Exact three-caption schema and validator.
- Atomic/resumable two-shard MOSS annotation.
- Merged annotation builder with immutable audio/base-annotation hashes.
- Exact grouped split solver for 196 train / 35 validation / 0 test.
- Dataset builder with one audio latent and three prompt embeddings per record.
- Strict rank-48 q/k/v/o LoRA scope gate.
- Uniform train caption selection, canonical-only validation and CFG semantics.
- Exact optimizer-step stop, metric history, prompt counters, VRAM monitoring and resume state.
- A 66-step DDP smoke plan with one-step checkpoint resume.
- Main train/validation, checkpoint sync, fixed-sample generation and MOSS listening jobs.
- Best-step selection and fresh all-231 step scaling.
- A pre-final `best-val` package/upload/clean-inference gate so the selected adapter can be tested before all-data retraining starts.
- Private Hugging Face packaging/upload and clean-redownload verification jobs.
- Detailed training guide, model card and Colab Pro inference notebook.
- Forty-three focused pipeline tests pass locally and on Vast.

## Next automatic gates

1. Finish the single train/validation run by early stopping or epoch 150 and finalize every tenth private checkpoint upload.
2. Generate the same three fixed prompt/seed examples for all tenth checkpoints plus best/last, then run MOSS listening and feature checks.
3. Select the optimizer step from validation, alignment, melody, structure, audio quality, diversity and similarity evidence.
4. Publish `best-val` to the private model repository and require a clean immutable redownload plus 48 kHz stereo inference before continuing.
5. Fresh-train all 231 records to `round(best_step × 231 / 196)`.
6. Package/upload both adapters, redownload the final immutable revision and pass clean inference.
7. Run the objective audit, back up completion evidence and update GitHub with final hashes, metrics and links.

## Safety

`/workspace` is not persistent across instance destruction. Do not close the Vast instance until final metadata, checkpoints, adapters, reports and fixed audio examples have been confirmed on private Hugging Face repositories. Secrets, source audio, stems, tensors and MOSS reasoning are excluded from GitHub/model releases.

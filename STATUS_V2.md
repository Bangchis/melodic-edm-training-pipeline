# Training V2 status

Last reconciled: 2026-07-18 (Asia/Ho_Chi_Minh)

## Current state

- V1 is complete and remains unchanged.
- V2 runs from branch `agent/training-v2-r48` and a separate ACE-Step worktree.
- The 231 validated V1 audio records are reused; no audio is redownloaded and no record is deduplicated.
- Two MOSS-Music-8B-Thinking annotator shards are running under Supervisor, one per RTX 4090.
- MOSS is an annotation/listening model only. It is not loaded by the ACE-Step trainer and is not updated.
- A stricter validator now rejects missing/zero overall MOSS confidence and incomplete audible-fact schemas. Atomic valid files are retained; rejected files are retried.
- At the latest checkpoint the Vast workspace had about 193 GiB free.

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
- Private Hugging Face packaging/upload and clean-redownload verification jobs.
- Detailed training guide, model card and Colab Pro inference notebook.
- Six focused V2 pipeline unit tests pass locally and on Vast.

## Next automatic gates

1. Finish all 231 MOSS supplements with no schema errors.
2. Merge annotations and verify exact three-caption coverage plus 196/35 grouped split.
3. Preprocess and validate 196 train, 35 validation and 231 all-data tensors.
4. Run the 66-step smoke; do not start main training unless it passes.
5. Run one fixed train/validation configuration, score fixed checkpoints and select the best optimizer step.
6. Fresh-train all 231 records to `round(best_step × 231 / 196)`.
7. Package and upload both adapters, then redownload at an immutable commit and pass 48 kHz stereo inference.
8. Publish the V2 pipeline/docs to GitHub and update this status with final hashes, metrics and links.

## Safety

`/workspace` is not persistent across instance destruction. Do not close the Vast instance until final metadata, checkpoints, adapters, reports and fixed audio examples have been confirmed on private Hugging Face repositories. Secrets, source audio, stems, tensors and MOSS reasoning are excluded from GitHub/model releases.

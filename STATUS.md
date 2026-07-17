# Current server status

Last reconciled: 2026-07-18 (Asia/Ho_Chi_Minh)

## Verified input and audio selection

- Catalog records: 240.
- Records with valid downloaded audio: 236.
- Physical raw FLAC files: 221; all 221 passed `ffprobe` decoding checks.
- Accepted training records: 231, with 231 unique `sample_id` values and 231 valid paths.
- Rejected after vocal cleanup: 5. There are no pending selections.
- Selected sources: 170 original, 52 full instrumental stems, 3 clean instrumental
  stems and 6 checked clean sections.
- Deduplication is disabled. Repeated catalog records remain separate samples.

The four catalog rows without downloaded audio are outside the 236 valid records:

- MyoMouse — Viet Nam
- MyoMouse / performance collaboration — Fight! (performance)
- MGD & StarlingEDM — Z-PERADOR
- 4sta. & StarlingEDM — What Do You Feel?

The five audio records rejected after repeated vocal cleanup are:

- TheFatRat feat. Laura Brehm — Monody (`thefatrat__002`)
- TheFatRat feat. Laura Brehm — The Calling (`thefatrat__003`)
- TheFatRat, Slaydit & Anjulie — Stronger (`thefatrat__005`)
- TheFatRat & AleXa — Rule the World (`thefatrat__016`)
- TheFatRat & RIELL — Hiding in the Blue (`thefatrat__018`)

These five were rejected for persistent intelligible lyrics and no validated clean
section, not because they were duplicates.

## Active stage

MIR analysis is running as two resumable Supervisor jobs, one partition per RTX 4090.
It will produce BPM, beats/downbeats, section boundaries, key and confidence for all
231 accepted samples. The temporary demixing directory is expected to grow before
the per-track JSON files appear.

During the first MIR launch, a basename collision was caught before any result JSON
was written: many separated files are named `instrumental.flac`. MIR inputs now use
unique `<sample_id>.flac` hard links, so All-In-One cannot overwrite analysis files
between records. The incomplete temporary pass was discarded and restarted.

## Completed infrastructure

- ACE-Step 1.5 repository pinned at commit
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.
- XL-Base, VAE and Qwen embedding checkpoints downloaded on the server.
- GPU audio-separation and MIR environments installed on the server.
- ACE-Step training patch supports four caption variants per record, deterministic
  validation captions, periodic validation, global DDP validation loss, `best_val`
  checkpointing and early stopping.
- Focused tests: 7 passed. Full ACE-Step training-v2 tests: 37 passed.
- The fixed training config is XL-Base LoRA rank 32 / alpha 64 / dropout 0.1,
  learning rate 1e-4, effective batch 16, CFG dropout 0.15 and two-GPU DDP.

## Remaining sequence

1. Verify exactly 231 MIR JSON files and audit outliers/errors.
2. Generate and validate 231 master annotations plus four caption variants.
3. Build 48 kHz stereo triplets and an 85/15 grouped train/validation split.
4. Preprocess tensors on both GPUs and verify exact counts.
5. Run one-epoch smoke test, then one 150-epoch main training run.
6. Compare middle, best-validation and final checkpoints using fixed prompts.
7. Package code/LoRA/config/examples; upload safe artifacts; test a clean download
   and inference before the Vast instance is destroyed.

Do not upload raw/separated audio, dataset tensors, cookies or secrets.

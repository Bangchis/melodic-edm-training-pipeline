# Current server status

Last reconciled: 2026-07-18 (Asia/Ho_Chi_Minh)

## Verified audio corpus

- Catalog records: 240.
- Records with valid downloaded audio: 236.
- Accepted training records: 231, with 231 unique `sample_id` values and 231
  decodable 48 kHz stereo training paths.
- Selected sources: 170 original, 52 full instrumental stems, 3 clean instrumental
  stems and 6 checked clean sections.
- Deduplication is disabled. Repeated catalog records remain separate, equally
  weighted samples and are grouped into the same split when they share source audio.
- Rejected after vocal cleanup: 5. These were rejected for persistent intelligible
  lyrics and no validated clean section, not because they were duplicates.

The four catalog rows without downloaded audio are outside the 236 valid records:

- MyoMouse — Viet Nam
- MyoMouse / performance collaboration — Fight! (performance)
- MGD & StarlingEDM — Z-PERADOR
- 4sta. & StarlingEDM — What Do You Feel?

The five persistent-vocal rejects are:

- TheFatRat feat. Laura Brehm — Monody (`thefatrat__002`)
- TheFatRat feat. Laura Brehm — The Calling (`thefatrat__003`)
- TheFatRat, Slaydit & Anjulie — Stronger (`thefatrat__005`)
- TheFatRat & AleXa — Rule the World (`thefatrat__016`)
- TheFatRat & RIELL — Hiding in the Blue (`thefatrat__018`)

## Completed MIR and annotation gates

- MIR coverage is complete: 231 expected, 231 files and 231 validated records.
- Eleven clear EDM half-time estimates were normalized to their double-tempo value;
  the final MIR report passes with no missing record.
- Master annotations are complete: 231 accepted records and 231 annotation files.
- Provider mix: 175 Gemini annotations and 56 pinned local Qwen2.5-Omni annotations.
  Three malformed-schema cases passed after a supervised 90-second representative
  audio-window retry; all other local windows used up to 240 seconds.
- Every record has exactly four variants (`full`, `composition`, `production`,
  `tags`). The full variant exactly equals its canonical caption.
- Final canonical captions range from 43 to 76 words (mean 59.06).
- Section captions cover the MIR labels, contain 5–40 words, and are semantically
  distinct after label boilerplate is removed.
- Artist names, BPM, exact key names, hype/quality claims and use-case language are
  rejected or sanitized from training captions.
- Final annotation gate: `status=pass`, 0 errors and 0 warnings.

## Active stage

`edm-train-main` is running the single fixed two-GPU LoRA configuration. The
one-epoch DDP smoke gate passed before main training was allowed to start.

The latest verified resumable main checkpoint is epoch 10 / global step 130:

- Train loss: 0.7271 (down from 1.4926 at epoch 1).
- Validation loss: 0.7541, current best at epoch 10.
- Training state contains optimizer and scheduler state.
- Both the epoch and `best_val` adapters contain 512/512 finite, nonzero tensors.
- Main output size was 722 MiB and the instance had about 221 GiB free.
- A fresh private metadata backup completed with 1,200 files and zero secret
  findings; audio, tensors, model checkpoints and tokens were excluded.

Dataset construction and preprocessing are complete:

- Final triplets: 231/231, all decodable 48 kHz stereo with matching JSON and
  `.lyrics.txt` files.
- Grouped split: 196 train and 35 validation; no test set and no shared-source leak.
- Fifteen repeated catalog records were preserved rather than deduplicated.
- Train preprocess shards: 98/98 and 98/98, balanced at about 5.57 hours each.
- Deep tensor gate: 196 train + 35 validation = 231/231 readable tensors, 0 errors.
- The merged train loader manifest resolves all 196 real shard tensors without
  copying or deduplicating them.

At the final tensor checkpoint the instance had about 222 GiB free. `/workspace` is
not a persistent volume, so metadata is backed up to the private Hugging Face dataset
at major gates before later destructive instance actions.

## Completed infrastructure

- Pipeline code, tests, docs and trainer patch are backed up in the private GitHub
  repository `Bangchis/melodic-edm-training-pipeline`; completion work remains in
  draft PR #1.
- ACE-Step 1.5 is pinned at commit
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.
- XL-Base, VAE, Qwen embedding checkpoints and the pinned Qwen2.5-Omni annotator are
  downloaded on the server only.
- Focused pipeline tests: 24 passed locally and on Vast. Full ACE-Step training-v2
  tests: 37 passed.
- The fixed training config is XL-Base LoRA rank 32 / alpha 64 / dropout 0.1,
  learning rate 1e-4, effective batch 16, CFG dropout 0.15 and two-GPU DDP.
- Smoke passed after 13 optimizer steps: finite train loss 1.3718, validation loss
  1.3535, both GPUs observed at 100% utilization, 512/512 nonzero adapter tensors,
  and a successful clean XL-Base adapter reload.
- Smoke and main training gates verify both GPUs, finite loss, readable/non-empty
  adapters, resumable state and the real nested PEFT adapter layout.

## Remaining sequence

1. Run the single fixed 150-epoch LoRA training configuration with validation,
   best-checkpoint selection and early stopping.
2. Compare middle, best-validation and final checkpoints with three fixed prompts.
3. Package safe code/LoRA/config/examples, upload private artifacts, redownload into
   a clean directory and verify inference.

Do not upload raw/separated audio, dataset tensors, cookies or secrets.

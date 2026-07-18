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
- MOSS-Music was not used, is not loaded, and is not part of ACE-Step training or
  the release.
- Every record has exactly four variants (`full`, `composition`, `production`,
  `tags`). The full variant exactly equals its canonical caption.
- Final canonical captions range from 43 to 76 words (mean 59.06).
- Section captions cover the MIR labels, contain 5–40 words, and are semantically
  distinct after label boilerplate is removed.
- Artist names, BPM, exact key names, hype/quality claims and use-case language are
  rejected or sanitized from training captions.
- Final annotation gate: `status=pass`, 0 errors and 0 warnings.

## Completed training and release

`edm-train-main` completed the single fixed two-GPU LoRA configuration and exited
normally after early stopping at epoch 70 / global step 910. The one-epoch DDP smoke
gate passed before main training was allowed to start.

The latest verified resumable checkpoint is epoch 70; the selected release adapter
is the best-validation checkpoint from epoch 45:

- Epoch-70 train loss: 0.7344 (down from 1.4926 at epoch 1).
- Epoch-70 validation loss: 0.7282. Best validation loss: 0.6984 at epoch 45.
- Early stopping triggered after five validation checks without improvement.
- Training state contains optimizer and scheduler state.
- The middle, `best_val` and final adapters each contain 512/512 finite, nonzero
  tensors; the final adapter hash exactly matches the epoch-70 adapter.
- The main training validator reports `status=pass`, and both RTX 4090s were
  observed at 100% utilization.
- The final private metadata backup completed with 1,201 files at commit `6d4f061`
  and zero secret findings; audio, tensors, model checkpoints and tokens were
  excluded.
- A 252 MB resumable epoch-70 checkpoint was uploaded to the private dataset
  `Bangchis/melodic-edm-training-resume` at commit `349b0ac`; it contains the LoRA,
  optimizer/scheduler state and safety metadata, with zero secret findings and no
  audio or preprocessed tensors.
- Fixed-prompt comparison produced 9/9 distinct, fully decodable 30-second WAVs:
  three prompts across middle, best-validation and final checkpoints.
- The 19-file `best_val` release passed its secret scan and every SHA-256 check, then
  uploaded to the private model repo `Bangchis/melodic-edm-core-v1` at commit
  `4ca7240`.
- A clean 20-file Hub snapshot (19 release files plus `.gitattributes`) was downloaded,
  its adapter hash matched the packaged adapter, every checksum passed, and clean
  inference produced a fully decodable 30-second 48 kHz stereo WAV.
- The instance had about 217 GiB free after clean release verification.

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
- Focused pipeline tests: 26 passed on Vast. Full ACE-Step training-v2 tests:
  38 passed plus 8 subtests.
- The fixed training config is XL-Base LoRA rank 32 / alpha 64 / dropout 0.1,
  learning rate 1e-4, effective batch 16, CFG dropout 0.15 and two-GPU DDP.
- Smoke passed after 13 optimizer steps: finite train loss 1.3718, validation loss
  1.3535, both GPUs observed at 100% utilization, 512/512 nonzero adapter tensors,
  and a successful clean XL-Base adapter reload.
- Smoke and main training gates verify both GPUs, finite loss, readable/non-empty
  adapters, resumable state and the real nested PEFT adapter layout.

## Remaining operational choices

The required pipeline sequence is complete. Optional user actions are to listen to
the three packaged examples, merge draft PR #1, and stop the Vast instance when it
is no longer needed. Do not destroy the instance until the private Hub/GitHub copies
are confirmed accessible from the user's account. Raw/separated audio, dataset
tensors, cookies and secrets remain server-only and were not published.

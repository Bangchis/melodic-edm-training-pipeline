# Runbook — record-preserving training pipeline

Project root on Vast:

`/workspace/melodic_edm_training_pipeline`

## 1. Prepare all downloaded records

This validates every catalog record and preserves every downloaded file.

```bash
python3 scripts/prepare_audio.py \
  --checkpoint /workspace/edm_audio_v5/state/checkpoint.csv \
  --selection /workspace/instrumental_edm_catalog_v5/catalog/selection_pre_server.csv \
  --project-root /workspace/melodic_edm_training_pipeline \
  --workers 4
```

Outputs:

- `data/canonical/<sample_id>.flac`: 48 kHz stereo FLAC.
- `data/annotation_preview/<sample_id>.mp3`: 44.1 kHz stereo 128 kbps.
- `data/audio_manifest.jsonl`.
- `data/rejected_downloads.jsonl`.

## 2. Vocal classification and separation

Requires `OPENROUTER_API_KEY` in `/workspace/.env`. The key must never be committed.
Use `google/gemini-3.1-flash-lite`, temperature 0 and strict JSON output. Keep natural
instrumentals and vocal chops; separate only clear lyrics. Every valid catalog record
continues downstream as its own sample, even when two records contain identical audio.

```bash
python3 scripts/classify_vocals.py
```

The classifier checkpoints `data/vocal_manifest.jsonl` after every record and resumes
without calling the API again for completed samples.

## 3. MIR and annotation

Analyze BPM, beats/downbeats, sections and key. Uncertain optional fields remain
empty. Annotate against the fixed taxonomy in `configs/taxonomy.json`; artist names
must not appear in canonical captions.

MIR is resumable and split deterministically across the two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/mir/bin/python scripts/analyze_mir.py \
  --project-root "$PWD" --part-index 0 --num-parts 2 --device cuda
CUDA_VISIBLE_DEVICES=1 .venvs/mir/bin/python scripts/analyze_mir.py \
  --project-root "$PWD" --part-index 1 --num-parts 2 --device cuda
```

Both processes may first demix their complete partition before `data/mir/*.json`
starts appearing. A running Demucs child with GPU utilization is valid progress.
Do not start annotation until there is one valid MIR JSON for every accepted row in
`data/training_audio_manifest.jsonl`.

Run the exact-coverage validator before annotation:

```bash
python3 scripts/validate_mir.py --project-root "$PWD"
```

It must report `status=pass`, `expected_records=231`, `mir_files=231` and
`validated_records=231`. Low-confidence key or time-signature omissions are warnings;
missing/invalid BPM, beats, downbeats or sections are hard errors.

Annotation uses one master annotation plus exactly four prompt variants (`full`,
`composition`, `production`, `tags`) for each record. The full variant equals the
40–80 word canonical caption. BPM, key, time signature and artist names are excluded
from captions. Each training epoch randomly chooses one variant without multiplying
the number or weight of samples.

```bash
python3 scripts/annotate_openrouter.py \
  --project-root "$PWD" --env-file /workspace/.env
```

On a fresh provider/model combination, first run the Supervisor
`edm-annotate-smoke` job. It annotates exactly one record through the same state file;
verify the schema/caption gate, then start `edm-annotate`, which resumes with the
remaining records.

Only records with confidence at least 0.70 and a schema-valid annotation pass. Retry
failures; resolve `data/manual_review.csv` before building the final dataset.

## 4. ACE-Step dataset

Build one audio/JSON/lyrics triplet per accepted catalog `sample_id`. Do not collapse
records that share the same video or audio fingerprint.
Samples over 240 seconds must be cut on musical boundaries. Validate every triplet
before preprocessing.

The train/validation split is grouped by shared source audio. Thus two catalog rows
that share audio are both retained, but cannot leak across train and validation.

```bash
python3 scripts/build_acestep_dataset.py --project-root "$PWD"
```

On Vast, run the equivalent long step through Supervisor as
`edm-build-dataset`.

Required gate: final triplet count equals accepted annotation count, every FLAC is
48 kHz stereo and decodes, every JSON has a non-empty caption and four variants, and
every audio has its matching `.json` and `.lyrics.txt`.

## 5. ACE-Step preprocess/train

Pin the ACE-Step repository commit, inspect `python train.py fixed --help`, split
triplets across both GPUs, preprocess in parallel, merge tensors, run a one-epoch
DDP smoke test and only then run the configured 150-epoch LoRA training.

Training is one fixed configuration only: XL-Base, LoRA rank 32/alpha 64,
dropout 0.1, learning rate 1e-4, effective batch 16, CFG dropout 0.15 and DDP on
two RTX 4090s. Split complete songs 85/15 into train/validation; do not create a
test split for this first research run. Validate every 5 epochs, checkpoint every
5 epochs, and early-stop after 5 validation checks without improvement. This is
checkpoint selection within one run, not a hyperparameter sweep.

Recreate the tested trainer checkout with:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git vendor/ACE-Step-1.5
git -C vendor/ACE-Step-1.5 checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git -C vendor/ACE-Step-1.5 apply --check \
  ../../patches/acestep-xl-validation-caption-variants.patch
git -C vendor/ACE-Step-1.5 apply \
  ../../patches/acestep-xl-validation-caption-variants.patch
```

The patch adds four-caption sampling within one sample, deterministic validation,
XL-Base CLI path validation, periodic global DDP validation loss, `best_val` saving
and early stopping. Its SHA-256 is
`f6f7e2b1a1aaa49db5573be67862579df2f4c758a973c7cc96c0e24b9ecaf257`.

Preprocess train part 0 on GPU 0 and part 1 on GPU 1, then preprocess validation and
merge with `scripts/merge_tensors.py`. The merged tensor count must equal the final
manifest count. Run the one-epoch smoke job before the 150-epoch job. Smoke passes
only when both GPUs work, loss is finite, validation runs, and the saved adapter can
be loaded again.

## 6. Release and backup

Upload only adapter/config/inference code and examples. Do not upload source audio,
separated audio, tensors, cookies or API tokens. Verify a clean re-download and one
successful inference before destroying Vast.

Select only three checkpoints: a middle checkpoint, `best_val`, and the last
checkpoint. Generate the same fixed prompts for each. Publish the LoRA/config/code
and permitted examples, never the copyrighted source dataset. Before stopping the
instance, download the release into a clean directory and produce one valid WAV.

## Resume safety

Long server jobs run through Supervisor and write checkpoints/manifests atomically.
After a reconnect, use `supervisorctl status`, then rerun the current script; completed
records are skipped. `/workspace` on this instance is not assumed persistent, so the
release, logs, manifests and training state must be uploaded before instance removal.
Run `edm-backup-metadata` at major gates to update the private Hugging Face dataset
`Bangchis/melodic-edm-training-metadata`. The uploader uses an explicit text-file
allowlist and fails closed if its secret scan finds a token or private key.

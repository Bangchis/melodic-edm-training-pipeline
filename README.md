# Melodic EDM training pipeline

Server pipeline from downloaded YouTube audio to an ACE-Step 1.5 LoRA.

The completed V1 state is tracked in [`STATUS.md`](STATUS.md). The active V2 state is
tracked separately in [`STATUS_V2.md`](STATUS_V2.md), with its fixed training contract
in [`TRAINING_V2.md`](TRAINING_V2.md) and Colab inference guide in
[`COLAB_INFERENCE_V2.md`](COLAB_INFERENCE_V2.md). The V1 executable order and recovery
rules remain in [`RUNBOOK.md`](RUNBOOK.md); exact tested V1 server packages remain in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

## Dataset policy

- Every successfully downloaded catalog record and canonical file is retained.
- Every valid catalog record remains a separate training sample, including records
  that point to identical audio content.
- Duplicate audio may share cached analysis results, but records are never collapsed,
  excluded, or down-weighted in the dataset.
- `sample_id` remains derived from `record_key` (`source__rank`), never only from `video_id`.
- Raw downloads are never deleted.

`deduplication_performed` is deliberately `false`. Fingerprints may be reported for
audit, but no active stage reads a `unique_manifest`, removes a repeated record or
changes its training weight.

The current Vast workspace is not a persistent volume. Back up artifacts before
recycling or destroying the instance.

See `RUNBOOK.md` and `scripts/`.

## ACE-Step pin and patch

Clone ACE-Step 1.5 at commit
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`, then apply
`patches/acestep-xl-validation-caption-variants.patch`. The patch is kept outside
`vendor/` so the published pipeline includes the exact tested trainer changes.

## Training V2

V2 keeps the same 231 validated audio records and performs a new XL-Base LoRA run
with rank 32 / alpha 32 / dropout 0.1. Each record has exactly three song-specific
annotation views (canonical, composition and production), fused from its own old prompt
and an independent audio reading. Only the single fused canonical prompt is embedded
and used for both training and validation. MOSS-Music-8B-Thinking is used only
as an audio annotation/listening model; it is never part of ACE-Step training.

Before preprocessing, MOSS first annotates without title, artist, MIR or prior
instrument claims. A multi-view verifier then checks each named sound source from
the old and new annotations with
two full-track prompts plus an intro/middle/late montage. Verified exact names are
preserved; absent claims are removed and unresolved names retain a qualified
`name-like` token instead of collapsing to an unrelated generic label. A stratified
compiler then fuses the old prompt for that exact song with the waveform facts.
A stratified listening gate checks the complete caption fidelity. Checkpoint acceptance also
requires every generated evaluation sample to reach at least 3/5 prompt alignment,
so clean but off-prompt audio is rejected. The run is capped at 20 epochs
and evaluates epochs 5/10/15/20 at the single fixed LoRA scale 0.5.

V2 produces both a validation-selected `best-val` adapter and a fresh
`final-all-data` adapter retrained on all 231 records. The private Hugging Face model
release includes a detailed Colab Pro notebook under
`notebooks/melodic_edm_core_v2_colab.ipynb`.

# Melodic EDM training pipeline

Server pipeline from downloaded YouTube audio to an ACE-Step 1.5 LoRA.

Current working state is tracked in [`STATUS.md`](STATUS.md). The executable order,
resume rules and verification gates are in [`RUNBOOK.md`](RUNBOOK.md).

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

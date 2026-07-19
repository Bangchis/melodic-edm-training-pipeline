#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
mkdir -p data_v2/openrouter_audio_audits
for spec in \
  "starling_edm__008:data/final_dataset/train/starling_edm__008.flac" \
  "thefatrat__020:data/final_dataset/validation/thefatrat__020.flac"; do
  sample_id="${spec%%:*}"
  audio_path="${spec#*:}"
  python3 -u scripts/judge_annotation_openrouter_audio.py \
    --audio "$audio_path" \
    --annotation "data_v2/annotations/${sample_id}.json" \
    --output "data_v2/openrouter_audio_audits/${sample_id}.json" \
    --sample-id "$sample_id"
done

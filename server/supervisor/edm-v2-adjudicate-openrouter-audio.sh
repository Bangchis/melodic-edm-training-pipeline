#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
mkdir -p data_v2/openrouter_audio_audits
for sample_id in starling_edm__008 thefatrat__020; do
  python3 -u scripts/judge_annotation_openrouter_audio.py \
    --audio "data/final_dataset/validation/${sample_id}.flac" \
    --annotation "data_v2/annotations/${sample_id}.json" \
    --output "data_v2/openrouter_audio_audits/${sample_id}.json" \
    --sample-id "$sample_id"
done

#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec python3 -u scripts/prepare_v2_audio_dataset.py \
  --project-root "$project" --repo-id Bangchis/melodic-edm-audio-v2

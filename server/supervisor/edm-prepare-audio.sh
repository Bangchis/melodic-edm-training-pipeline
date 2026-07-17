#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

cd /workspace/melodic_edm_training_pipeline
exec /usr/bin/python3 -u scripts/prepare_audio.py \
  --checkpoint /workspace/edm_audio_v5/state/checkpoint.csv \
  --selection /workspace/instrumental_edm_catalog_v5/catalog/selection_pre_server.csv \
  --project-root /workspace/melodic_edm_training_pipeline \
  --workers 4

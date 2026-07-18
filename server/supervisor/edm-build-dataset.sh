#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

cd /workspace/melodic_edm_training_pipeline
exec /usr/bin/python3 -u scripts/build_acestep_dataset.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --validation-ratio 0.15 \
  --seed 42 \
  --max-duration 240

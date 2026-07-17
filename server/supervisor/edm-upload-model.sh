#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/upload_model_hf.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --repo-id Bangchis/melodic-edm-core-v1

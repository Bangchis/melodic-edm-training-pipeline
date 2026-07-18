#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

ace=/workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5-v2
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
cd "$ace"
exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/upload_v2_model_hf.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --repo-id Bangchis/melodic-edm-core-v2

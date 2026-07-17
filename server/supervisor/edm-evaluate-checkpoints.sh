#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0
cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/evaluate_checkpoints.py \
  --project-root /workspace/melodic_edm_training_pipeline

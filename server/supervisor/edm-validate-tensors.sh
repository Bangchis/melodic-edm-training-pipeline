#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/validate_tensors.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --expected-total 231

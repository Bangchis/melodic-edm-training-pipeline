#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

set -a
. /workspace/.env
set +a

cd /workspace/melodic_edm_training_pipeline
exec vendor/ACE-Step-1.5/.venv/bin/python -u scripts/backup_metadata_hf.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --env-file /workspace/.env \
  --repo-id Bangchis/melodic-edm-training-metadata

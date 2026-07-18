#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/vendor/ACE-Step-1.5-v2/.venv/bin/python" -u scripts/upload_v2_completion_hf.py \
  --project-root "$project" \
  --repo-id Bangchis/melodic-edm-core-v2-training

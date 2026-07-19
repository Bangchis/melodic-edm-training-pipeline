#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
cd "$project"
exec "$ace/.venv/bin/python" -u scripts/upload_v2_evaluation_hf.py \
  --project-root "$project" \
  --repo-id Bangchis/melodic-edm-core-v2-r32-training

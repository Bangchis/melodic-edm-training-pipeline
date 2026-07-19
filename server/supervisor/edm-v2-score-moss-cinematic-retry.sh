#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0
project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venvs/moss-music/bin/python" -u scripts/score_v2_checkpoints_moss.py \
  --project-root "$project" \
  --evaluation-dir outputs/v2/cinematic-retry-evaluation

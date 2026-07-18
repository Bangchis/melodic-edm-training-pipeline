#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export CUDA_VISIBLE_DEVICES=1
project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venvs/moss-music/bin/python" -u scripts/verify_v2_audio_claims_moss.py \
  --project-root "$project" --shard-index 1 --num-shards 2 --attempts 5

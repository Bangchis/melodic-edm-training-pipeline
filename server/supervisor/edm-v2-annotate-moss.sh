#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:?GPU_INDEX is required}"
args=(
  --project-root "$project"
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
)
if [ -n "${LIMIT:-}" ]; then
  args+=(--limit "$LIMIT")
fi
cd "$project"
exec .venvs/moss-music/bin/python -u scripts/annotate_moss_music.py "${args[@]}"

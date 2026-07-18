#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:?GPU_INDEX is required}"
cd "$project"
exec "$ace/.venv/bin/python" -u -m acestep.training_v2.cli.train_fixed --preprocess \
  --dataset-json "$project/${DATASET_JSON:?DATASET_JSON is required}" \
  --tensor-output "$project/${TENSOR_OUTPUT:?TENSOR_OUTPUT is required}" \
  --checkpoint-dir "$project/checkpoints" \
  --model-variant xl_base \
  --max-duration 240 \
  --device cuda:0 \
  --precision bf16

#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0
cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u -m acestep.training_v2.cli.train_fixed \
  --preprocess \
  --audio-dir /workspace/melodic_edm_training_pipeline/data/final_dataset/validation \
  --dataset-json /workspace/melodic_edm_training_pipeline/data/dataset_validation.json \
  --tensor-output /workspace/melodic_edm_training_pipeline/data/tensors_validation \
  --checkpoint-dir /workspace/melodic_edm_training_pipeline/checkpoints \
  --model-variant xl_base --max-duration 240 --device cuda

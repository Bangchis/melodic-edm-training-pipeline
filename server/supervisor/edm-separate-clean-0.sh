#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0
cd /workspace/melodic_edm_training_pipeline
exec .venvs/separator/bin/python -u scripts/separate_vocals.py \
  --manifest data/vocal_clean_targets.jsonl \
  --preset instrumental_clean \
  --project-root /workspace/melodic_edm_training_pipeline \
  --part-index 0 --num-parts 2

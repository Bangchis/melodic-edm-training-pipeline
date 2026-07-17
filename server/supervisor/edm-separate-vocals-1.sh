#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=1
cd /workspace/melodic_edm_training_pipeline
exec .venvs/separator/bin/python -u scripts/separate_vocals.py \
  --manifest data/vocal_manifest.jsonl \
  --project-root /workspace/melodic_edm_training_pipeline \
  --part-index 1 --num-parts 2

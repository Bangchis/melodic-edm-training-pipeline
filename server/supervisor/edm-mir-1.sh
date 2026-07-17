#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=1
cd /workspace/melodic_edm_training_pipeline
exec .venvs/mir/bin/python -u scripts/analyze_mir.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --part-index 1 --num-parts 2 --device cuda

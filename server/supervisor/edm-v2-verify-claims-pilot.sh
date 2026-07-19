#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export CUDA_VISIBLE_DEVICES=0
project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venvs/moss-music/bin/python" -u scripts/verify_v2_audio_claims_moss.py \
  --project-root "$project" \
  --sample-id diversity__001 \
  --sample-id myomouse__009 \
  --sample-id xomu__001 \
  --sample-id xu_mengyuan__001 \
  --sample-id xu_mengyuan__040

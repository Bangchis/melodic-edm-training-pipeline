#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=1
project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venvs/moss-music/bin/python" -u scripts/audit_v2_annotation_fidelity_moss.py \
  --project-root "$project" --per-group 6

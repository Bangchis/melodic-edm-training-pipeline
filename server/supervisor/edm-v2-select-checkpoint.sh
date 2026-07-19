#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venvs/mir/bin/python" -u scripts/select_v2_checkpoint.py \
  --project-root "$project"

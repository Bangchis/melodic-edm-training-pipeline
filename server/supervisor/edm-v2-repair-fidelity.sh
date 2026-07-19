#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec python3 -u scripts/repair_v2_fidelity_failures_openrouter.py \
  --project-root "$project"

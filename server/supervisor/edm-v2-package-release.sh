#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

cd /workspace/melodic_edm_training_pipeline
exec python3 -u scripts/package_v2_release.py \
  --project-root /workspace/melodic_edm_training_pipeline

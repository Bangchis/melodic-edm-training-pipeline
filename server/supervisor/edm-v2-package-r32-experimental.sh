#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec python3 -u scripts/package_v2_r32_experimental.py --project-root "$project"

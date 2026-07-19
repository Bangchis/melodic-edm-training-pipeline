#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
cd "$project"
exec "$ace/.venv/bin/python" -u scripts/upload_v2_r32_experimental_hf.py --project-root "$project"

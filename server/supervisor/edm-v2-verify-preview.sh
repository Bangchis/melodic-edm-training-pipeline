#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
revision="$(python3 -c 'import json; print(json.load(open("/workspace/melodic_edm_training_pipeline/outputs/v2/checkpoint-evaluation/preview_upload_report.json"))["sha"])')"
cd "$project"
exec "$ace/.venv/bin/python" -u scripts/verify_v2_preview.py \
  --project-root "$project" --repo-id Bangchis/melodic-edm-core-v2 --revision "$revision"

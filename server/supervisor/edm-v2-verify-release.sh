#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
revision="$("$ace/.venv/bin/python" -c 'import json; print(json.load(open("/workspace/melodic_edm_training_pipeline/outputs/release/melodic-edm-core-v2/upload_report.json"))["sha"])')"

cd "$ace"
exec .venv/bin/python -u "${project}/scripts/verify_v2_release.py" \
  --project-root "${project}" \
  --repo-id Bangchis/melodic-edm-core-v2 \
  --revision "${revision}"

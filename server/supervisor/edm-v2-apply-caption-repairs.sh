#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
project=/workspace/melodic_edm_training_pipeline
cd "$project"
"$project/.venv/bin/python" -u scripts/apply_v2_annotation_repairs.py --project-root "$project"
"$project/.venv/bin/python" -u scripts/audit_v2_annotation_quality.py --project-root "$project"

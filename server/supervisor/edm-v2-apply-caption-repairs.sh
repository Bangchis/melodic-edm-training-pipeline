#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
project=/workspace/melodic_edm_training_pipeline
cd "$project"
python3 -u scripts/apply_v2_annotation_repairs.py --project-root "$project"
python3 -u scripts/audit_v2_annotation_quality.py --project-root "$project"

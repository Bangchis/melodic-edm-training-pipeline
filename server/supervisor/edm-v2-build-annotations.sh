#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
python3 -u scripts/merge_annotations_v2.py --project-root "$project"
exec python3 -u scripts/build_v2_dataset.py --project-root "$project"

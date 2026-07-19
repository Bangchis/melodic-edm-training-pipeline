#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
cd "$project"
python3 -u scripts/merge_v2_tensors.py --project-root "$project" \
  --destination data_v2/tensors_train --expected 196 \
  --source data_v2/tensors_train_part0 --source data_v2/tensors_train_part1
python3 -u scripts/merge_v2_tensors.py --project-root "$project" \
  --destination data_v2/tensors_validation --expected 35 \
  --source data_v2/tensors_validation_raw
python3 -u scripts/merge_v2_tensors.py --project-root "$project" \
  --destination data_v2/tensors_all --expected 231 \
  --source data_v2/tensors_train_part0 --source data_v2/tensors_train_part1 \
  --source data_v2/tensors_validation_raw
python3 -u scripts/build_v2_dedup_tensor_views.py --project-root "$project"
exec "$ace/.venv/bin/python" -u scripts/validate_v2_tensors.py \
  --project-root "$project"

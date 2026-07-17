#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
project=/workspace/melodic_edm_training_pipeline
cd "$project/vendor/ACE-Step-1.5"
exec .venv/bin/python -u "$project/scripts/annotate_qwen_local.py" \
  --project-root "$project" \
  --model-path "$project/checkpoints/Qwen2.5-Omni-7B"

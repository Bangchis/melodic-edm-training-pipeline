#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec "$project/.venv/bin/python" -u scripts/validate_v2_listening_quality.py \
  --project-root "$project" \
  --report outputs/v2/final-all-data/evaluation/listening_scores.json \
  --output outputs/v2/final-all-data/evaluation/listening_quality_report.json \
  --profile candidate \
  --reference-report outputs/v2/checkpoint-evaluation/listening_scores.json \
  --selection-report outputs/v2/checkpoint-evaluation/selection.json \
  --maximum-prompt-alignment-regression 0.34

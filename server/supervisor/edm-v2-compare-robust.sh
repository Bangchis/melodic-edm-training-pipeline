#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
cd "$project"
exec python3 -u scripts/compare_v2_seed_robustness.py \
  --project-root "$project" \
  --lora-report outputs/v2/robust-evaluation/listening_scores.json \
  --base-report outputs/v2/robust-base-evaluation/listening_scores.json \
  --output outputs/v2/robust-comparison.json \
  --minimum-score 3

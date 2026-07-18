#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0
project=/workspace/melodic_edm_training_pipeline
cd "$project"
"$project/.venvs/moss-music/bin/python" -u scripts/score_v2_checkpoints_moss.py \
  --project-root "$project" \
  --evaluation-dir outputs/v2/robust-evaluation
exec python3 -u scripts/summarize_v2_seed_robustness.py \
  --project-root "$project" \
  --report outputs/v2/robust-evaluation/listening_scores.json \
  --output outputs/v2/robust-evaluation/seed_robustness.json \
  --expected-seeds 5 \
  --minimum-score 3 \
  --minimum-pass-rate 0.8
